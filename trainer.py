"""
trainer.py

Trains a universal multi-stock directional classifier using XGBoost.

Labels (30-day forward return):
    0 = SELL  (return < -5%)
    1 = HOLD  (-5% <= return <= +5%)
    2 = BUY   (return >  +5%)

Bias-prevention measures
-------------------------
1. Chronological 70/15/15 split on DATE BOUNDARIES — never random.
2. Class balance via sklearn compute_sample_weight('balanced').
3. Regularised XGBoost (max_depth=3, alpha=2, lambda=2, gamma=1.0,
   min_child_weight=20, subsample=0.75, colsample_bytree=0.75).
4. Early stopping (50 rounds on validation loss).
5. Walk-forward validation: 2-year rolling train, 3-month test window.
6. Automatic bias detection after training.
"""

import datetime
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import backtest
import database
from encoders import SECTOR_ENCODER_PATH

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature set — normalized / ratio features only.
# Absolute-price columns (raw SMA, EMA, BB bands, OBV, VWAP) are excluded
# because their dollar magnitudes do not generalise across stocks.
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    # Momentum oscillators (0–100 or % scale)
    "rsi_14", "rsi_28",
    "stoch_k", "stoch_d",
    "williams_r",
    "roc_10",
    "macd_hist",          # histogram (divergence from signal) is more informative
                          # than raw macd/signal for cross-stock use
    # Volatility & bands (normalised)
    "bb_width",           # (upper - lower) / middle  — % bandwidth
    "bb_pct_b",           # (close - lower) / (upper - lower), 0–1
    "atr_14",             # absolute ATR; tree splits handle scale differences
    # Volume
    "volume_ratio",       # daily volume / 20-day avg volume
    # Return series (already %)
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",     # 20-day rolling std of daily returns
    # Position features (%)
    "pct_from_52w_high",  # (close - expanding_max) / expanding_max  <= 0
    "dist_from_sma50",    # (close - sma50) / sma50
    # Regime signal
    "golden_cross",       # 1 if sma50 > sma200 else 0
    # Sector / size context
    "sector_encoded",     # integer-encoded GICS sector (stable cross-stock anchor)
    # Macro / market-regime features (same value for every ticker on a given date)
    "fed_funds_rate",     # FRED FEDFUNDS — current rate environment
    "treasury_10y",       # FRED GS10 — long-end rate
    "yield_curve_spread", # 10Y - fed funds; negative = inversion / recession signal
    "vix",                # daily VIX close — market fear gauge
    "vix_sma20",          # 20-day VIX average — baseline volatility level
    "vix_regime",         # 1 if vix > vix_sma20 (rising vol), else 0
    "sp500_return_20d",   # S&P 500 trailing 20-day return — bull/bear momentum
    "sp500_above_sma50",  # 1 if S&P 500 above its 50-day SMA — trend regime
    "unemployment_rate",  # US unemployment rate (FRED UNRATE, monthly fwd-filled)
    "cpi_yoy",            # CPI year-over-year % change (inflation signal)
    "spread_10y2y",       # 10Y-2Y Treasury spread (canonical recession indicator)
    # EDGAR 8-K filing features (computed by edgar_collector.py)
    "days_since_8k",      # calendar days since last material event filing
    "count_8k_90d",       # count of 8-K filings in prior 90 days
    # Earnings surprise features (computed by earnings_collector.py)
    "earnings_surprise_pct",  # (actual - estimate) / abs(estimate) — PEAD signal
    "days_since_earnings",    # calendar days since last earnings report
    "earnings_beat",          # 1 if last surprise_pct > 0, else 0
    # Sentiment (computed by feature_engine.py from sentiment_collector data)
    "sentiment_7d_avg",       # 7-day rolling average of news_sentiment score
    # Sector-relative features (computed post-hoc in feature_engine.py)
    # Each value = raw feature − sector mean for that date (same-day cross-section)
    "rsi_14_vs_sector",       # RSI-14 deviation from sector mean
    "return_5d_vs_sector",    # 5-day return deviation from sector mean
    "return_20d_vs_sector",   # 20-day return deviation from sector mean
    "macd_hist_vs_sector",    # MACD histogram deviation from sector mean
    "vol_20d_vs_sector",      # 20-day volatility deviation from sector mean
    "dist_sma50_vs_sector",   # distance from SMA-50 deviation from sector mean
]

LABEL_SELL, LABEL_HOLD, LABEL_BUY = 0, 1, 2
LABEL_NAMES = {LABEL_SELL: "SELL", LABEL_HOLD: "HOLD", LABEL_BUY: "BUY"}

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "universal_model.joblib"


# ---------------------------------------------------------------------------

class UniversalStockModel:
    """
    Multi-class XGBoost classifier for 30-day directional stock prediction.

    Usage
    -----
    >>> m = UniversalStockModel()
    >>> m.fit()           # loads data from DB, trains, evaluates
    >>> m.save()
    >>> m2 = UniversalStockModel.load()
    >>> preds = m2.predict(feature_df)
    """

    def __init__(self) -> None:
        self.model:          XGBClassifier | None = None
        self.feature_cols:   list[str]            = FEATURE_COLS
        self.split_dates:    dict                 = {}
        self.metrics:        dict                 = {}
        self.holdout_df:     pd.DataFrame | None  = None
        self.feature_bounds: dict | None          = None
        self.buy_threshold:         float            = 0.33   # fallback scalar
        self.buy_threshold_dict:    dict[int, float] = {0: 0.33, 1: 0.45}  # {regime: threshold}
        self.buy_top_fraction:      float            = 0.10   # cross-sectional top-K% for BUY
        self.sector_encoder_path:   str | None       = None

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_data(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """
        Inner-join features + labels tables.  Returns a DataFrame sorted
        ascending by date with columns: ticker, date, <FEATURE_COLS>,
        label, forward_return.
        """
        # ---- point-in-time fundamentals guard --------------------------------
        # fundamental_metadata holds live yfinance values (not historical).
        # If any of those column names appear in FEATURE_COLS the model would
        # train on 2026 data to predict 2021-2025 outcomes — pure look-ahead.
        with database.connection() as conn:
            fund_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(fundamental_metadata)")
            }
        _structural = {"ticker", "fetch_date", "is_point_in_time"}
        fund_cols -= _structural
        for col in self.feature_cols:
            if col in fund_cols:
                raise RuntimeError(
                    f"BIAS GUARD: fundamental column '{col}' found in FEATURE_COLS. "
                    "Remove it — yfinance fundamentals are not point-in-time and "
                    "will introduce look-ahead bias into the model."
                )
        # ----------------------------------------------------------------------

        feat_select = ", ".join(f"f.{c}" for c in FEATURE_COLS)
        base_sql = f"""
            SELECT f.ticker, f.date,
                   {feat_select},
                   l.label, l.forward_return
            FROM   features f
            JOIN   labels   l ON l.ticker = f.ticker AND l.date = f.date
        """
        with database.connection() as conn:
            if tickers:
                ph   = ",".join("?" * len(tickers))
                rows = conn.execute(
                    base_sql + f" WHERE f.ticker IN ({ph}) ORDER BY f.date, f.ticker",
                    tickers,
                ).fetchall()
            else:
                rows = conn.execute(
                    base_sql + " ORDER BY f.date, f.ticker"
                ).fetchall()

        df = pd.DataFrame([dict(r) for r in rows])
        log.info(
            "Loaded %d rows covering %d tickers  (%s to %s).",
            len(df), df["ticker"].nunique(),
            df["date"].min(), df["date"].max(),
        )

        # Drop the first 20 rows per ticker (unstabilized rolling windows)
        df = df.sort_values(["ticker", "date"])
        df = df[df.groupby("ticker").cumcount() >= 20].reset_index(drop=True)
        log.info(
            "After masking first 20 rows per ticker: %d rows remain.",
            len(df),
        )
        return df

    # -----------------------------------------------------------------------
    # Chronological split — NEVER random
    # -----------------------------------------------------------------------

    def chronological_split(
        self,
        df: pd.DataFrame,
        train_frac:   float = 0.70,
        val_frac:     float = 0.15,
        holdout_frac: float = 0.05,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split on UNIQUE DATE BOUNDARIES so every row for a given date
        lands in exactly one partition.

        Timeline order: [train 70%][val 15%][test 10%][holdout 5%]

        When holdout_frac > 0, the holdout is carved from the very END
        of the date range before train/val/test boundaries are computed.
        The holdout is returned as the 4th element and must NEVER be used
        for any tuning decision (thresholds, hyperparameters, features).
        """
        unique_dates = sorted(df["date"].unique())
        n = len(unique_dates)

        # Carve holdout from the end first
        if holdout_frac > 0:
            holdout_start_idx = int(n * (1.0 - holdout_frac))
            holdout_start_date = unique_dates[holdout_start_idx]
            working_dates = unique_dates[:holdout_start_idx]
        else:
            holdout_start_date = None
            working_dates = unique_dates

        nw = len(working_dates)
        # Rescale train/val fractions relative to the working (non-holdout) portion.
        # E.g. holdout=0.05 → train=70/95, val=15/95, test=10/95 of working dates.
        total_non_holdout = 1.0 - holdout_frac if holdout_frac > 0 else 1.0
        eff_train = train_frac / total_non_holdout
        eff_val   = val_frac   / total_non_holdout

        train_end_date = working_dates[int(nw * eff_train) - 1]
        val_end_date   = working_dates[int(nw * (eff_train + eff_val)) - 1]

        train   = df[df["date"] <= train_end_date]
        val     = df[(df["date"] >  train_end_date) & (df["date"] <= val_end_date)]
        test    = df[(df["date"] >  val_end_date) & (
                      (df["date"] <  holdout_start_date) if holdout_start_date else True
                  )]
        holdout = df[df["date"] >= holdout_start_date] if holdout_start_date else df.iloc[0:0]

        self.split_dates = {
            "train_end":  train_end_date,
            "val_end":    val_end_date,
            "test_start": test["date"].min() if len(test) else "",
            "test_end":   test["date"].max() if len(test) else "",
        }

        log.info(
            "Split — train: %d rows (up to %s) | "
            "val: %d rows (%s–%s) | "
            "test: %d rows (%s–%s)",
            len(train), train_end_date,
            len(val),   val["date"].min() if len(val) else "",  val_end_date,
            len(test),  test["date"].min() if len(test) else "", test["date"].max() if len(test) else "",
        )
        if holdout_start_date:
            log.info(
                "Holdout: %d rows (%s to %s) — DO NOT USE FOR ANY TUNING DECISION.",
                len(holdout),
                holdout["date"].min() if len(holdout) else "",
                holdout["date"].max() if len(holdout) else "",
            )
        return train, val, test, holdout

    # -----------------------------------------------------------------------
    # XGBoost factory
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # BUY threshold calibration helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _apply_buy_threshold(
        proba: np.ndarray,
        threshold: "float | dict[int, float]",
        regime_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply a per-class threshold for BUY.

        # buy_threshold path is unused at inference; kept for offline analysis only.
        # Inference uses _apply_buy_percentile via predict() and predict_proba().
        If P(BUY) >= threshold → predict BUY.
        Otherwise → argmax of (SELL, HOLD) probabilities for the other two.

        threshold may be:
          - float: single global threshold
          - dict {regime_value: threshold}: per-regime thresholds.
            regime_arr must be provided (same length as proba rows).
        """
        buy_p  = proba[:, LABEL_BUY]
        sell_p = proba[:, LABEL_SELL]
        hold_p = proba[:, LABEL_HOLD]
        base   = np.where(sell_p > hold_p, LABEL_SELL, LABEL_HOLD)

        if isinstance(threshold, dict) and regime_arr is not None:
            # Build a per-row threshold array from the regime values
            thr_arr = np.full(len(proba), threshold.get(0, 0.33))
            for regime_val, t in threshold.items():
                thr_arr[regime_arr == regime_val] = t
            return np.where(buy_p >= thr_arr, LABEL_BUY, base)
        else:
            t = float(threshold) if not isinstance(threshold, float) else threshold
            return np.where(buy_p >= t, LABEL_BUY, base)

    @staticmethod
    def _apply_buy_percentile(proba: np.ndarray, top_fraction: float) -> np.ndarray:
        """
        Cross-sectional BUY/SELL signals:
          - Top    `top_fraction` by P(BUY)  → BUY
          - Bottom `top_fraction` by P(BUY)  → SELL  (symmetric, regime-invariant)
          - Remainder                         → HOLD

        Using P(BUY) for both tails means the same axis drives both signals,
        which is consistent: the least-bullish stocks are flagged as SELL.
        This avoids the structural bug where P(SELL) < P(HOLD) in bull markets
        causes SELL signals to disappear entirely.
        """
        buy_p = proba[:, LABEL_BUY]

        if len(buy_p) == 0 or top_fraction <= 0:
            return np.full(len(buy_p), LABEL_HOLD, dtype=int)

        buy_cutoff  = np.percentile(buy_p, (1.0 - top_fraction) * 100.0)
        sell_cutoff = np.percentile(buy_p, top_fraction * 100.0)

        preds = np.full(len(buy_p), LABEL_HOLD, dtype=int)
        preds[buy_p >= buy_cutoff]  = LABEL_BUY
        preds[buy_p <= sell_cutoff] = LABEL_SELL
        # If BUY and SELL cutoffs overlap (tiny universe), BUY takes priority
        preds[(buy_p >= buy_cutoff) & (buy_p <= sell_cutoff)] = LABEL_BUY
        return preds

    def _calibrate_buy_fraction(
        self,
        val_proba: np.ndarray,
        val_returns: np.ndarray,
        min_fraction: float = 0.10,
        max_fraction: float = 0.25,
    ) -> float:
        """
        Grid-search the top-K fraction on the validation set.

        Objective: maximise (avg_return - bah_return) × fraction, which
        jointly rewards edge quality AND coverage (total portfolio gain).

        Constrained to [min_fraction, max_fraction] to ensure enough BUY
        signals for meaningful recall without over-selecting.

        Falls back to min_fraction if no fraction beats buy-and-hold.
        """
        valid_mask = ~np.isnan(val_returns)
        if valid_mask.sum() < 100:
            return min_fraction
        bah_ret = float(val_returns[valid_mask].mean())

        best_frac  = None
        best_score = -np.inf

        for frac in np.arange(min_fraction, max_fraction + 0.001, 0.01):
            pred     = self._apply_buy_percentile(val_proba, frac)
            buy_mask = (pred == LABEL_BUY) & valid_mask
            n_sig    = int(buy_mask.sum())
            if n_sig == 0:
                continue
            avg_ret = float(val_returns[buy_mask].mean())
            edge    = avg_ret - bah_ret
            # Objective: edge × fraction — rewards quality AND coverage
            score = edge * frac
            if score > best_score:
                best_score = score
                best_frac  = frac

        if best_frac is not None and best_score > 0:
            pred  = self._apply_buy_percentile(val_proba, best_frac)
            n_sig = int(((pred == LABEL_BUY) & valid_mask).sum())
            avg_r = float(val_returns[(pred == LABEL_BUY) & valid_mask].mean())
            log.info(
                "Cross-sectional BUY fraction calibrated: %.0f%%  "
                "(val avg_ret=%.4f vs bah=%.4f, n=%d, score=%.6f)",
                best_frac * 100, avg_r, bah_ret, n_sig, best_score,
            )
            return float(best_frac)

        log.warning(
            "No fraction beat bah (%.4f); using min_fraction %.0f%%.",
            bah_ret, min_fraction * 100,
        )
        return min_fraction

    def _calibrate_buy_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_returns: np.ndarray | None = None,
        val_regime: np.ndarray | None = None,
        min_signals: int = 200,
    ) -> dict[int, float]:
        """
        Grid-search BUY probability thresholds on the VALIDATION set.

        # buy_threshold path is unused at inference; kept for offline analysis only.
        # Inference uses _apply_buy_percentile via predict() and predict_proba().

        Returns a dict {regime_value: threshold} with separate thresholds
        for each market regime (sp500_above_sma50: 0=bear, 1=bull).

        Objective: maximise average 30-day forward return of flagged-BUY
        positions within each regime sub-set, subject to >= min_signals.
        Falls back to precision-based (>= 35%) if returns unavailable.

        Calibrated on val set only — zero look-ahead when applied to test.
        """
        proba = self.model.predict_proba(X_val)
        buy_p = proba[:, LABEL_BUY]

        regimes = [0, 1] if val_regime is not None else [None]
        result: dict[int, float] = {}

        for regime in regimes:
            if regime is None:
                mask = np.ones(len(X_val), dtype=bool)
                label = "all"
                default_t = 0.40
            else:
                mask = (val_regime == regime)
                label = f"regime={regime}"
                default_t = 0.33 if regime == 0 else 0.45

            n_mask = int(mask.sum())
            if n_mask < min_signals:
                log.warning("Not enough val rows for %s (%d); using default %.2f",
                            label, n_mask, default_t)
                result[regime] = default_t
                continue

            # ---- return-based calibration ----------------------------------
            best_t   = None
            best_ret = -np.inf

            if val_returns is not None:
                valid   = mask & ~np.isnan(val_returns)
                bah_ret = float(val_returns[valid].mean()) if valid.any() else 0.0

                for t in np.arange(0.15, 0.71, 0.01):
                    buy_mask = (buy_p >= t) & valid
                    n_sig    = int(buy_mask.sum())
                    if n_sig < min_signals:
                        continue
                    avg_ret = float(val_returns[buy_mask].mean())
                    if avg_ret > best_ret:
                        best_ret = avg_ret
                        best_t   = float(t)

                if best_t is not None and best_ret > bah_ret:
                    n_sig = int(((buy_p >= best_t) & valid).sum())
                    log.info(
                        "BUY threshold [%s]: %.2f  "
                        "(val avg_ret=%.4f vs bah=%.4f, n=%d)",
                        label, best_t, best_ret, bah_ret, n_sig,
                    )
                    result[regime] = best_t
                    continue

                log.warning(
                    "[%s] Return-based calibration found no threshold > bah (%.4f); "
                    "trying precision fallback.", label, bah_ret if val_returns is not None else 0.0,
                )

            # ---- precision-based fallback ----------------------------------
            y_sub  = y_val[mask]
            buy_true   = y_sub == LABEL_BUY
            n_buy_true = int(buy_true.sum())
            best_prec_t = None
            best_prec   = 0.0

            if n_buy_true > 0:
                for t in np.arange(0.15, 0.71, 0.01):
                    buy_pred = (buy_p[mask]) >= t
                    n_pred   = int(buy_pred.sum())
                    if n_pred == 0:
                        continue
                    tp        = int((buy_pred & buy_true).sum())
                    precision = tp / n_pred
                    recall    = tp / n_buy_true
                    if precision < 0.35 or recall < 0.05:
                        continue
                    if precision > best_prec:
                        best_prec   = precision
                        best_prec_t = float(t)

            if best_prec_t is not None:
                log.info("BUY threshold [%s] (precision fallback): %.2f  (prec=%.3f)",
                         label, best_prec_t, best_prec)
                result[regime] = best_prec_t
            else:
                log.warning("BUY threshold [%s]: using default %.2f", label, default_t)
                result[regime] = default_t

        return result

    def _make_xgb(
        self,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> XGBClassifier:
        return XGBClassifier(
            objective              = "multi:softprob",
            num_class              = 3,
            n_estimators           = n_estimators,
            # --- regularisation ------------------------------------------
            max_depth              = 4,      # extra level for macro×technical interactions
            reg_alpha              = 2.0,    # L1 — restored to original for sparsity
            reg_lambda             = 2.0,    # L2 — restored to original for shrinkage
            min_child_weight       = 15,     # moderate — prevents single-tree dominance
            gamma                  = 0.5,    # moderate pruning — requires meaningful splits
            subsample              = 0.75,   # row subsampling per tree
            colsample_bytree       = 0.75,   # column subsampling per tree
            # --- hardware / speed ----------------------------------------
            learning_rate          = 0.02,   # slower lr — forces more trees, reduces best_iter=1
            tree_method            = "hist",
            n_jobs                 = -1,
            eval_metric            = "mlogloss",
            early_stopping_rounds  = early_stopping_rounds,   # XGB >= 1.6
            random_state           = 42,
            verbosity              = 0,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _Xy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[self.feature_cols].values.astype(float)   # NaN handled by XGB
        y = df["label"].values.astype(int)
        return X, y

    # -----------------------------------------------------------------------
    # Main training entry point
    # -----------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame | None = None,
        tickers: list[str] | None = None,
    ) -> "UniversalStockModel":
        """
        Full pipeline: load → split → train → evaluate → bias-check →
        feature importances → return comparison → walk-forward validation.
        """
        if df is None:
            df = self.load_data(tickers)

        # holdout_df is reserved for final production validation only.
        # Never use it to tune thresholds, features, or hyperparameters.
        train_df, val_df, test_df, holdout_df = self.chronological_split(df)
        self.holdout_df = holdout_df

        X_train, y_train = self._Xy(train_df)
        X_val,   y_val   = self._Xy(val_df)
        X_test,  y_test  = self._Xy(test_df)

        # ---- class balance -------------------------------------------------
        train_weights = compute_sample_weight("balanced", y_train)

        # ---- train ---------------------------------------------------------
        log.info(
            "Fitting XGBoost on %d train rows "
            "(early_stopping_rounds=50 on %d val rows)…",
            len(X_train), len(X_val),
        )
        self.model = self._make_xgb(n_estimators=1000, early_stopping_rounds=50)
        self.model.fit(
            X_train, y_train,
            sample_weight = train_weights,
            eval_set      = [(X_val, y_val)],
            verbose       = False,
        )
        log.info(
            "Best iteration: %d  |  best val mlogloss: %.4f",
            self.model.best_iteration,
            self.model.best_score,
        )

        # ---- calibrate BUY fraction (cross-sectional, zero look-ahead) -----
        val_returns = val_df["forward_return"].values.astype(float)
        val_regime  = val_df["sp500_above_sma50"].values.astype(int)

        train_proba = self.model.predict_proba(X_train)
        val_proba   = self.model.predict_proba(X_val)
        test_proba  = self.model.predict_proba(X_test)

        self.buy_top_fraction = self._calibrate_buy_fraction(val_proba, val_returns)

        # Also calibrate per-regime absolute thresholds as reference (not used for main eval)
        self.buy_threshold_dict = self._calibrate_buy_threshold(
            X_val, y_val, val_returns, val_regime
        )
        self.buy_threshold = self.buy_threshold_dict.get(0, 0.33)

        # ---- evaluate with cross-sectional percentile ----------------------
        train_pred = self._apply_buy_percentile(train_proba, self.buy_top_fraction)
        val_pred   = self._apply_buy_percentile(val_proba,   self.buy_top_fraction)
        test_pred  = self._apply_buy_percentile(test_proba,  self.buy_top_fraction)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc   = accuracy_score(y_val,   val_pred)
        test_acc  = accuracy_score(y_test,  test_pred)

        self.metrics.update({
            "train_accuracy": train_acc,
            "val_accuracy":   val_acc,
            "test_accuracy":  test_acc,
            "best_iteration": self.model.best_iteration,
        })

        log.info(
            "Accuracy — train: %.3f  |  val: %.3f  |  test: %.3f",
            train_acc, val_acc, test_acc,
        )
        log.info(
            "Classification report (test set):\n%s",
            classification_report(y_test, test_pred,
                                  target_names=["SELL", "HOLD", "BUY"]),
        )

        # ---- feature bounds (min/max per feature from training data) -------
        self.feature_bounds = {
            col: (float(train_df[col].min()), float(train_df[col].max()))
            for col in self.feature_cols
        }

        # ---- post-training checks ------------------------------------------
        self._detect_bias(y_train, y_test, test_pred, train_acc, test_acc)
        self._print_feature_importances()
        self._compare_returns(test_df, test_pred)

        # ---- walk-forward --------------------------------------------------
        log.info("=== Walk-Forward Validation ===")
        self.walk_forward_validate(df)

        return self

    # -----------------------------------------------------------------------
    # Bias detection
    # -----------------------------------------------------------------------

    def _detect_bias(
        self,
        y_train:   np.ndarray,
        y_test:    np.ndarray,
        test_pred: np.ndarray,
        train_acc: float,
        test_acc:  float,
    ) -> None:
        log.info("=== Bias Detection ===")
        clean = True

        # 1. Suspiciously high test accuracy → possible data leakage
        if test_acc > 0.70:
            log.warning(
                "BIAS [leakage?]  test accuracy %.1f%% > 70%%. "
                "Check for forward-looking features.",
                test_acc * 100,
            )
            clean = False

        # 2. Large train-test gap → overfitting
        gap = train_acc - test_acc
        if gap > 0.20:
            log.warning(
                "BIAS [overfit]   train-test gap %.1f%% > 20%%. "
                "Consider stronger regularisation or less data.",
                gap * 100,
            )
            clean = False

        # 3. Any class predicted < 10% of the time → imbalance not addressed
        n = len(test_pred)
        for label_id, name in LABEL_NAMES.items():
            frac = (test_pred == label_id).sum() / n
            if frac < 0.10:
                log.warning(
                    "BIAS [imbalance] class %-4s predicted only %.1f%% of the time "
                    "— model may be collapsed onto majority classes.",
                    name, frac * 100,
                )
                clean = False

        if clean:
            log.info("No bias issues detected.")

    # -----------------------------------------------------------------------
    # Feature importances
    # -----------------------------------------------------------------------

    def _print_feature_importances(self) -> None:
        pairs = sorted(
            zip(self.feature_cols, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        log.info("Top 10 feature importances:")
        for rank, (feat, imp) in enumerate(pairs[:10], 1):
            bar = "#" * int(imp * 200)
            log.info("  %2d. %-22s  %.4f  %s", rank, feat, imp, bar)

    # -----------------------------------------------------------------------
    # Return comparison
    # -----------------------------------------------------------------------

    def _compare_returns(
        self,
        test_df:   pd.DataFrame,
        test_pred: np.ndarray,
    ) -> None:
        fwd   = test_df["forward_return"].values.astype(float)
        mask  = ~np.isnan(fwd)
        dates = test_df["date"].values[mask]
        fwd, pred = fwd[mask], test_pred[mask]

        bah_ret   = fwd.mean()

        buy_mask  = pred == LABEL_BUY
        buy_ret   = fwd[buy_mask].mean()   if buy_mask.any()  else float("nan")

        hold_mask = pred != LABEL_SELL
        hold_ret  = fwd[hold_mask].mean()  if hold_mask.any() else float("nan")

        log.info("=== Return Comparison (test period, avg 30-day forward return) ===")
        log.info(
            "  Buy-and-hold (always invested):  %+.2f%%  (%d positions)",
            bah_ret * 100, len(fwd),
        )
        log.info(
            "  Model — BUY signals only:        %+.2f%%  (%d positions)",
            buy_ret  * 100, int(buy_mask.sum()),
        )
        log.info(
            "  Model — avoid-SELL strategy:     %+.2f%%  (%d positions)",
            hold_ret * 100, int(hold_mask.sum()),
        )

        edge = buy_ret - bah_ret if not np.isnan(buy_ret) else float("nan")
        if not np.isnan(edge):
            log.info(
                "  BUY-signal edge vs buy-and-hold: %+.2f%%", edge * 100
            )

        self.metrics.update({
            "bah_return":        float(bah_ret),
            "model_buy_return":  float(buy_ret),
            "avoid_sell_return": float(hold_ret),
        })

        # Group returns by date for block bootstrap (preserves cross-sectional correlation)
        buy_by_date: dict = {}
        all_by_date: dict = {}
        for d, r, p in zip(dates, fwd, pred):
            key = str(d)
            all_by_date.setdefault(key, []).append(float(r))
            if p == LABEL_BUY:
                buy_by_date.setdefault(key, []).append(float(r))

        # ---- viability verdict via backtest module -------------------------
        backtest.run({
            "buy_edge_gross":      float(buy_ret) if not np.isnan(buy_ret) else float("nan"),
            "bah_return":          float(bah_ret),
            "buy_returns":         fwd[buy_mask].tolist() if buy_mask.any() else [],
            "all_returns":         fwd.tolist(),
            "buy_returns_by_date": buy_by_date,
            "all_returns_by_date": all_by_date,
        })

    # -----------------------------------------------------------------------
    # Walk-forward validation
    # -----------------------------------------------------------------------

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        train_years: float = 2.0,
        test_months: int   = 3,
    ) -> list[dict]:
        """
        Rolling walk-forward:
          - Training window: last 2 years (slides forward each iteration)
          - Test window:     next 3 months
          - Step:            3 months

        For early stopping inside each window, the last 15% of the training
        dates are used as a within-window validation set (chronological).
        """
        # Restrict to only required columns early to reduce peak memory
        keep_cols = ["date", "_dt"] + self.feature_cols + ["label", "forward_return"]
        df = df[[c for c in keep_cols if c != "_dt"]].copy()
        df["_dt"] = pd.to_datetime(df["date"])
        unique_dts = sorted(df["_dt"].unique())

        train_delta = datetime.timedelta(days=int(train_years * 365))
        test_delta  = datetime.timedelta(days=test_months * 30)

        results: list[dict] = []
        cursor = unique_dts[0]
        window_num = 0

        while True:
            train_start = cursor
            train_end   = cursor + train_delta
            test_end    = train_end + test_delta

            # Stop when the test window would exceed available data
            if test_end > unique_dts[-1] + datetime.timedelta(days=1):
                break

            wf_train_all = df[
                (df["_dt"] >= train_start) & (df["_dt"] < train_end)
            ]
            wf_test = df[
                (df["_dt"] >= train_end) & (df["_dt"] < test_end)
            ]

            if len(wf_train_all) < 200 or len(wf_test) < 30:
                cursor += test_delta
                continue

            # Chronological val split from the end of the training block
            inner_dates = sorted(wf_train_all["_dt"].unique())
            val_cutoff  = inner_dates[int(len(inner_dates) * 0.85)]
            wf_train = wf_train_all[wf_train_all["_dt"] <  val_cutoff]
            wf_val   = wf_train_all[wf_train_all["_dt"] >= val_cutoff]

            X_tr, y_tr = self._Xy(wf_train)
            X_vl, y_vl = self._Xy(wf_val)
            X_te, y_te = self._Xy(wf_test)

            # Skip if any split lacks class variety
            if min(len(np.unique(y_tr)), len(np.unique(y_te))) < 2:
                cursor += test_delta
                continue

            w_tr = compute_sample_weight("balanced", y_tr)
            m    = self._make_xgb(n_estimators=500, early_stopping_rounds=50)
            m.fit(
                X_tr, y_tr,
                sample_weight = w_tr,
                eval_set      = [(X_vl, y_vl)],
                verbose       = False,
            )

            preds = m.predict(X_te)
            acc   = accuracy_score(y_te, preds)
            window_num += 1

            # Capture sizes before freeing arrays
            n_tr, n_vl, n_te = len(X_tr), len(X_vl), len(X_te)
            best_iter = m.best_iteration

            # Free arrays now — they can be large on a 500-stock dataset
            del X_tr, y_tr, X_vl, y_vl, X_te, y_te, w_tr, preds, m

            res = {
                "window":      window_num,
                "train_start": train_start.date().isoformat(),
                "train_end":   (train_end - datetime.timedelta(days=1)).date().isoformat(),
                "test_start":  train_end.date().isoformat(),
                "test_end":    (test_end  - datetime.timedelta(days=1)).date().isoformat(),
                "train_rows":  n_tr,
                "val_rows":    n_vl,
                "test_rows":   n_te,
                "test_acc":    acc,
                "best_iter":   best_iter,
            }
            results.append(res)

            log.info(
                "  WF #%02d  train %s→%s  test %s→%s  "
                "acc=%.3f  best_iter=%d  (tr/va/te=%d/%d/%d)",
                window_num,
                res["train_start"], res["train_end"],
                res["test_start"],  res["test_end"],
                acc, best_iter,
                n_tr, n_vl, n_te,
            )

            cursor += test_delta

        if results:
            accs = [r["test_acc"] for r in results]
            log.info(
                "Walk-forward summary — %d windows | "
                "acc:  mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                len(results),
                np.mean(accs), np.std(accs), np.min(accs), np.max(accs),
            )
            self.metrics["wf_mean_accuracy"] = float(np.mean(accs))
            self.metrics["wf_std_accuracy"]  = float(np.std(accs))
            self.metrics["wf_windows"]       = len(results)
        else:
            log.warning("Walk-forward: no complete windows found in the data range.")

        del df
        return results

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return class predictions using cross-sectional top/bottom-K% BUY/SELL rule."""
        if self.model is None:
            raise RuntimeError("Model not trained — call fit() first.")
        proba = self.model.predict_proba(df[self.feature_cols].values.astype(float))
        return self._apply_buy_percentile(proba, self.buy_top_fraction)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return class probabilities (n_rows × 3)."""
        if self.model is None:
            raise RuntimeError("Model not trained — call fit() first.")
        return self.model.predict_proba(df[self.feature_cols].values.astype(float))

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        """Save model + metadata with joblib."""
        save_path = Path(path) if path else MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model":                self.model,
                "feature_cols":         self.feature_cols,
                "split_dates":          self.split_dates,
                "metrics":              self.metrics,
                "feature_bounds":       self.feature_bounds,
                "buy_threshold":        self.buy_threshold,
                "buy_threshold_dict":   self.buy_threshold_dict,
                "buy_top_fraction":     self.buy_top_fraction,
                "sector_encoder_path":  str(SECTOR_ENCODER_PATH),
            },
            save_path,
        )
        log.info("Saved → %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "UniversalStockModel":
        """Load a previously saved model."""
        load_path = Path(path) if path else MODEL_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"No model file at {load_path}")
        payload = joblib.load(load_path)
        inst                = cls()
        inst.model          = payload["model"]
        inst.feature_cols   = payload["feature_cols"]
        inst.split_dates    = payload.get("split_dates", {})
        inst.metrics        = payload.get("metrics", {})
        inst.feature_bounds = payload.get("feature_bounds")
        inst.buy_threshold        = payload.get("buy_threshold", 0.33)
        inst.buy_threshold_dict   = payload.get("buy_threshold_dict", {0: 0.33, 1: 0.45})
        inst.buy_top_fraction     = payload.get("buy_top_fraction", 0.10)
        inst.sector_encoder_path  = payload.get("sector_encoder_path")
        log.info("Loaded ← %s", load_path)
        return inst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    m = UniversalStockModel()
    m.fit(tickers=_tickers)
    m.save()
