"""
portfolio_backtest.py

NAV-based portfolio backtest using realized daily close-to-close returns.

Key design decisions
--------------------
* Daily returns come from prices.close pct_change(), never from labels.forward_return.
  The 30-day overlapping forward_return must NOT be compounded daily — that is the
  bug that produced +457,877% CAGR in the old compare_models.py.
* At every rebalance date t, only features and prices with date <= t are used.
* SPY is loaded from the prices table first; yfinance is the fallback.
  A flat 0% benchmark is never silently substituted.
* Round-trip cost is applied proportionally to turnover on every rebalance.
* Slippage is uniform[0, slippage_max] × turnover (fixed seed).
* Nothing is refitted on holdout; models are loaded from disk only.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

import database

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LABEL_BUY lazy import — avoids hard dependency on trainer at module load
# ---------------------------------------------------------------------------
_LABEL_BUY: Optional[int] = None


def _get_label_buy() -> int:
    global _LABEL_BUY
    if _LABEL_BUY is None:
        try:
            from trainer import LABEL_BUY
            _LABEL_BUY = int(LABEL_BUY)
        except ImportError:
            _LABEL_BUY = 2  # BUY is always class index 2
    return _LABEL_BUY


# ---------------------------------------------------------------------------
# Block-bootstrap CI (annualized)
# ---------------------------------------------------------------------------

def _block_bootstrap_ci(
    series: np.ndarray,
    block_size: int = 20,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    95% block-bootstrap CI on the annualized mean of a daily excess-return series.
    Preserves serial correlation by resampling contiguous blocks.
    """
    n = len(series)
    if n < 4:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)

    if n < block_size:
        # Fall back to iid bootstrap when series is shorter than one block
        boot = np.array(
            [np.mean(rng.choice(series, size=n, replace=True)) for _ in range(n_resamples)]
        )
    else:
        starts = np.arange(n - block_size + 1)
        n_blocks = math.ceil(n / block_size)
        boot = np.empty(n_resamples)
        for i in range(n_resamples):
            chosen = rng.choice(starts, size=n_blocks, replace=True)
            sample = np.concatenate([series[s: s + block_size] for s in chosen])[:n]
            boot[i] = np.mean(sample)

    boot_annual = boot * 252  # annualize mean daily excess return
    return float(np.percentile(boot_annual, 2.5)), float(np.percentile(boot_annual, 97.5))


# ===========================================================================
# PortfolioBacktest
# ===========================================================================

class PortfolioBacktest:
    """
    NAV-based portfolio backtest for a single loaded model.

    Parameters
    ----------
    model        : loaded UniversalStockModel or UniversalRanker instance
    scoring_mode : 'classifier' (uses P(BUY)) or 'ranker' (uses raw score)
    """

    def __init__(
        self,
        model,
        scoring_mode: str,
        starting_capital: float = 10_000.0,
        top_n: int = 20,
        rebalance_days: int = 20,
        sector_cap: float = 0.30,
        cash_buffer: float = 0.05,
        round_trip_cost: float = 0.012,
        slippage_max: float = 0.0015,
        vol_target: Optional[float] = 0.15,
        seed: int = 42,
    ) -> None:
        if scoring_mode not in ("classifier", "ranker"):
            raise ValueError(f"scoring_mode must be 'classifier' or 'ranker', got {scoring_mode!r}")
        self.model = model
        self.scoring_mode = scoring_mode
        self.starting_capital = starting_capital
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.sector_cap = sector_cap
        self.cash_buffer = cash_buffer
        self.round_trip_cost = round_trip_cost
        self.slippage_max = slippage_max
        self.vol_target = vol_target
        self.seed = seed
        self._annualized_turnover: float = float("nan")

    # -----------------------------------------------------------------------
    # Internal: feature loading
    # -----------------------------------------------------------------------

    def _load_features(self, date: str) -> Optional[pd.DataFrame]:
        """
        Load all feature rows for a single rebalance date, joining stocks for sector.
        Uses SELECT f.* so it picks up all columns added by feature_engine migrations.
        Missing model feature_cols are back-filled with NaN (XGBoost handles NaN).
        """
        try:
            with database.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT f.*, COALESCE(s.sector, 'Unknown') AS sector
                    FROM   features f
                    JOIN   stocks   s ON s.ticker = f.ticker
                    WHERE  f.date = ?
                    """,
                    (date,),
                ).fetchall()
            if not rows:
                return None
            df = pd.DataFrame([dict(r) for r in rows])
            # Guarantee all model feature columns exist
            for col in self.model.feature_cols:
                if col not in df.columns:
                    df[col] = np.nan
            return df
        except Exception as exc:
            log.error("_load_features(%s): %s", date, exc)
            return None

    # -----------------------------------------------------------------------
    # Internal: inverse-volatility weights
    # -----------------------------------------------------------------------

    def _get_vol(
        self,
        tickers: list[str],
        date: str,
        daily_ret_pivot: pd.DataFrame,
    ) -> dict[str, float]:
        """Trailing 20-day return stddev for each ticker, ending at date."""
        try:
            mask = daily_ret_pivot.index <= date
            window = daily_ret_pivot.loc[mask].tail(21)  # 21 rows → 20 returns
            result: dict[str, float] = {}
            for t in tickers:
                if t in window.columns:
                    ret_series = window[t].dropna()
                    result[t] = float(ret_series.std()) if len(ret_series) >= 2 else float("nan")
                else:
                    result[t] = float("nan")
            return result
        except Exception:
            return {t: float("nan") for t in tickers}

    # -----------------------------------------------------------------------
    # Internal: scoring and sector-capped selection
    # -----------------------------------------------------------------------

    def _score_and_select(self, df_t: pd.DataFrame) -> pd.DataFrame:
        """
        Score all tickers in df_t and sort descending by score.
        classifier → score = P(BUY)
        ranker     → score = raw relevance score
        """
        LABEL_BUY = _get_label_buy()
        df_t = df_t.copy()
        if self.scoring_mode == "classifier":
            proba = self.model.predict_proba(df_t)
            df_t["_score"] = proba[:, LABEL_BUY]
        else:  # ranker
            df_t["_score"] = self.model.predict(df_t)
        return df_t.sort_values("_score", ascending=False).reset_index(drop=True)

    def _apply_sector_cap(self, ranked: pd.DataFrame) -> list[str]:
        """
        Greedy sector cap: walk ranked list, add ticker only if sector's
        accumulated provisional weight stays <= sector_cap.
        Provisional weight = 1/top_n (equal-weight approximation).
        """
        sector_weight: dict[str, float] = {}
        selected: list[str] = []
        provisional = 1.0 / self.top_n

        for _, row in ranked.iterrows():
            if len(selected) >= self.top_n:
                break
            ticker = str(row["ticker"])
            sector = str(row.get("sector") or "Unknown")
            current = sector_weight.get(sector, 0.0)
            if current + provisional <= self.sector_cap + 1e-9:
                selected.append(ticker)
                sector_weight[sector] = current + provisional

        return selected

    # -----------------------------------------------------------------------
    # Internal: weight computation
    # -----------------------------------------------------------------------

    def _compute_weights(
        self,
        tickers: list[str],
        date: str,
        daily_ret_pivot: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Compute target portfolio weights summing to (1 - cash_buffer).
        vol_target is None → equal-weight; else → inverse-vol scaling.
        """
        invest = 1.0 - self.cash_buffer
        n = len(tickers)
        if n == 0:
            return {}

        if self.vol_target is None:
            w = invest / n
            return {t: w for t in tickers}

        # Inverse-vol weights
        vols = self._get_vol(tickers, date, daily_ret_pivot)
        inv_vol: dict[str, float] = {}
        for t in tickers:
            v = vols.get(t, float("nan"))
            inv_vol[t] = 1.0 / v if (not math.isnan(v) and v > 1e-8) else float("nan")

        valid_vals = [v for v in inv_vol.values() if not math.isnan(v)]
        if not valid_vals:
            w = invest / n
            return {t: w for t in tickers}

        mean_inv = float(np.mean(valid_vals))
        total = sum(v if not math.isnan(v) else mean_inv for v in inv_vol.values())
        if total <= 0:
            w = invest / n
            return {t: w for t in tickers}

        weights = {}
        for t in tickers:
            v = inv_vol[t] if not math.isnan(inv_vol[t]) else mean_inv
            weights[t] = (v / total) * invest
        return weights

    # -----------------------------------------------------------------------
    # SPY NAV loading
    # -----------------------------------------------------------------------

    def _load_spy_nav(self, start_date: str, end_date: str) -> pd.Series:
        """
        Load SPY NAV normalized to starting_capital.
        Tries prices table first; falls back to yfinance.
        Never silently falls back to a flat 0% benchmark — raises RuntimeError if both fail.
        """
        # -- Try prices table ------------------------------------------------
        with database.connection() as conn:
            rows = conn.execute(
                "SELECT date, close FROM prices "
                "WHERE ticker = 'SPY' AND date BETWEEN ? AND ? ORDER BY date",
                (start_date, end_date),
            ).fetchall()

        if rows:
            log.info("SPY: loaded %d rows from prices table", len(rows))
            spy_df = pd.DataFrame([dict(r) for r in rows])
            spy_close = spy_df.set_index("date")["close"].sort_index()
            spy_close.index = pd.to_datetime(spy_close.index)
            spy_nav = spy_close / spy_close.iloc[0] * self.starting_capital
            spy_nav.index.name = "date"
            return spy_nav

        # -- Fallback: yfinance ----------------------------------------------
        log.info("SPY not found in prices table — fetching from yfinance")
        try:
            import yfinance as yf
            end_dt_plus1 = (
                pd.to_datetime(end_date) + pd.Timedelta(days=1)
            ).date().isoformat()
            spy = yf.Ticker("SPY").history(
                start=start_date,
                end=end_dt_plus1,
                auto_adjust=True,
            )["Close"]
            # Strip timezone
            spy.index = pd.DatetimeIndex(spy.index.strftime("%Y-%m-%d"))
            spy = spy.sort_index()
            spy = spy[(spy.index >= start_date) & (spy.index <= end_date)]
            if spy.empty:
                raise RuntimeError("yfinance returned empty SPY data")
            spy_nav = spy / spy.iloc[0] * self.starting_capital
            spy_nav.index.name = "date"
            log.info("SPY: loaded %d rows from yfinance", len(spy_nav))
            return spy_nav
        except Exception as exc:
            raise RuntimeError(
                f"SPY data unavailable from both prices table and yfinance: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Main simulation
    # -----------------------------------------------------------------------

    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Run the NAV-based portfolio backtest.

        Returns DataFrame indexed by trading date (DatetimeIndex) with columns:
          nav, daily_return, drawdown, n_holdings, turnover, cost_drag
        """
        log.info(
            "PortfolioBacktest.run: %s → %s  mode=%s  top_n=%d  rebalance_days=%d",
            start_date, end_date, self.scoring_mode, self.top_n, self.rebalance_days,
        )

        # ---- Validate window -----------------------------------------------
        if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
            raise RuntimeError(
                f"holdout_start ({start_date}) >= end_date ({end_date}). "
                "No holdout window available."
            )

        # ---- Load all prices into memory -----------------------------------
        # Use a 45-day lookback buffer so vol estimates are stable from day 1
        buffer_start = (
            pd.to_datetime(start_date) - pd.Timedelta(days=45)
        ).date().isoformat()

        log.info("Loading prices from DB (%s → %s)…", buffer_start, end_date)
        with database.connection() as conn:
            price_rows = conn.execute(
                "SELECT ticker, date, close FROM prices "
                "WHERE date BETWEEN ? AND ? ORDER BY ticker, date",
                (buffer_start, end_date),
            ).fetchall()

        if not price_rows:
            raise RuntimeError(
                f"No price data in DB between {buffer_start} and {end_date}"
            )

        prices_df = pd.DataFrame([dict(r) for r in price_rows])
        prices_df["date"] = prices_df["date"].astype(str)
        prices_df = prices_df.sort_values(["ticker", "date"])
        # close-to-close daily return (NaN for each ticker's first row)
        prices_df["daily_return"] = prices_df.groupby("ticker")["close"].pct_change(fill_method=None)

        # Pivot: index=date (str), columns=ticker, values=daily_return
        daily_ret_pivot = prices_df.pivot(
            index="date", columns="ticker", values="daily_return"
        )
        daily_ret_pivot.index.name = "date"
        log.info("Prices loaded: %d tickers across %d dates",
                 daily_ret_pivot.shape[1], daily_ret_pivot.shape[0])

        # ---- Trading-day calendar ------------------------------------------
        with database.connection() as conn:
            cal_rows = conn.execute(
                "SELECT DISTINCT date FROM prices "
                "WHERE date BETWEEN ? AND ? ORDER BY date",
                (start_date, end_date),
            ).fetchall()
        trading_days = [r[0] for r in cal_rows]

        if not trading_days:
            raise RuntimeError(
                f"No trading days found in prices table for {start_date} → {end_date}"
            )

        n_td = len(trading_days)
        if n_td < self.rebalance_days:
            raise RuntimeError(
                f"Holdout window has only {n_td} trading days "
                f"(< rebalance_days={self.rebalance_days}). "
                "Check holdout_start vs end_date and exit non-zero."
            )

        log.info("Trading days: %d  (%s → %s)", n_td, trading_days[0], trading_days[-1])

        # ---- Rebalance dates: every rebalance_days trading days ------------
        rebalance_dates = {trading_days[i] for i in range(0, n_td, self.rebalance_days)}
        log.info("Scheduled rebalance dates: %d", len(rebalance_dates))

        # ---- Simulation state ----------------------------------------------
        nav = self.starting_capital
        weights: dict[str, float] = {}
        n_holdings = 0
        total_turnover = 0.0
        rng = np.random.default_rng(self.seed)
        rows_out: list[dict] = []

        # ---- Main loop -----------------------------------------------------
        for day in trading_days:
            turnover_t = 0.0
            cost_drag_t = 0.0

            if day in rebalance_dates:
                # Load features for this date
                df_t = self._load_features(day)

                if df_t is None or df_t.empty:
                    log.warning(
                        "Rebalance date %s: no features — skipping rebalance, keeping old weights",
                        day,
                    )
                else:
                    # Optional: prediction_guard range validation
                    try:
                        from prediction_guard import validate_feature_ranges  # type: ignore[import]
                        validate_feature_ranges(df_t, self.model.feature_bounds)
                    except (ImportError, AttributeError):
                        log.debug(
                            "prediction_guard.validate_feature_ranges not available; skipping"
                        )

                    # Score, sort, select
                    ranked = self._score_and_select(df_t)
                    selected = self._apply_sector_cap(ranked)

                    if not selected:
                        log.warning("Rebalance date %s: no tickers selected after sector cap", day)
                    else:
                        new_weights = self._compute_weights(selected, day, daily_ret_pivot)

                        # Turnover vs prior weights
                        all_tickers = set(new_weights) | set(weights)
                        turnover_t = 0.5 * sum(
                            abs(new_weights.get(t, 0.0) - weights.get(t, 0.0))
                            for t in all_tickers
                        )
                        total_turnover += turnover_t

                        # Cost drag: round-trip + slippage
                        slippage_t = float(rng.uniform(0.0, self.slippage_max)) * turnover_t
                        cost_drag_t = self.round_trip_cost * turnover_t + slippage_t

                        weights = new_weights
                        n_holdings = len(weights)

            # ---- Daily portfolio return ------------------------------------
            if weights and day in daily_ret_pivot.index:
                day_rets = daily_ret_pivot.loc[day]
                portfolio_ret = 0.0
                for t, w in weights.items():
                    raw = day_rets.get(t, 0.0)
                    portfolio_ret += w * (0.0 if (raw is None or pd.isna(raw)) else float(raw))
                if math.isnan(portfolio_ret):
                    portfolio_ret = 0.0
            else:
                portfolio_ret = 0.0

            # Subtract cost drag on rebalance day
            portfolio_ret -= cost_drag_t

            nav *= 1.0 + portfolio_ret

            rows_out.append({
                "nav":          nav,
                "daily_return": portfolio_ret,
                "drawdown":     0.0,          # filled below
                "n_holdings":   n_holdings,
                "turnover":     turnover_t,
                "cost_drag":    cost_drag_t,
            })

        # ---- Build output DataFrame ----------------------------------------
        nav_df = pd.DataFrame(
            rows_out,
            index=pd.DatetimeIndex(pd.to_datetime(trading_days), name="date"),
        )
        # Drawdown recomputed cleanly
        nav_df["drawdown"] = (nav_df["nav"] / nav_df["nav"].cummax()) - 1.0

        # Store annualized turnover for use in summary_text
        years = n_td / 252.0
        self._annualized_turnover = total_turnover / years if years > 0 else float("nan")

        log.info(
            "Backtest complete: $%.0f → $%.0f  (%+.1f%%)  "
            "| ann_turnover=%.1f× | rebalances_with_trades=%d",
            self.starting_capital,
            nav_df["nav"].iloc[-1],
            (nav_df["nav"].iloc[-1] / self.starting_capital - 1.0) * 100,
            self._annualized_turnover,
            int((nav_df["turnover"] > 0).sum()),
        )
        return nav_df

    # -----------------------------------------------------------------------
    # Metrics (static — uses only the daily NAV series)
    # -----------------------------------------------------------------------

    @staticmethod
    def metrics(nav: pd.Series) -> dict:
        """
        Compute performance metrics from a daily NAV series (DatetimeIndex).

        Sharpe rf=0 (documented: risk-free rate set to 0 for simplicity).

        Returns
        -------
        total_return, cagr, ann_vol, sharpe, sortino, max_dd,
        calmar, skew, kurt, pct_positive_months, longest_dd_days
        """
        nan_dict: dict = {k: float("nan") for k in [
            "total_return", "cagr", "ann_vol", "sharpe", "sortino",
            "max_dd", "calmar", "skew", "kurt",
            "pct_positive_months", "longest_dd_days",
        ]}
        if nav is None or len(nav) < 2:
            return nan_dict

        n = len(nav)
        daily_ret = nav.pct_change().dropna()

        total_return = float(nav.iloc[-1] / nav.iloc[0]) - 1.0
        # cagr: uses n trading days (not calendar days) to avoid calendar bias
        cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (252.0 / n) - 1.0)
        ann_vol = float(daily_ret.std() * math.sqrt(252))
        sharpe = cagr / ann_vol if ann_vol > 1e-10 else float("nan")

        neg = daily_ret[daily_ret < 0]
        down_dev = float(neg.std()) if len(neg) > 1 else 1e-10
        sortino = cagr / (down_dev * math.sqrt(252)) if down_dev > 1e-10 else float("nan")

        cummax = nav.cummax()
        dd_series = (nav / cummax) - 1.0
        max_dd = float(dd_series.min())
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else float("nan")

        skew = float(daily_ret.skew())
        kurt = float(daily_ret.kurt())

        # Monthly return statistics
        try:
            monthly = nav.resample("ME").last()
            monthly_ret = monthly.pct_change().dropna()
            pct_pos = float((monthly_ret > 0).mean()) if len(monthly_ret) > 0 else float("nan")
        except Exception:
            pct_pos = float("nan")

        # Longest drawdown duration (consecutive trading days below prior peak)
        in_dd = dd_series < -1e-6
        longest_dd = 0
        current_streak = 0
        for v in in_dd:
            if v:
                current_streak += 1
                longest_dd = max(longest_dd, current_streak)
            else:
                current_streak = 0

        return {
            "total_return":        total_return,
            "cagr":                cagr,
            "ann_vol":             ann_vol,
            "sharpe":              sharpe,
            "sortino":             sortino,
            "max_dd":              max_dd,
            "calmar":              calmar,
            "skew":                skew,
            "kurt":                kurt,
            "pct_positive_months": pct_pos,
            "longest_dd_days":     longest_dd,
        }

    # -----------------------------------------------------------------------
    # Benchmark-relative metrics (static)
    # -----------------------------------------------------------------------

    @staticmethod
    def vs_benchmark(nav: pd.Series, bench_nav: pd.Series) -> dict:
        """
        Compute benchmark-relative metrics vs a benchmark NAV series.

        Block-bootstrap CI: 1000 iterations, block size 20 trading days.

        Returns
        -------
        alpha_annual, beta, info_ratio, tracking_error,
        pct_months_beat, ci_lower, ci_upper
        """
        nan_dict: dict = {k: float("nan") for k in [
            "alpha_annual", "beta", "info_ratio", "tracking_error",
            "pct_months_beat", "ci_lower", "ci_upper",
        ]}
        if nav is None or bench_nav is None or len(nav) < 4 or len(bench_nav) < 4:
            return nan_dict

        port_ret = nav.pct_change().dropna()
        bench_ret = bench_nav.pct_change().dropna()
        common = port_ret.index.intersection(bench_ret.index)
        if len(common) < 4:
            log.warning("vs_benchmark: only %d common dates — metrics may be unreliable", len(common))
            return nan_dict

        port_ret = port_ret.loc[common]
        bench_ret = bench_ret.loc[common]
        excess = port_ret - bench_ret

        alpha_annual = float(excess.mean()) * 252

        cov_mat = np.cov(port_ret.values, bench_ret.values)
        bench_var = float(np.var(bench_ret.values, ddof=1))
        beta = float(cov_mat[0, 1] / bench_var) if bench_var > 1e-12 else float("nan")

        tracking_error = float(excess.std() * math.sqrt(252))
        info_ratio = (
            alpha_annual / tracking_error if tracking_error > 1e-10 else float("nan")
        )

        # % months where portfolio beat benchmark
        try:
            port_m = nav.resample("ME").last().pct_change().dropna()
            bench_m = bench_nav.resample("ME").last().pct_change().dropna()
            common_m = port_m.index.intersection(bench_m.index)
            pct_months_beat = (
                float((port_m.loc[common_m] > bench_m.loc[common_m]).mean())
                if len(common_m) > 0 else float("nan")
            )
        except Exception:
            pct_months_beat = float("nan")

        # Block-bootstrap 95% CI on annualized alpha
        ci_lower, ci_upper = _block_bootstrap_ci(
            excess.values, block_size=20, n_resamples=1000
        )

        return {
            "alpha_annual":   alpha_annual,
            "beta":           beta,
            "info_ratio":     info_ratio,
            "tracking_error": tracking_error,
            "pct_months_beat": pct_months_beat,
            "ci_lower":       ci_lower,
            "ci_upper":       ci_upper,
        }

    # -----------------------------------------------------------------------
    # Verdict (static helper — takes the merged metrics+bench dict)
    # -----------------------------------------------------------------------

    @staticmethod
    def assign_verdict(m: dict) -> str:
        """
        Assign verdict tier from the combined metrics dict.

        STRONG   alpha_annual > 0 AND ci_lower > 0 AND sharpe > 0.7 AND max_dd > -0.20
        PASS     alpha_annual > 0 AND ci_lower > 0 AND sharpe > 0.4
        MARGINAL alpha_annual > 0 AND ci_lower <= 0
        FAIL     alpha_annual <= 0
        """
        alpha  = m.get("alpha_annual", float("nan"))
        ci_lo  = m.get("ci_lower", float("nan"))
        sharpe = m.get("sharpe", float("nan"))
        max_dd = m.get("max_dd", float("nan"))

        if math.isnan(alpha) or alpha <= 0:
            return "FAIL"
        if math.isnan(ci_lo):
            return "MARGINAL"
        if ci_lo > 0:
            sh_ok = not math.isnan(sharpe)
            dd_ok = not math.isnan(max_dd)
            if sh_ok and sharpe > 0.7 and dd_ok and max_dd > -0.20:
                return "STRONG"
            if sh_ok and sharpe > 0.4:
                return "PASS"
        return "MARGINAL"

    # -----------------------------------------------------------------------
    # Summary text
    # -----------------------------------------------------------------------

    def summary_text(
        self,
        nav: pd.Series,
        spy_nav: pd.Series,
        bench_metrics: dict,
    ) -> str:
        """
        Format a human-readable backtest summary matching the spec format.

        PORTFOLIO BACKTEST   (HOLDOUT — <scoring_mode>)
        ===============================================
        Period     : YYYY-MM-DD to YYYY-MM-DD
        Starting   : $10,000
        Ending     : $X,XXX  (+XX.XX%)
        SPY return : +X.XX%
        Alpha      : +X.XX%   (CI 95%: [-X.X%, +X.X%])
        Sharpe     : X.XX
        Max DD     : -X.XX%
        Turnover   : X.X x annualized
        Verdict    : <TIER> — <reason>
        """
        def _pct(v: float, d: int = 2) -> str:
            return f"{v * 100:+.{d}f}%" if not math.isnan(v) else "N/A"

        def _f2(v: float) -> str:
            return f"{v:.2f}" if not math.isnan(v) else "N/A"

        # Date range
        if len(nav) > 0 and hasattr(nav.index[0], "strftime"):
            start_str = nav.index[0].strftime("%Y-%m-%d")
            end_str   = nav.index[-1].strftime("%Y-%m-%d")
        else:
            start_str = str(nav.index[0]) if len(nav) > 0 else "N/A"
            end_str   = str(nav.index[-1]) if len(nav) > 0 else "N/A"

        # Ending NAV and total return (relative to starting_capital, not nav[0])
        ending_nav = float(nav.iloc[-1]) if len(nav) > 0 else float("nan")
        total_ret  = (ending_nav / self.starting_capital - 1.0) if not math.isnan(ending_nav) else float("nan")

        # SPY total return
        spy_ret = float("nan")
        if spy_nav is not None and len(spy_nav) >= 2:
            spy_ret = float(spy_nav.iloc[-1] / spy_nav.iloc[0]) - 1.0

        alpha  = bench_metrics.get("alpha_annual", float("nan"))
        ci_lo  = bench_metrics.get("ci_lower", float("nan"))
        ci_hi  = bench_metrics.get("ci_upper", float("nan"))
        sharpe = bench_metrics.get("sharpe", float("nan"))
        max_dd = bench_metrics.get("max_dd", float("nan"))
        turnover = getattr(self, "_annualized_turnover", float("nan"))

        # Verdict (use pre-computed key if present, else compute)
        verdict = bench_metrics.get("verdict") or self.assign_verdict(bench_metrics)
        verdict_reason = {
            "STRONG":   "positive alpha, CI > 0, Sharpe > 0.7, drawdown < 20%",
            "PASS":     "positive alpha, CI > 0, Sharpe > 0.4",
            "MARGINAL": "positive alpha but CI includes zero",
            "FAIL":     "no demonstrated alpha over benchmark",
        }.get(verdict, "unknown")

        ci_str = (
            f"CI 95%: [{ci_lo * 100:+.1f}%, {ci_hi * 100:+.1f}%]"
            if not (math.isnan(ci_lo) or math.isnan(ci_hi))
            else "CI 95%: N/A"
        )
        turn_str = (
            f"{turnover:.1f} x annualized"
            if not math.isnan(turnover)
            else "N/A"
        )

        sep = "=" * 47
        lines = [
            f"PORTFOLIO BACKTEST   (HOLDOUT — {self.scoring_mode})",
            sep,
            f"Period     : {start_str} to {end_str}",
            f"Starting   : ${self.starting_capital:,.0f}",
            f"Ending     : ${ending_nav:,.0f}  ({_pct(total_ret)})",
            f"SPY return : {_pct(spy_ret)}",
            f"Alpha      : {_pct(alpha)}   ({ci_str})",
            f"Sharpe     : {_f2(sharpe)}",
            f"Max DD     : {_pct(max_dd)}",
            f"Turnover   : {turn_str}",
            f"Verdict    : {verdict} — {verdict_reason}",
        ]
        return "\n".join(lines)
