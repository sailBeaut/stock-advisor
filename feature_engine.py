"""
feature_engine.py

Computes technical and derived features from OHLCV price data and writes
results to the features table in trading.db.

Anti-bias guarantees
--------------------
* All indicators from the `ta` library are causal (rolling backward only).
* Derived features use only pct_change(n>0), rolling(), and expanding().
* 52-week high uses expanding().max() — never rolling(252).max() which
  would include future data if applied naively to the full series.
* shift() is never called with a negative argument anywhere in this file.

Entry points
------------
    compute_all(tickers=None, workers=N)        – full history
    compute_incremental(tickers=None, days=10)  – recent rows only
"""

import datetime
import logging
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume

import database
from macro import get_macro_dataframe

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

from encoders import encode_sector, get_sector_encoder

# ---------------------------------------------------------------------------
# Module-level state (initialised in _ensure_schema before workers start)
# ---------------------------------------------------------------------------
_MACRO_DF: pd.DataFrame = pd.DataFrame()
_SECTOR_ENCODER = None          # LabelEncoder; set by _ensure_schema()

# ---------------------------------------------------------------------------
# Schema migration — add columns not present in the original schema
# ---------------------------------------------------------------------------
_EXTRA_COLUMNS = [
    ("volatility_20d",    "REAL"),
    ("high_52w",          "REAL"),
    ("pct_from_52w_high", "REAL"),
    ("dist_from_sma50",   "REAL"),
    ("golden_cross",      "INTEGER"),
    ("sector_encoded",    "INTEGER"),
    ("mcap_tier",         "INTEGER"),   # market-cap quintile 1(small)–5(large)
    # Macro / market-regime features (shared across all tickers on a given date)
    ("fed_funds_rate",     "REAL"),
    ("treasury_10y",       "REAL"),
    ("yield_curve_spread", "REAL"),
    ("vix",                "REAL"),
    ("vix_sma20",          "REAL"),
    ("vix_regime",         "INTEGER"),
    ("sp500_return_20d",   "REAL"),
    ("sp500_above_sma50",  "INTEGER"),
    # EDGAR filing features — set by edgar_collector.py, not by this engine
    ("days_since_8k",  "INTEGER"),
    ("count_8k_90d",   "INTEGER"),
    # Earnings surprise features — set by earnings_collector.py, not by this engine
    ("earnings_surprise_pct", "REAL"),
    ("days_since_earnings",   "INTEGER"),
    ("earnings_beat",         "INTEGER"),
    # Additional FRED macro series
    ("unemployment_rate", "REAL"),
    ("cpi_yoy",           "REAL"),
    ("spread_10y2y",      "REAL"),
    # Sentiment — 7-day rolling average of news_sentiment, set by this engine
    ("sentiment_7d_avg",  "REAL"),
    # Sector-relative features — populated by _compute_sector_relative_features()
    # after all per-ticker workers complete (requires cross-ticker cross-section).
    ("rsi_14_vs_sector",     "REAL"),
    ("return_5d_vs_sector",  "REAL"),
    ("return_20d_vs_sector", "REAL"),
    ("macd_hist_vs_sector",  "REAL"),
    ("vol_20d_vs_sector",    "REAL"),
    ("dist_sma50_vs_sector", "REAL"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add derived-feature columns that were absent from the original DDL."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(features)")}
    for col, col_type in _EXTRA_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} {col_type}")
            log.info("Schema: added features.%s (%s)", col, col_type)


# ---------------------------------------------------------------------------
# Scalar helpers (module-level so workers can pickle them)
# ---------------------------------------------------------------------------

def _f(val) -> float | None:
    """Scalar → float, or None if null/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _i(val) -> int | None:
    """Scalar → int, or None if null/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def _compute_features(
    ticker: str,
    df: pd.DataFrame,
    sector: str = "Unknown",
    mcap_tier: int = 3,
    macro_df: pd.DataFrame = None,
    sentiment_lookup: dict | None = None,
    encoder=None,
) -> list[dict]:
    """
    Compute all features for one ticker from its full OHLCV history.

    Parameters
    ----------
    ticker : str
    df     : DataFrame with columns date, open, high, low, close, volume
             sorted ascending by date, integer RangeIndex.
    encoder : LabelEncoder, optional
        Pre-loaded sector encoder.  When None, encode_sector() loads it
        from disk on each call — pass it explicitly for efficiency.

    Returns
    -------
    List of row dicts ready for upsert into the features table.
    """
    sector_int = encode_sector(sector, encoder=encoder)

    df = df.sort_values("date").reset_index(drop=True)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ------------------------------------------------------------------ SMAs
    sma_5   = ta.trend.SMAIndicator(close, window=5,   fillna=False).sma_indicator()
    sma_10  = ta.trend.SMAIndicator(close, window=10,  fillna=False).sma_indicator()
    sma_20  = ta.trend.SMAIndicator(close, window=20,  fillna=False).sma_indicator()
    sma_50  = ta.trend.SMAIndicator(close, window=50,  fillna=False).sma_indicator()
    sma_200 = ta.trend.SMAIndicator(close, window=200, fillna=False).sma_indicator()

    # ------------------------------------------------------------------ EMAs
    ema_12 = ta.trend.EMAIndicator(close, window=12, fillna=False).ema_indicator()
    ema_26 = ta.trend.EMAIndicator(close, window=26, fillna=False).ema_indicator()

    # ------------------------------------------------------------------ RSI
    rsi_14 = ta.momentum.RSIIndicator(close, window=14, fillna=False).rsi()
    rsi_28 = ta.momentum.RSIIndicator(close, window=28, fillna=False).rsi()

    # ------------------------------------------------------------------ MACD
    _macd      = ta.trend.MACD(close, window_slow=26, window_fast=12,
                                window_sign=9, fillna=False)
    macd       = _macd.macd()
    macd_sig   = _macd.macd_signal()
    macd_hist  = _macd.macd_diff()

    # ----------------------------------------------------------- Stochastic
    _stoch  = ta.momentum.StochasticOscillator(
        high, low, close, window=14, smooth_window=3, fillna=False
    )
    stoch_k = _stoch.stoch()
    stoch_d = _stoch.stoch_signal()

    # ---------------------------------------------------------- Williams %R
    williams_r = ta.momentum.WilliamsRIndicator(
        high, low, close, lbp=14, fillna=False
    ).williams_r()

    # ------------------------------------------------------------------ ROC
    roc_10 = ta.momentum.ROCIndicator(close, window=10, fillna=False).roc()

    # ------------------------------------------------------------------ ATR
    atr_14 = ta.volatility.AverageTrueRange(
        high, low, close, window=14, fillna=False
    ).average_true_range()

    # -------------------------------------------------------- Bollinger Bands
    _bb      = ta.volatility.BollingerBands(close, window=20, window_dev=2,
                                             fillna=False)
    bb_upper  = _bb.bollinger_hband()
    bb_middle = _bb.bollinger_mavg()
    bb_lower  = _bb.bollinger_lband()
    bb_width  = _bb.bollinger_wband()
    bb_pct_b  = _bb.bollinger_pband()

    # ------------------------------------------------------------------ OBV
    obv = ta.volume.OnBalanceVolumeIndicator(
        close, volume, fillna=False
    ).on_balance_volume()

    # ------------------------------------------------ Rolling VWAP (14-day)
    # typical_price * volume, summed over window, divided by sum of volume
    typical_price = (high + low + close) / 3.0
    vwap = (
        (typical_price * volume).rolling(14).sum()
        / volume.rolling(14).sum()
    )

    # ------------------------------------------------------------ Volume
    vol_sma_20   = volume.rolling(20).mean()
    volume_ratio = volume / vol_sma_20

    # ------------------------------------------------------------ Returns
    # pct_change(n) == (close[t] - close[t-n]) / close[t-n]  — past only
    # fill_method=None: don't forward-fill gaps before computing returns.
    ret_1d  = close.pct_change(1,  fill_method=None)
    ret_5d  = close.pct_change(5,  fill_method=None)
    ret_20d = close.pct_change(20, fill_method=None)

    # ---------------------------------------------------- Rolling volatility
    volatility_20d = ret_1d.rolling(20).std()

    # ----------------------------------------- 52-week high (NO look-ahead)
    # rolling(252).max() at time t = max(close[t-252..t]) — past 252 trading
    # days only. Using expanding() was wrong: it gave the all-time high, which
    # compresses the signal for stocks that peaked years earlier.
    high_52w          = close.rolling(252, min_periods=1).max()
    pct_from_52w_high = (close - high_52w) / high_52w   # always <= 0

    # ------------------------------------------------- Distance from SMA50
    dist_from_sma50 = (close - sma_50) / sma_50

    # ---------------------------------------------------- Golden cross (0/1)
    # NaN where either SMA is not yet defined
    _gc_float = np.where(
        sma_50.isna() | sma_200.isna(),
        np.nan,
        (sma_50 > sma_200).astype(float),
    )
    golden_cross = pd.Series(_gc_float, index=df.index)

    # --------------------------------------------------- Macro lookup dict
    # Build date_str -> macro values mapping from the passed DataFrame.
    # Workers receive this as a plain dict so no pandas overhead per row.
    _macro_lookup: dict[str, dict] = {}
    if macro_df is not None and not macro_df.empty:
        _macro_cols = [
            "fed_funds_rate", "treasury_10y", "yield_curve_spread",
            "vix", "vix_sma20", "vix_regime",
            "sp500_return_20d", "sp500_above_sma50",
            "unemployment_rate", "cpi_yoy", "spread_10y2y",
        ]
        for idx_dt, mrow in macro_df.iterrows():
            date_key = idx_dt.strftime("%Y-%m-%d")
            _macro_lookup[date_key] = {
                c: (None if pd.isna(mrow[c]) else mrow[c])
                for c in _macro_cols
                if c in mrow.index
            }

    # --------------------------------------------------- Assemble row dicts
    dates  = df["date"].tolist()
    n      = len(df)

    # Pre-extract numpy arrays for fast .iat-free access
    def _arr(s: pd.Series) -> np.ndarray:
        return s.to_numpy(dtype=float, na_value=np.nan)

    a_sma5    = _arr(sma_5);    a_sma10   = _arr(sma_10)
    a_sma20   = _arr(sma_20);   a_sma50   = _arr(sma_50)
    a_sma200  = _arr(sma_200);  a_ema12   = _arr(ema_12)
    a_ema26   = _arr(ema_26);   a_rsi14   = _arr(rsi_14)
    a_rsi28   = _arr(rsi_28);   a_macd    = _arr(macd)
    a_msig    = _arr(macd_sig); a_mhist   = _arr(macd_hist)
    a_sk      = _arr(stoch_k);  a_sd      = _arr(stoch_d)
    a_wr      = _arr(williams_r); a_roc   = _arr(roc_10)
    a_atr     = _arr(atr_14);   a_bbu    = _arr(bb_upper)
    a_bbm     = _arr(bb_middle); a_bbl   = _arr(bb_lower)
    a_bbw     = _arr(bb_width); a_bbp    = _arr(bb_pct_b)
    a_obv     = _arr(obv);      a_vwap   = _arr(vwap)
    a_vsma    = _arr(vol_sma_20); a_vrat = _arr(volume_ratio)
    a_r1      = _arr(ret_1d);   a_r5     = _arr(ret_5d)
    a_r20     = _arr(ret_20d);  a_vol20  = _arr(volatility_20d)
    a_h52w    = _arr(high_52w); a_p52w   = _arr(pct_from_52w_high)
    a_dsma50  = _arr(dist_from_sma50); a_gc = _arr(golden_cross)

    rows: list[dict] = []
    for i in range(n):
        rows.append({
            "ticker":            ticker,
            "date":              dates[i],
            # SMAs
            "sma_5":             _f(a_sma5[i]),
            "sma_10":            _f(a_sma10[i]),
            "sma_20":            _f(a_sma20[i]),
            "sma_50":            _f(a_sma50[i]),
            "sma_200":           _f(a_sma200[i]),
            # EMAs
            "ema_12":            _f(a_ema12[i]),
            "ema_26":            _f(a_ema26[i]),
            # Momentum
            "rsi_14":            _f(a_rsi14[i]),
            "rsi_28":            _f(a_rsi28[i]),
            "macd":              _f(a_macd[i]),
            "macd_signal":       _f(a_msig[i]),
            "macd_hist":         _f(a_mhist[i]),
            "stoch_k":           _f(a_sk[i]),
            "stoch_d":           _f(a_sd[i]),
            "williams_r":        _f(a_wr[i]),
            "roc_10":            _f(a_roc[i]),
            # Volatility / bands
            "atr_14":            _f(a_atr[i]),
            "bb_upper":          _f(a_bbu[i]),
            "bb_middle":         _f(a_bbm[i]),
            "bb_lower":          _f(a_bbl[i]),
            "bb_width":          _f(a_bbw[i]),
            "bb_pct_b":          _f(a_bbp[i]),
            # Volume
            "obv":               _f(a_obv[i]),
            "vwap":              _f(a_vwap[i]),
            "volume_sma_20":     _f(a_vsma[i]),
            "volume_ratio":      _f(a_vrat[i]),
            # Returns
            "return_1d":         _f(a_r1[i]),
            "return_5d":         _f(a_r5[i]),
            "return_20d":        _f(a_r20[i]),
            # Derived (new columns)
            "volatility_20d":    _f(a_vol20[i]),
            "high_52w":          _f(a_h52w[i]),
            "pct_from_52w_high": _f(a_p52w[i]),
            "dist_from_sma50":   _f(a_dsma50[i]),
            "golden_cross":      _i(a_gc[i]),
            # Sector / size context — constant per ticker
            "sector_encoded":    sector_int,
            "mcap_tier":         mcap_tier,   # quintile 1–5 (3 = median fallback)
            # Macro / market-regime — looked up by date
            **{k: v for k, v in _macro_lookup.get(dates[i], {}).items()},
            # Sentiment — 7-day rolling average, None if no sentiment data
            "sentiment_7d_avg": _f((sentiment_lookup or {}).get(dates[i])),
        })

    return rows


# ---------------------------------------------------------------------------
# Worker — runs inside a subprocess; must be module-level for pickling
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> tuple[str, list[dict] | None, str | None]:
    """
    Load full price history for one ticker, compute all features, return rows.
    Opens its own SQLite connection (safe under WAL mode).
    """
    ticker, db_path, mcap_tier, macro_df, encoder = args
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        price_rows = conn.execute(
            "SELECT date, open, high, low, close, volume "
            "FROM prices WHERE ticker = ? ORDER BY date",
            (ticker,),
        ).fetchall()

        sector_row = conn.execute(
            "SELECT sector FROM stocks WHERE ticker = ?", (ticker,)
        ).fetchone()

        sent_rows = conn.execute(
            "SELECT date, news_sentiment FROM sentiment "
            "WHERE ticker = ? AND news_sentiment IS NOT NULL ORDER BY date",
            (ticker,),
        ).fetchall()
        conn.close()

        if not price_rows:
            return ticker, None, "no price data in DB"

        sector = sector_row["sector"] if sector_row and sector_row["sector"] else "Unknown"

        # Build 7-day rolling average sentiment lookup
        sentiment_lookup: dict[str, float] = {}
        if sent_rows:
            sent_s = pd.Series(
                {r["date"]: float(r["news_sentiment"]) for r in sent_rows}
            ).sort_index()
            sent_7d = sent_s.rolling(7, min_periods=1).mean()
            sentiment_lookup = sent_7d.to_dict()

        df = pd.DataFrame([dict(r) for r in price_rows])
        feature_rows = _compute_features(
            ticker, df, sector=sector, mcap_tier=mcap_tier,
            macro_df=macro_df, sentiment_lookup=sentiment_lookup,
            encoder=encoder,
        )
        return ticker, feature_rows, None

    except Exception as exc:
        return ticker, None, str(exc)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

_UPSERT_SQL = """
INSERT INTO features (
    ticker, date,
    sma_5, sma_10, sma_20, sma_50, sma_200,
    ema_12, ema_26,
    rsi_14, rsi_28,
    macd, macd_signal, macd_hist,
    stoch_k, stoch_d,
    williams_r, roc_10,
    atr_14,
    bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b,
    obv, vwap,
    volume_sma_20, volume_ratio,
    return_1d, return_5d, return_20d,
    volatility_20d, high_52w, pct_from_52w_high,
    dist_from_sma50, golden_cross, sector_encoded, mcap_tier,
    fed_funds_rate, treasury_10y, yield_curve_spread,
    vix, vix_sma20, vix_regime, sp500_return_20d, sp500_above_sma50,
    unemployment_rate, cpi_yoy, spread_10y2y,
    sentiment_7d_avg
)
VALUES (
    :ticker, :date,
    :sma_5, :sma_10, :sma_20, :sma_50, :sma_200,
    :ema_12, :ema_26,
    :rsi_14, :rsi_28,
    :macd, :macd_signal, :macd_hist,
    :stoch_k, :stoch_d,
    :williams_r, :roc_10,
    :atr_14,
    :bb_upper, :bb_middle, :bb_lower, :bb_width, :bb_pct_b,
    :obv, :vwap,
    :volume_sma_20, :volume_ratio,
    :return_1d, :return_5d, :return_20d,
    :volatility_20d, :high_52w, :pct_from_52w_high,
    :dist_from_sma50, :golden_cross, :sector_encoded, :mcap_tier,
    :fed_funds_rate, :treasury_10y, :yield_curve_spread,
    :vix, :vix_sma20, :vix_regime, :sp500_return_20d, :sp500_above_sma50,
    :unemployment_rate, :cpi_yoy, :spread_10y2y,
    :sentiment_7d_avg
)
ON CONFLICT(ticker, date) DO UPDATE SET
    sma_5              = excluded.sma_5,
    sma_10             = excluded.sma_10,
    sma_20             = excluded.sma_20,
    sma_50             = excluded.sma_50,
    sma_200            = excluded.sma_200,
    ema_12             = excluded.ema_12,
    ema_26             = excluded.ema_26,
    rsi_14             = excluded.rsi_14,
    rsi_28             = excluded.rsi_28,
    macd               = excluded.macd,
    macd_signal        = excluded.macd_signal,
    macd_hist          = excluded.macd_hist,
    stoch_k            = excluded.stoch_k,
    stoch_d            = excluded.stoch_d,
    williams_r         = excluded.williams_r,
    roc_10             = excluded.roc_10,
    atr_14             = excluded.atr_14,
    bb_upper           = excluded.bb_upper,
    bb_middle          = excluded.bb_middle,
    bb_lower           = excluded.bb_lower,
    bb_width           = excluded.bb_width,
    bb_pct_b           = excluded.bb_pct_b,
    obv                = excluded.obv,
    vwap               = excluded.vwap,
    volume_sma_20      = excluded.volume_sma_20,
    volume_ratio       = excluded.volume_ratio,
    return_1d          = excluded.return_1d,
    return_5d          = excluded.return_5d,
    return_20d         = excluded.return_20d,
    volatility_20d     = excluded.volatility_20d,
    high_52w           = excluded.high_52w,
    pct_from_52w_high  = excluded.pct_from_52w_high,
    dist_from_sma50    = excluded.dist_from_sma50,
    golden_cross       = excluded.golden_cross,
    sector_encoded     = excluded.sector_encoded,
    mcap_tier          = excluded.mcap_tier,
    fed_funds_rate     = excluded.fed_funds_rate,
    treasury_10y       = excluded.treasury_10y,
    yield_curve_spread = excluded.yield_curve_spread,
    vix                = excluded.vix,
    vix_sma20          = excluded.vix_sma20,
    vix_regime         = excluded.vix_regime,
    sp500_return_20d   = excluded.sp500_return_20d,
    sp500_above_sma50  = excluded.sp500_above_sma50,
    unemployment_rate  = excluded.unemployment_rate,
    cpi_yoy            = excluded.cpi_yoy,
    spread_10y2y       = excluded.spread_10y2y,
    sentiment_7d_avg   = excluded.sentiment_7d_avg
"""


def _with_retry(fn, *args, max_attempts: int = 5, base_delay: float = 0.5, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential back-off on sqlite3.OperationalError
    ('database is locked').  All other exceptions are re-raised immediately.

    Delays: 0.5 s, 1 s, 2 s, 4 s … (doubles each attempt, jitter-free).
    """
    import time

    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == max_attempts:
                raise
            log.warning(
                "_with_retry: DB locked on attempt %d/%d — retrying in %.1f s",
                attempt, max_attempts, delay,
            )
            time.sleep(delay)
            delay *= 2


def _save_features(feature_rows: list[dict]) -> int:
    with database.connection() as conn:
        conn.executemany(_UPSERT_SQL, feature_rows)
    return len(feature_rows)


# ---------------------------------------------------------------------------
# Sector-relative features (cross-ticker, post-parallel step)
# ---------------------------------------------------------------------------

#: Pairs of (base_column_in_features, new_vs_sector_column).
_SECTOR_BASE_VS: list[tuple[str, str]] = [
    ("rsi_14",          "rsi_14_vs_sector"),
    ("return_5d",       "return_5d_vs_sector"),
    ("return_20d",      "return_20d_vs_sector"),
    ("macd_hist",       "macd_hist_vs_sector"),
    ("volatility_20d",  "vol_20d_vs_sector"),
    ("dist_from_sma50", "dist_sma50_vs_sector"),
]

#: Minimum number of non-NaN tickers in a (date, sector) group to emit a mean.
#: Groups below this threshold store NULL — cross-section too thin to be reliable.
_MIN_SECTOR_SIZE = 10


def _compute_sector_relative_features(cutoff_date: str | None = None) -> None:
    """
    Post-process step: compute same-day sector-relative deviations for six features.

    For each (date, sector_encoded) group:
      - Uses only non-NaN values for the mean (.transform("mean") is NaN-aware).
      - Stores NULL when the group has fewer than _MIN_SECTOR_SIZE valid rows.
      - Writes: vs_sector_col = base_col - sector_mean(base_col, date).

    Anti-bias: sector mean for date D is computed exclusively from rows where
    features.date == D — strict same-day cross-section, zero look-ahead.

    Parameters
    ----------
    cutoff_date
        ISO date string (e.g. "2025-10-01").  When given, only rows on or after
        this date are written back.  All rows are still loaded so sector means
        use the full cross-section even for partial incremental updates.
    """
    base_cols = [pair[0] for pair in _SECTOR_BASE_VS]
    vs_cols   = [pair[1] for pair in _SECTOR_BASE_VS]

    log.info(
        "=== Sector-relative features: loading features table%s ===",
        f" (will write rows >= {cutoff_date})" if cutoff_date else "",
    )

    # Load ALL rows — sector means need the full (date, sector) cross-section,
    # even when only a subset of dates will be written back.
    query = (
        "SELECT ticker, date, sector_encoded, "
        + ", ".join(base_cols)
        + " FROM features ORDER BY date"
    )
    with database.connection() as conn:
        db_rows = conn.execute(query).fetchall()

    if not db_rows:
        log.warning("features table is empty — skipping sector-relative computation.")
        return

    df = pd.DataFrame([dict(r) for r in db_rows])
    n_total = len(df)

    # ── Compute sector means and deviations ──────────────────────────────────
    # groupby excludes NaN keys by default (sector_encoded IS NULL rows → NaN result).
    # transform("count") counts non-NaN values per group.
    # transform("mean")  computes mean of non-NaN values per group (skipna=True).
    for base_col, vs_col in _SECTOR_BASE_VS:
        grp       = df.groupby(["date", "sector_encoded"], sort=False)[base_col]
        valid_n   = grp.transform("count")          # non-NaN count per group
        grp_mean  = grp.transform("mean")           # NaN-skipping mean per group
        # Nullify groups below minimum cross-section
        sector_mean = grp_mean.where(valid_n >= _MIN_SECTOR_SIZE, other=np.nan)
        df[vs_col]  = df[base_col] - sector_mean

    # ── Filter rows to write ─────────────────────────────────────────────────
    update_df = df[df["date"] >= cutoff_date] if cutoff_date else df

    # ── Build update records — convert NaN → None for SQLite ─────────────────
    update_cols    = ["ticker", "date"] + vs_cols
    update_records: list[dict] = []
    for tup in update_df[update_cols].itertuples(index=False):
        rec = {"ticker": tup.ticker, "date": tup.date}
        for vs_col in vs_cols:
            v = getattr(tup, vs_col)
            # NaN != NaN is True — reliable cross-platform NaN test
            rec[vs_col] = None if (v != v) else float(v)
        update_records.append(rec)

    # ── Write back ────────────────────────────────────────────────────────────
    set_clause = ",\n            ".join(f"{c} = :{c}" for c in vs_cols)
    update_sql = f"""
        UPDATE features
        SET
            {set_clause}
        WHERE ticker = :ticker AND date = :date
    """
    with database.connection() as conn:
        conn.executemany(update_sql, update_records)

    log.info(
        "=== Sector-relative features written: %d rows updated"
        " (%d total in table), %d columns ===",
        len(update_records), n_total, len(vs_cols),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_tickers(tickers: list[str] | None) -> list[str]:
    """
    Return tickers that have price data in the DB.

    Includes BOTH active (is_active=1) AND inactive/removed (is_active=0)
    tickers so that the model trains on companies that were eventually
    removed from the index — closing the survivorship-bias loop.
    Only tickers that actually have price rows are returned; removed tickers
    with no price data are silently skipped.
    """
    with database.connection() as conn:
        if tickers:
            ph = ",".join("?" * len(tickers))
            rows = conn.execute(
                f"SELECT DISTINCT s.ticker FROM stocks s "
                f"JOIN prices p ON p.ticker = s.ticker "
                f"WHERE s.ticker IN ({ph})",
                tickers,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT s.ticker FROM stocks s "
                "JOIN prices p ON p.ticker = s.ticker"
            ).fetchall()
    return [r[0] for r in rows]


def _ensure_schema() -> None:
    global _MACRO_DF, _SECTOR_ENCODER
    database.initialize()
    with database.connection() as conn:
        _migrate(conn)

        # ---- load macro data once before workers start ----------------------
    _MACRO_DF = get_macro_dataframe()
    if _MACRO_DF.empty:
        log.warning(
            "macro_features table is empty. "
            "Run python macro.py before feature_engine.py "
            "to include macro context in features."
        )
    else:
        log.info("Macro features loaded: %d rows (%s to %s).",
                 len(_MACRO_DF), _MACRO_DF.index.min().date(),
                 _MACRO_DF.index.max().date())

    _SECTOR_ENCODER = get_sector_encoder()
    log.info(
        "Sector encoder loaded: %d classes (%s … %s).",
        len(_SECTOR_ENCODER.classes_),
        _SECTOR_ENCODER.classes_[0],
        _SECTOR_ENCODER.classes_[-1],
    )

    # bias guard for fundamental cols lives in trainer.load_data()


def _run_parallel(
    tickers: list[str],
    workers: int,
    cutoff_date: str | None,
    encoder,
) -> tuple[int, int]:
    """Dispatch workers, collect results, save to DB. Returns (ok, fail)."""
    db_path = str(database.DB_PATH)
    ok = fail = 0
    total = len(tickers)

    # mcap_tier removed Apr 2026 due to look-ahead bias.
    # Re-add only with point-in-time market cap from
    # price * shares-outstanding-at-date. Until then, every row gets
    # the neutral median tier so XGBoost ignores it (and trainer/
    # ranker no longer load this column anyway).
    mcap_tier_map: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker, (t, db_path, mcap_tier_map.get(t, 3), _MACRO_DF, encoder)): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, rows, err = future.result()
            except Exception as exc:
                log.error("%s: worker crash: %s", ticker, exc)
                fail += 1
                continue

            if err:
                log.warning("%s: skipped — %s", ticker, err)
                fail += 1
                continue

            if cutoff_date:
                rows = [r for r in rows if r["date"] >= cutoff_date]

            _with_retry(_save_features, rows)
            ok += 1

            done = ok + fail
            if done % 50 == 0 or done == total:
                log.info("Progress: %d/%d  (failed: %d)", done, total, fail)

    return ok, fail


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def compute_all(
    tickers: list[str] | None = None,
    workers: int | None = None,
) -> None:
    """
    Compute and store features across the full price history for every
    active ticker (or the supplied subset).  Safe to re-run (upsert).
    """
    _ensure_schema()
    all_tickers = _resolve_tickers(tickers)
    if not all_tickers:
        log.warning("No active tickers found in the database.")
        return

    n_workers = workers or min(4, os.cpu_count() or 1)
    log.info(
        "=== compute_all: %d tickers, %d workers ===",
        len(all_tickers), n_workers,
    )
    ok, fail = _run_parallel(all_tickers, n_workers, cutoff_date=None, encoder=_SECTOR_ENCODER)
    log.info(
        "=== compute_all complete — saved: %d  |  failed: %d ===", ok, fail
    )
    _compute_sector_relative_features(cutoff_date=None)


def compute_incremental(
    tickers: list[str] | None = None,
    days: int = 10,
    workers: int | None = None,
) -> None:
    """
    Recompute features using full price history (required for correct
    expanding/long-window indicators) but persist only rows from the last
    `days` trading days to minimise DB write load.
    """
    _ensure_schema()
    all_tickers = _resolve_tickers(tickers)
    if not all_tickers:
        log.warning("No active tickers found in the database.")
        return

    with database.connection() as conn:
        row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
    latest_date = row[0] if (row and row[0]) else None

    if latest_date:
        cutoff = (
            datetime.date.fromisoformat(latest_date)
            - datetime.timedelta(days=days + 10)   # buffer for weekends/holidays
        ).isoformat()
    else:
        cutoff = None

    n_workers = workers or min(4, os.cpu_count() or 1)
    log.info(
        "=== compute_incremental: %d tickers, saving rows >= %s ===",
        len(all_tickers), cutoff,
    )
    ok, fail = _run_parallel(all_tickers, n_workers, cutoff_date=cutoff, encoder=_SECTOR_ENCODER)
    log.info(
        "=== compute_incremental complete — saved: %d  |  failed: %d ===",
        ok, fail,
    )
    _compute_sector_relative_features(cutoff_date=cutoff)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    _tickers = sys.argv[2:] if len(sys.argv) > 2 else None

    if _mode == "incremental":
        compute_incremental(tickers=_tickers)
    else:
        compute_all(tickers=_tickers)
