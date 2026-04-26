"""
macro.py

Fetches market-wide macro features and stores them in the
macro_features table in trading.db.

Sources
-------
FRED API      : Federal Funds Rate (FEDFUNDS), 10Y Treasury (GS10)
Yahoo Finance : VIX (^VIX), S&P 500 (^GSPC)

All series are aligned to the same daily date index covering the
full price history in the DB (2021-01-01 to today). Monthly FRED
series are forward-filled to produce a value for every trading day.

Entry points
------------
    fetch_macro()           — download and store all macro features
    get_macro_dataframe()   — return stored macro data as a DataFrame
    audit_macro()           — print row counts and date range, PASS/FAIL
"""

import logging
import os
import sqlite3

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

import database

load_dotenv()

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
FRED_API_KEY = os.environ["FRED_API_KEY"]
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

_MACRO_COLS = [
    "fed_funds_rate",
    "treasury_10y",
    "yield_curve_spread",
    "vix",
    "vix_sma20",
    "vix_regime",
    "sp500_return_20d",
    "sp500_above_sma50",
    "unemployment_rate",
    "cpi_yoy",
    "spread_10y2y",
]

_NEW_MACRO_COLS = [
    ("unemployment_rate", "REAL"),
    ("cpi_yoy",           "REAL"),
    ("spread_10y2y",      "REAL"),
]

_UPSERT_SQL = """
INSERT OR REPLACE INTO macro_features (
    date, fed_funds_rate, treasury_10y, yield_curve_spread,
    vix, vix_sma20, vix_regime, sp500_return_20d, sp500_above_sma50,
    unemployment_rate, cpi_yoy, spread_10y2y
) VALUES (
    :date, :fed_funds_rate, :treasury_10y, :yield_curve_spread,
    :vix, :vix_sma20, :vix_regime, :sp500_return_20d, :sp500_above_sma50,
    :unemployment_rate, :cpi_yoy, :spread_10y2y
)
"""


def _migrate_macro(conn: sqlite3.Connection) -> None:
    """Add new macro columns to existing macro_features table if absent."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(macro_features)")}
    for col, col_type in _NEW_MACRO_COLS:
        if col not in existing:
            conn.execute(f"ALTER TABLE macro_features ADD COLUMN {col} {col_type}")
            log.info("Schema: added macro_features.%s (%s)", col, col_type)


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a FRED series via the REST API. Returns a named pd.Series."""
    params = {
        "series_id":           series_id,
        "observation_start":   start,
        "observation_end":     end,
        "api_key":             FRED_API_KEY,
        "file_type":           "json",
    }
    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()

    observations = resp.json().get("observations", [])
    values = {}
    for obs in observations:
        if obs["value"] == ".":       # FRED uses "." for missing values
            continue
        try:
            values[pd.to_datetime(obs["date"])] = float(obs["value"])
        except (ValueError, KeyError):
            continue

    s = pd.Series(values, name=series_id)
    log.info("FRED %s: fetched %d non-null observations.", series_id, len(s))
    return s


def _fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance. Returns df with lowercase columns."""
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Yahoo Finance returned no data for {ticker}")

    # Flatten MultiIndex columns (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    log.info("Yahoo %s: fetched %d rows (%s to %s).",
             ticker, len(df), df.index.min().date(), df.index.max().date())
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fetch_macro() -> None:
    """Download all macro series and upsert into macro_features table."""
    database.initialize()

    # ── 0. Schema migration for new columns ────────────────────────────────
    with database.connection() as conn:
        _migrate_macro(conn)

    # ── 1. Determine date range from prices table ──────────────────────────
    with database.connection() as conn:
        row = conn.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()

    if row and row[0] and row[1]:
        start, end = row[0], row[1]
    else:
        import datetime
        start = "2021-01-01"
        end   = datetime.date.today().isoformat()

    log.info("Fetching macro features for %s to %s.", start, end)

    # ── 2. FRED series (monthly → daily via forward-fill) ──────────────────
    fed_funds    = _fetch_fred("FEDFUNDS", start, end)
    treasury_10y = _fetch_fred("GS10",     start, end)
    unemployment = _fetch_fred("UNRATE",   start, end)
    cpi_raw      = _fetch_fred("CPIAUCSL", start, end)
    spread_raw   = _fetch_fred("T10Y2Y",   start, end)   # already daily

    fed_funds_daily    = fed_funds.resample("D").ffill()
    treasury_10y_daily = treasury_10y.resample("D").ffill()
    unemployment_daily = unemployment.resample("D").ffill()
    cpi_yoy_daily      = cpi_raw.pct_change(12).resample("D").ffill()
    spread_10y2y_daily = spread_raw.resample("D").ffill()

    # ── 3. Yahoo Finance series ─────────────────────────────────────────────
    vix_df = _fetch_yahoo("^VIX",  start, end)
    sp_df  = _fetch_yahoo("^GSPC", start, end)

    # ── 4. Align to business-day index ────────────────────────────────────
    bdays = pd.bdate_range(start=start, end=end)

    fed    = fed_funds_daily.reindex(bdays).ffill()
    t10y   = treasury_10y_daily.reindex(bdays).ffill()
    unemp  = unemployment_daily.reindex(bdays).ffill()
    cpi_yoy = cpi_yoy_daily.reindex(bdays).ffill()
    s10y2y = spread_10y2y_daily.reindex(bdays).ffill()
    vix_c  = vix_df["close"].reindex(bdays).ffill()
    sp_c   = sp_df["close"].reindex(bdays).ffill()

    # ── 5. Derived features (all backward-looking — no negative shifts) ────
    yield_spread      = t10y - fed
    vix_sma20         = vix_c.rolling(20).mean()
    vix_regime        = (vix_c > vix_sma20).astype(int)
    sp500_return_20d  = sp_c.pct_change(20, fill_method=None)
    sp500_sma50       = sp_c.rolling(50).mean()
    sp500_above_sma50 = (sp_c > sp500_sma50).astype(int)

    # ── 6. Assemble DataFrame ──────────────────────────────────────────────
    macro = pd.DataFrame({
        "fed_funds_rate":    fed,
        "treasury_10y":      t10y,
        "yield_curve_spread": yield_spread,
        "vix":               vix_c,
        "vix_sma20":         vix_sma20,
        "vix_regime":        vix_regime,
        "sp500_return_20d":  sp500_return_20d,
        "sp500_above_sma50": sp500_above_sma50,
        "unemployment_rate": unemp,
        "cpi_yoy":           cpi_yoy,
        "spread_10y2y":      s10y2y,
    }, index=bdays)

    # ── 7. Upsert into DB ─────────────────────────────────────────────────
    def _nan_to_none(v):
        if v is None:
            return None
        try:
            import math
            return None if math.isnan(float(v)) else float(v)
        except (TypeError, ValueError):
            return None

    def _int_or_none(v):
        f = _nan_to_none(v)
        return None if f is None else int(f)

    records = []
    for dt, row in macro.iterrows():
        records.append({
            "date":               dt.strftime("%Y-%m-%d"),
            "fed_funds_rate":     _nan_to_none(row["fed_funds_rate"]),
            "treasury_10y":       _nan_to_none(row["treasury_10y"]),
            "yield_curve_spread": _nan_to_none(row["yield_curve_spread"]),
            "vix":                _nan_to_none(row["vix"]),
            "vix_sma20":          _nan_to_none(row["vix_sma20"]),
            "vix_regime":         _int_or_none(row["vix_regime"]),
            "sp500_return_20d":   _nan_to_none(row["sp500_return_20d"]),
            "sp500_above_sma50":  _int_or_none(row["sp500_above_sma50"]),
            "unemployment_rate":  _nan_to_none(row["unemployment_rate"]),
            "cpi_yoy":            _nan_to_none(row["cpi_yoy"]),
            "spread_10y2y":       _nan_to_none(row["spread_10y2y"]),
        })

    with database.connection() as conn:
        conn.executemany(_UPSERT_SQL, records)

    log.info("macro_features: upserted %d rows.", len(records))


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def get_macro_dataframe() -> pd.DataFrame:
    """Load macro_features from DB. Returns DataFrame indexed by date."""
    try:
        with database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM macro_features ORDER BY date"
            ).fetchall()
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df.index = pd.to_datetime(df["date"])
    df = df.drop(columns=["date"])
    return df


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_macro() -> None:
    """Print row counts, date range, and non-null counts per column."""
    df = get_macro_dataframe()

    print("\n=== macro_features audit ===")
    if df.empty:
        print("  FAIL — macro_features table is empty.")
        return

    print(f"  Total rows : {len(df)}")
    print(f"  Date range : {df.index.min().date()}  to  {df.index.max().date()}")
    print(f"  Non-null counts per column:")
    for col in _MACRO_COLS:
        if col in df.columns:
            print(f"    {col:<25s}: {df[col].notna().sum()}")

    if len(df) > 1000:
        print("  PASS — row count > 1000.")
    else:
        print(f"  FAIL — only {len(df)} rows (need > 1000).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    fetch_macro()
    audit_macro()
