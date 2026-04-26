"""
labeler.py

Generates 30-day forward-return labels for every ticker in the database
and writes them to the labels table.

Label rules
-----------
    forward_return > +threshold  →  BUY  (2)
    forward_return < -threshold  →  SELL (0)
    otherwise                    →  HOLD (1)

    Default threshold: ±5%.
    Health Care and Consumer Staples use ±3% because their lower
    volatility means ±5% produces too few SELL/BUY labels.

    forward_return = (close[t+30] - close[t]) / close[t]

IMPORTANT: shift(-30) is used here to compute forward returns and is the
ONLY legitimate use of a negative shift in this project.  All rows where
the future window is incomplete (the last 30 rows per ticker) are dropped
before saving so no label is ever based on non-existent future data.
"""

import logging
import math
import sqlite3

import pandas as pd

import database

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FORWARD_DAYS      = 30
DEFAULT_THRESHOLD = 0.05   # ±5% for most sectors (used only when flag is False)

# When True, dispatch generate_labels to the volatility-normalized labeler
# instead of the fixed-threshold labeler.  Set to False to revert to the
# original behaviour for comparison.
USE_VOL_NORMALIZED_LABELS = True

# Z-score thresholds for vol-normalized labeling
VOL_NORM_BUY_Z  =  0.5
VOL_NORM_SELL_Z = -0.5
VOL_MIN         =  0.001  # drop rows where trailing vol is essentially zero

# Tighter thresholds for low-volatility sectors where ±5% is too rare
# to produce balanced SELL/BUY labels.
SECTOR_THRESHOLDS: dict[str, float] = {
    "Health Care":      0.03,
    "Consumer Staples": 0.03,
    "Utilities":        0.03,
    "Real Estate":      0.03,
}

LABEL_SELL, LABEL_HOLD, LABEL_BUY = 0, 1, 2
LABEL_NAMES = {LABEL_SELL: "SELL", LABEL_HOLD: "HOLD", LABEL_BUY: "BUY"}
IMBALANCE_WARN = 0.15    # warn if any class is below 15 %

# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

def _ensure_labels_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS labels (
            ticker          TEXT    NOT NULL,
            date            TEXT    NOT NULL,
            label           INTEGER NOT NULL,
            forward_return  REAL    NOT NULL,
            forward_zscore  REAL,
            PRIMARY KEY (ticker, date),
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_labels_date
            ON labels (date);

        CREATE INDEX IF NOT EXISTS idx_labels_ticker_date
            ON labels (ticker, date DESC);

        CREATE INDEX IF NOT EXISTS idx_labels_label
            ON labels (label);
    """)


def _migrate_labels_table(conn: sqlite3.Connection) -> None:
    """Add forward_zscore column to the labels table if it does not exist yet."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(labels)")}
    if "forward_zscore" not in existing:
        conn.execute("ALTER TABLE labels ADD COLUMN forward_zscore REAL")
        log.info("Schema migration: added forward_zscore column to labels table.")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _label_ticker(ticker: str, df: pd.DataFrame, sector: str = "") -> pd.DataFrame:
    """
    Compute labels for one ticker using a sector-specific threshold.

    Parameters
    ----------
    ticker : str
    df     : DataFrame with columns [date, close], sorted ascending by date.
    sector : GICS sector name — used to look up the threshold in
             SECTOR_THRESHOLDS (falls back to DEFAULT_THRESHOLD).

    Returns
    -------
    DataFrame with columns [ticker, date, label, forward_return].
    Last FORWARD_DAYS rows are excluded (incomplete future window).
    """
    threshold      = SECTOR_THRESHOLDS.get(sector, DEFAULT_THRESHOLD)
    buy_threshold  =  threshold
    sell_threshold = -threshold

    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)

    # ----------------------------------------------------------------
    # shift(-30) is the ONE sanctioned use of a negative shift.
    # It gives close[t+30] aligned to time t.
    # ----------------------------------------------------------------
    future_close    = close.shift(-FORWARD_DAYS)          # close 30 days ahead
    forward_return  = (future_close - close) / close      # return at time t

    labels = pd.Series(LABEL_HOLD, index=df.index, dtype=int)
    labels[forward_return >  buy_threshold]  = LABEL_BUY
    labels[forward_return <  sell_threshold] = LABEL_SELL

    out = pd.DataFrame({
        "ticker":         ticker,
        "date":           df["date"],
        "label":          labels,
        "forward_return": forward_return,
    })

    # Drop the last FORWARD_DAYS rows — future_close is NaN there.
    # Also drop any mid-series NaN (price gaps, zero prices) so the
    # NOT NULL constraint on forward_return is never violated.
    out = out.iloc[:-FORWARD_DAYS].copy()
    out = out.dropna(subset=["forward_return"])
    return out


def _label_ticker_vol_normalized(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility-normalized labels for one ticker.

    For each row at time t:
      1. Trailing 60-day realized vol:  vol_60d = std(daily_returns[t-60:t])
      2. Expected 30-day vol:           vol_30d = vol_60d * sqrt(30)
      3. Z-score of forward return:     z = forward_return / vol_30d
      4. Label:  z >  VOL_NORM_BUY_Z  → BUY (2)
                 z < VOL_NORM_SELL_Z  → SELL (0)
                 else                 → HOLD (1)

    forward_return stored in the labels table is the RAW return (not the
    z-score) so that backtest.py can use it directly.

    Rows with insufficient vol history (NaN) or essentially-zero vol
    (< VOL_MIN) are dropped and logged.

    Returns
    -------
    DataFrame with columns [ticker, date, label, forward_return, forward_zscore].
    Last FORWARD_DAYS rows are excluded (incomplete future window).
    """
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)

    # Vol computed strictly from past data — no look-ahead
    daily_return     = close.pct_change()
    vol_60d          = daily_return.rolling(window=60, min_periods=60).std()
    expected_30d_vol = vol_60d * math.sqrt(FORWARD_DAYS)

    # ----------------------------------------------------------------
    # shift(-30) is the ONE sanctioned use of a negative shift.
    # It gives close[t+30] aligned to time t.
    # ----------------------------------------------------------------
    future_close   = close.shift(-FORWARD_DAYS)
    forward_return = (future_close - close) / close
    forward_zscore = forward_return / expected_30d_vol

    labels = pd.Series(LABEL_HOLD, index=df.index, dtype=int)
    labels[forward_zscore >  VOL_NORM_BUY_Z]  = LABEL_BUY
    labels[forward_zscore <  VOL_NORM_SELL_Z] = LABEL_SELL

    out = pd.DataFrame({
        "ticker":         ticker,
        "date":           df["date"],
        "label":          labels,
        "forward_return": forward_return,   # raw — backtest.py uses this
        "forward_zscore": forward_zscore,
        "_vol_60d":       vol_60d,
    })

    # Drop the last FORWARD_DAYS rows — future_close is NaN there
    out = out.iloc[:-FORWARD_DAYS].copy()

    # Drop mid-series NaN on forward_return (price gaps / zero prices)
    out = out.dropna(subset=["forward_return"])

    # Drop rows where vol_60d is NaN (fewer than 60 days of history)
    n_before = len(out)
    out = out.dropna(subset=["_vol_60d"])
    n_nan_vol = n_before - len(out)
    if n_nan_vol:
        log.debug("%s: dropped %d rows with NaN vol_60d (insufficient history).",
                  ticker, n_nan_vol)

    # Drop rows where vol_60d is essentially zero (stale / halted stock)
    n_before = len(out)
    out = out[out["_vol_60d"] >= VOL_MIN]
    n_zero_vol = n_before - len(out)
    if n_zero_vol:
        log.info("%s: dropped %d rows with near-zero vol_60d (< %.3f) — stale/halted.",
                 ticker, n_zero_vol, VOL_MIN)

    out = out.drop(columns=["_vol_60d"])
    return out


def _print_distribution(df: pd.DataFrame, scope: str = "overall") -> None:
    """Print class counts and percentages; warn on imbalanced classes."""
    counts = df["label"].value_counts().sort_index()
    total  = len(df)

    log.info("--- Class distribution (%s, n=%d) ---", scope, total)
    for label_id in (LABEL_SELL, LABEL_HOLD, LABEL_BUY):
        count = counts.get(label_id, 0)
        pct   = count / total if total else 0.0
        flag  = "  <<< WARNING: below 15%" if pct < IMBALANCE_WARN else ""
        log.info(
            "  %-4s (label %d): %6d rows  %6.2f%%%s",
            LABEL_NAMES[label_id], label_id, count, pct * 100, flag,
        )

    if "forward_zscore" in df.columns:
        zs = df["forward_zscore"].dropna()
        if len(zs):
            log.info(
                "  forward_zscore : mean=%+.3f  std=%.3f  (n=%d)",
                zs.mean(), zs.std(), len(zs),
            )

    # Collect and emit all warnings in one place
    for label_id in (LABEL_SELL, LABEL_HOLD, LABEL_BUY):
        count = counts.get(label_id, 0)
        pct   = count / total if total else 0.0
        if pct < IMBALANCE_WARN:
            log.warning(
                "Class imbalance: %s is only %.1f%% of %s labels. "
                "Consider resampling or adjusting thresholds.",
                LABEL_NAMES[label_id], pct * 100, scope,
            )


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

_UPSERT_SQL = """
INSERT INTO labels (ticker, date, label, forward_return, forward_zscore)
VALUES (:ticker, :date, :label, :forward_return, :forward_zscore)
ON CONFLICT(ticker, date) DO UPDATE SET
    label          = excluded.label,
    forward_return = excluded.forward_return,
    forward_zscore = excluded.forward_zscore
"""


def _save_labels(rows: list[dict]) -> None:
    for row in rows:
        row.setdefault("forward_zscore", None)
    with database.connection() as conn:
        conn.executemany(_UPSERT_SQL, rows)


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def check_label_staleness() -> bool:
    """
    Compare the most recent price date against the most recent label
    date. Labels are always 30 days behind prices by design, but if
    the gap exceeds 35 days the labels table needs regenerating.

    Returns True if labels are fresh, False if stale.
    """
    with database.connection() as conn:
        max_price = conn.execute(
            "SELECT MAX(date) FROM prices"
        ).fetchone()[0]
        max_label = conn.execute(
            "SELECT MAX(date) FROM labels"
        ).fetchone()[0]

    if not max_price or not max_label:
        log.warning("STALENESS CHECK: prices or labels table is empty.")
        return False

    from datetime import date
    price_dt = date.fromisoformat(max_price)
    label_dt = date.fromisoformat(max_label)
    gap_days = (price_dt - label_dt).days

    log.info(
        "STALENESS CHECK: latest price=%s  latest label=%s  gap=%d days",
        max_price, max_label, gap_days,
    )

    if gap_days > 35:
        log.warning(
            "STALE LABELS: gap is %d days (> 35). "
            "Run generate_labels() to refresh before retraining.",
            gap_days,
        )
        return False

    log.info("Labels are fresh (gap %d days <= 35).", gap_days)
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_labels(tickers: list[str] | None = None) -> None:
    """
    Generate and store BUY/HOLD/SELL labels for all active tickers
    (or the supplied subset).

    Prints per-ticker and overall class distributions and warns if
    any class is below 15 %.
    """
    # Warn if labels are stale — do not block, since this call IS the fix.
    if not check_label_staleness():
        log.warning("generate_labels() proceeding to refresh stale labels.")

    database.initialize()

    with database.connection() as conn:
        _ensure_labels_table(conn)
        _migrate_labels_table(conn)

        if tickers:
            ph   = ",".join("?" * len(tickers))
            rows = conn.execute(
                # Include ALL tickers that have price data, active OR inactive.
                # Inactive = removed from S&P 500 since 2021. Their historical
                # price data is valid training signal and excluding them causes
                # survivorship bias (the model never sees losing stocks).
                f"SELECT DISTINCT s.ticker, s.sector FROM stocks s "
                f"JOIN prices p ON p.ticker = s.ticker "
                f"WHERE s.ticker IN ({ph})",
                tickers,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT s.ticker, s.sector FROM stocks s "
                "JOIN prices p ON p.ticker = s.ticker"
            ).fetchall()

    # Build {ticker: sector} mapping so _label_ticker can use per-sector thresholds
    ticker_sector: dict[str, str] = {
        r["ticker"]: (r["sector"] or "") for r in rows
    }
    active = list(ticker_sector.keys())
    if not active:
        log.warning("No active tickers found.")
        return

    mode = "vol-normalized (z-score)" if USE_VOL_NORMALIZED_LABELS else "fixed-threshold"
    log.info("Generating labels for %d tickers (FORWARD_DAYS=%d, mode=%s)…",
             len(active), FORWARD_DAYS, mode)

    all_frames: list[pd.DataFrame] = []
    ok = fail = 0

    for ticker in active:
        with database.connection() as conn:
            price_rows = conn.execute(
                "SELECT date, close FROM prices "
                "WHERE ticker = ? ORDER BY date",
                (ticker,),
            ).fetchall()

        if not price_rows:
            log.warning("%s: no price data — skipped.", ticker)
            fail += 1
            continue

        df_prices = pd.DataFrame([dict(r) for r in price_rows])

        if len(df_prices) <= FORWARD_DAYS:
            log.warning(
                "%s: only %d rows — need > %d to generate any label.",
                ticker, len(df_prices), FORWARD_DAYS,
            )
            fail += 1
            continue

        sector = ticker_sector.get(ticker, "")
        if USE_VOL_NORMALIZED_LABELS:
            labeled = _label_ticker_vol_normalized(ticker, df_prices)
            log.info("%s: vol-normalized labels (z-thresholds ±%.1f, sector=%s)",
                     ticker, VOL_NORM_BUY_Z, sector or "unknown")
        else:
            labeled = _label_ticker(ticker, df_prices, sector=sector)
            thresh  = SECTOR_THRESHOLDS.get(sector, DEFAULT_THRESHOLD)
            log.info("%s: using threshold=%.0f%% (sector=%s)",
                     ticker, thresh * 100, sector or "unknown")

        if labeled.empty:
            log.warning("%s: no valid rows after labeling — skipped.", ticker)
            fail += 1
            continue

        _print_distribution(labeled, scope=ticker)

        _save_labels(labeled.to_dict("records"))
        all_frames.append(labeled)
        ok += 1

    log.info("Labels saved — tickers OK: %d  |  skipped: %d", ok, fail)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        _print_distribution(combined, scope="ALL TICKERS COMBINED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    tickers_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    generate_labels(tickers=tickers_arg)
