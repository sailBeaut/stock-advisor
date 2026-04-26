"""
earnings_collector.py

Fetches historical earnings dates and EPS data from yfinance and computes
three point-in-time features per (ticker, date) row in the features table:

  earnings_surprise_pct  — (eps_actual - eps_estimate) / abs(eps_estimate)
                           Forward-filled from last earnings date.
                           Post-Earnings Announcement Drift (PEAD) is one
                           of the strongest known short-term price signals.

  days_since_earnings    — calendar days since the most recent earnings
                           report on or before the feature date.

  earnings_beat          — 1 if most recent surprise_pct > 0, else 0.
                           Forward-filled from last earnings date.

All features are strictly point-in-time: only earnings ON OR BEFORE each
feature date are used. No negative shifts. No future earnings data.

Entry points
------------
    run_earnings(tickers=None)  — fetch + compute (full pipeline)
    audit_earnings()            — print coverage stats
"""

import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf

import database

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Polite delay between yfinance calls to avoid rate-limiting
_YF_DELAY = 0.5   # seconds

_UPSERT_EARNINGS_SQL = """
INSERT INTO earnings_events (ticker, report_date, eps_actual, eps_estimate, surprise_pct)
VALUES (:ticker, :report_date, :eps_actual, :eps_estimate, :surprise_pct)
ON CONFLICT(ticker, report_date) DO UPDATE SET
    eps_actual   = excluded.eps_actual,
    eps_estimate = excluded.eps_estimate,
    surprise_pct = excluded.surprise_pct
"""


# ---------------------------------------------------------------------------
# Fetch earnings from yfinance
# ---------------------------------------------------------------------------

def _fetch_one(ticker: str) -> list[dict]:
    """
    Fetch earnings dates + EPS for one ticker via yfinance.
    Returns a list of dicts ready for upsert into earnings_events.
    """
    try:
        t = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=20)
    except Exception as exc:
        log.debug("%s: get_earnings_dates failed — %s", ticker, exc)
        return []

    if df is None or df.empty:
        return []

    # Normalise column names (yfinance column names vary slightly)
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "reported" in cl or "actual" in cl:
            col_map[c] = "eps_actual"
        elif "estimate" in cl:
            col_map[c] = "eps_estimate"
        elif "surprise" in cl and "%" in c:
            col_map[c] = "surprise_raw"
    df = df.rename(columns=col_map)

    records = []
    for dt, row in df.iterrows():
        # Date index may be timezone-aware — normalise to YYYY-MM-DD string
        try:
            report_date = pd.Timestamp(dt).normalize().strftime("%Y-%m-%d")
        except Exception:
            continue

        eps_actual   = _to_float(row.get("eps_actual"))
        eps_estimate = _to_float(row.get("eps_estimate"))

        # Compute surprise_pct from actuals if present; fall back to yf column
        surprise_pct = None
        if eps_actual is not None and eps_estimate is not None:
            denom = abs(eps_estimate)
            if denom > 1e-9:
                surprise_pct = round((eps_actual - eps_estimate) / denom, 6)
        elif "surprise_raw" in row.index:
            # yfinance sometimes returns surprise as a percentage already
            raw = _to_float(row.get("surprise_raw"))
            if raw is not None:
                surprise_pct = round(raw / 100.0, 6)

        records.append({
            "ticker":       ticker,
            "report_date":  report_date,
            "eps_actual":   eps_actual,
            "eps_estimate": eps_estimate,
            "surprise_pct": surprise_pct,
        })

    return records


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f   # NaN check
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Fetch all tickers
# ---------------------------------------------------------------------------

def fetch_earnings(tickers: list[str] | None = None) -> int:
    """
    Fetch earnings for all active tickers (or a subset) and store in DB.
    Returns total rows stored.
    """
    database.initialize()

    if tickers is None:
        with database.connection() as conn:
            rows = conn.execute(
                "SELECT ticker FROM stocks WHERE is_active=1"
            ).fetchall()
        tickers = [r[0] for r in rows]

    log.info("Fetching earnings for %d tickers (%.0fs estimated)…",
             len(tickers), len(tickers) * _YF_DELAY)

    total = 0
    errors = 0

    for i, ticker in enumerate(tickers, 1):
        records = _fetch_one(ticker)
        if records:
            with database.connection() as conn:
                conn.executemany(_UPSERT_EARNINGS_SQL, records)
            total += len(records)
        else:
            errors += 1

        if i % 50 == 0 or i == len(tickers):
            log.info("  Progress: %d/%d  (stored: %d, no-data: %d)",
                     i, len(tickers), total, errors)

        time.sleep(_YF_DELAY)

    log.info("Earnings fetch complete — %d rows stored, %d tickers with no data.",
             total, errors)
    return total


# ---------------------------------------------------------------------------
# Compute per-(ticker, date) features
# ---------------------------------------------------------------------------

def compute_earnings_features(tickers: list[str] | None = None) -> int:
    """
    For every (ticker, date) in the features table, compute:
      earnings_surprise_pct, days_since_earnings, earnings_beat

    All values are strictly point-in-time (most recent earnings ON OR BEFORE
    each feature date). Uses binary search for efficiency.

    Writes directly to features table via batch UPDATE.
    Returns number of rows updated.
    """
    log.info("Computing earnings features…")

    # ── Ensure columns exist ─────────────────────────────────────────────
    with database.connection() as conn:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(features)")}
        for col, col_type in [
            ("earnings_surprise_pct", "REAL"),
            ("days_since_earnings",   "INTEGER"),
            ("earnings_beat",         "INTEGER"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE features ADD COLUMN {col} {col_type}")
                log.info("Schema: added features.%s (%s)", col, col_type)

    # ── Load earnings events ──────────────────────────────────────────────
    with database.connection() as conn:
        earn_rows = conn.execute(
            "SELECT ticker, report_date, surprise_pct "
            "FROM earnings_events ORDER BY ticker, report_date"
        ).fetchall()

    if not earn_rows:
        log.warning("earnings_events is empty — run fetch_earnings() first.")
        return 0

    earn_df = pd.DataFrame([dict(r) for r in earn_rows])
    earn_df["report_date"] = pd.to_datetime(earn_df["report_date"])

    # ── Load feature (ticker, date) pairs ────────────────────────────────
    if tickers:
        ph = ",".join("?" * len(tickers))
        with database.connection() as conn:
            feat_rows = conn.execute(
                f"SELECT ticker, date FROM features WHERE ticker IN ({ph})",
                tickers,
            ).fetchall()
    else:
        with database.connection() as conn:
            feat_rows = conn.execute(
                "SELECT ticker, date FROM features"
            ).fetchall()

    if not feat_rows:
        log.warning("features table is empty.")
        return 0

    feat_df = pd.DataFrame([dict(r) for r in feat_rows])
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    log.info("Computing earnings features for %d rows across %d tickers.",
             len(feat_df), feat_df["ticker"].nunique())

    all_updates: list[tuple] = []

    for ticker, grp in feat_df.groupby("ticker"):
        tkr_earn = earn_df[earn_df["ticker"] == ticker].sort_values("report_date")

        report_dates = tkr_earn["report_date"].values.astype("datetime64[D]")
        surprises    = tkr_earn["surprise_pct"].values   # float, may contain NaN

        feature_dates = grp["date"].sort_values().values.astype("datetime64[D]")

        _MAX_DAYS   = 180    # beyond 180 days recency signal is gone; store NULL
        _SURP_CLIP  = 5.0   # clip surprise_pct to ±500% — extreme outliers near-zero estimates

        for d in feature_dates:
            idx = np.searchsorted(report_dates, d, side="right") - 1

            if idx < 0:
                # No prior earnings — store NULL (not 9999) so XGBoost handles
                # as missing rather than an extreme outlier that warps splits
                all_updates.append((None, None, None, ticker, str(d)[:10]))
                continue

            raw_days = int((d - report_dates[idx]) / np.timedelta64(1, "D"))
            days_since = raw_days if raw_days <= _MAX_DAYS else None

            raw_surp = surprises[idx]
            if raw_surp != raw_surp:   # NaN check
                surprise_pct = None
                beat = None
            else:
                # Clip: near-zero EPS estimates cause extreme ratios (e.g. ±100x)
                # that shift tree splits away from the informative ±5% range
                surprise_pct = float(max(-_SURP_CLIP, min(_SURP_CLIP, raw_surp)))
                beat = 1 if surprise_pct > 0 else 0

            all_updates.append((surprise_pct, days_since, beat, ticker, str(d)[:10]))

    # ── Batch UPDATE ──────────────────────────────────────────────────────
    _UPDATE_SQL = """
    UPDATE features
    SET earnings_surprise_pct = ?,
        days_since_earnings   = ?,
        earnings_beat         = ?
    WHERE ticker = ? AND date = ?
    """

    BATCH = 5_000
    updated = 0
    for start in range(0, len(all_updates), BATCH):
        with database.connection() as conn:
            conn.executemany(_UPDATE_SQL, all_updates[start : start + BATCH])
        updated += min(BATCH, len(all_updates) - start)
        if updated % 100_000 == 0 or updated == len(all_updates):
            log.info("  Updated %d / %d rows.", updated, len(all_updates))

    log.info("Earnings features written: %d rows updated.", updated)
    return updated


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def run_earnings(tickers: list[str] | None = None) -> None:
    """Fetch earnings data then compute features. Safe to re-run."""
    log.info("=== Earnings pipeline start ===")
    database.initialize()
    fetch_earnings(tickers=tickers)
    compute_earnings_features(tickers=tickers)
    log.info("=== Earnings pipeline complete ===")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_earnings() -> None:
    """Print earnings coverage and feature stats."""
    with database.connection() as conn:
        n_events = conn.execute(
            "SELECT COUNT(*) FROM earnings_events"
        ).fetchone()[0]
        n_tickers = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM earnings_events"
        ).fetchone()[0]
        date_range = conn.execute(
            "SELECT MIN(report_date), MAX(report_date) FROM earnings_events"
        ).fetchone()
        n_feat = conn.execute(
            "SELECT COUNT(*) FROM features WHERE days_since_earnings IS NOT NULL"
        ).fetchone()[0]
        total_feat = conn.execute(
            "SELECT COUNT(*) FROM features"
        ).fetchone()[0]
        avg_days = conn.execute(
            "SELECT AVG(days_since_earnings) FROM features "
            "WHERE days_since_earnings IS NOT NULL"
        ).fetchone()[0]
        beat_pct = conn.execute(
            "SELECT AVG(earnings_beat) FROM features WHERE earnings_beat IS NOT NULL"
        ).fetchone()[0]

    print("\n=== earnings audit ===")
    print(f"  Earnings events stored : {n_events} across {n_tickers} tickers")
    if date_range and date_range[0]:
        print(f"  Date range             : {date_range[0]}  to  {date_range[1]}")
    print(f"  Feature rows updated   : {n_feat} / {total_feat}")
    if avg_days is not None:
        print(f"  Avg days_since_earnings: {avg_days:.0f} days")
    if beat_pct is not None:
        print(f"  Beat rate              : {beat_pct * 100:.1f}% of rows")
    result = "PASS" if n_events > 0 and n_feat > 0 else "FAIL"
    print(f"  {result}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    run_earnings()
    audit_earnings()
