"""
edgar_collector.py

Fetches SEC EDGAR 8-K filing metadata for S&P 500 tickers and computes
two time-series features per (ticker, date) row in the features table:

  days_since_8k  — calendar days since the most recent 8-K filing on or
                   before each feature date (material events: earnings,
                   CEO changes, M&A, restatements, dividend cuts, etc.)

  count_8k_90d   — count of 8-K filings in the 90 calendar days before
                   each feature date (proxy for corporate activity level)

No API key required. SEC EDGAR is fully public and free.
Rate limit: max 10 requests/second. This script uses a 0.15s sleep between
requests, staying comfortably below that limit.

Required User-Agent per SEC policy:
    "AppName/Version contact@email.com"

Data flows
----------
1. fetch_ticker_cik_map()     — SEC company_tickers.json → ticker_cik table
2. fetch_filings(tickers)     — SEC submissions API      → edgar_filings table
3. compute_edgar_features()   — joins filings + features → updates features table

Entry points
------------
    run_edgar(tickers=None)   — runs all three steps
    audit_edgar()             — print coverage and stats
"""

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

import database

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC configuration
# ---------------------------------------------------------------------------
EDGAR_USER_AGENT   = "StockMarketML/1.0 stockbstockk@gmail.com"
SEC_TICKERS_URL    = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS    = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSIONS_EX = "https://data.sec.gov/submissions/{filename}"

_HEADERS = {"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# 8-K is the primary target; include amendments too
_TARGET_FORMS = {"8-K", "8-K/A"}

# How long to wait between SEC API requests (be a good citizen)
_REQUEST_DELAY = 0.15   # seconds — stays well under 10 req/s limit

# Minimum number of 8-K records to consider EDGAR data useful
_MIN_FILINGS_FOR_COVERAGE = 100


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> dict | None:
    """GET with retry and polite delay. Returns parsed JSON or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            time.sleep(_REQUEST_DELAY)
            return resp.json()
        except requests.HTTPError as exc:
            if resp.status_code == 429:
                wait = 60
                log.warning("SEC rate-limited. Waiting %ds before retry %d.", wait, attempt)
                time.sleep(wait)
            elif resp.status_code in (403, 404):
                log.warning("SEC returned %d for %s — skipping.", resp.status_code, url)
                return None
            else:
                log.warning("HTTP %d on attempt %d: %s", resp.status_code, attempt, url)
                time.sleep(2.0 * attempt)
        except Exception as exc:
            log.warning("Request failed attempt %d: %s — %s", attempt, url, exc)
            time.sleep(2.0 * attempt)
    log.error("All %d attempts failed for %s", retries, url)
    return None


# ---------------------------------------------------------------------------
# Step 1: Ticker → CIK mapping
# ---------------------------------------------------------------------------

def fetch_ticker_cik_map() -> dict[str, str]:
    """
    Download the SEC company tickers JSON and store in the ticker_cik table.

    Returns a dict mapping ticker -> zero-padded 10-digit CIK string.
    CIKs from SEC are integers; we zero-pad to 10 digits for URL construction.
    """
    log.info("Fetching SEC company tickers map from %s", SEC_TICKERS_URL)
    data = _get(SEC_TICKERS_URL)
    if not data:
        raise RuntimeError("Failed to download SEC company tickers map.")

    # Only store CIKs for tickers that exist in our stocks table
    with database.connection() as conn:
        tracked = {r[0] for r in conn.execute("SELECT ticker FROM stocks")}

    # Response format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
    records = []
    mapping: dict[str, str] = {}
    for entry in data.values():
        ticker = entry.get("ticker", "").upper().strip()
        cik    = str(entry.get("cik_str", "")).strip()
        name   = entry.get("title", "")
        if ticker and cik and ticker in tracked:
            cik_padded = cik.zfill(10)
            mapping[ticker] = cik_padded
            records.append({"ticker": ticker, "cik": cik_padded, "name": name})

    # Upsert into ticker_cik table
    with database.connection() as conn:
        conn.executemany(
            """
            INSERT INTO ticker_cik (ticker, cik, name)
            VALUES (:ticker, :cik, :name)
            ON CONFLICT(ticker) DO UPDATE SET
                cik  = excluded.cik,
                name = excluded.name
            """,
            records,
        )

    log.info("Stored CIK mappings for %d tickers.", len(records))
    return mapping


def _load_ticker_cik_map() -> dict[str, str]:
    """Load the ticker → CIK map from DB. Fetch from SEC if empty."""
    with database.connection() as conn:
        rows = conn.execute("SELECT ticker, cik FROM ticker_cik").fetchall()
    if not rows:
        return fetch_ticker_cik_map()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Step 2: Fetch 8-K filing dates from EDGAR submissions API
# ---------------------------------------------------------------------------

def _extract_8k_rows(ticker: str, filings_block: dict) -> list[dict]:
    """
    Extract 8-K filing rows from a EDGAR submissions filings block.
    The block has parallel arrays: form[], filingDate[], accessionNumber[], ...
    """
    forms       = filings_block.get("form",            [])
    dates       = filings_block.get("filingDate",       [])
    accessions  = filings_block.get("accessionNumber",  [])

    rows = []
    for form, filed_date, acc in zip(forms, dates, accessions):
        if form in _TARGET_FORMS and filed_date and acc:
            rows.append({
                "ticker":           ticker,
                "filed_date":       filed_date,
                "form_type":        form,
                "accession_number": acc,
            })
    return rows


def _fetch_one_ticker(ticker: str, cik: str, cutoff_date: str) -> list[dict]:
    """
    Fetch all 8-K filings for a single ticker from SEC EDGAR.
    Handles both the 'recent' block and any historical overflow files.

    Parameters
    ----------
    ticker      : stock ticker (for storage)
    cik         : zero-padded 10-digit CIK string
    cutoff_date : only return filings on or after this date (YYYY-MM-DD)
    """
    url  = SEC_SUBMISSIONS.format(cik=cik)
    data = _get(url)
    if not data:
        return []

    all_rows: list[dict] = []

    # ── Recent filings (always present) ────────────────────────────────────
    recent = data.get("filings", {}).get("recent", {})
    all_rows.extend(_extract_8k_rows(ticker, recent))

    # ── Historical overflow files (older companies have many filings) ───────
    extra_files = data.get("filings", {}).get("files", [])
    for finfo in extra_files:
        filename = finfo.get("name", "")
        if not filename:
            continue
        # Stop fetching history if the most recent date in this file predates cutoff
        # The "date" field gives the latest filing date in this chunk
        chunk_date = finfo.get("date", "")
        if chunk_date and chunk_date < cutoff_date:
            break   # files are listed newest-first; safe to stop
        ext_data = _get(SEC_SUBMISSIONS_EX.format(filename=filename))
        if ext_data:
            all_rows.extend(_extract_8k_rows(ticker, ext_data))

    # Filter to requested date range
    all_rows = [r for r in all_rows if r["filed_date"] >= cutoff_date]
    return all_rows


def fetch_filings(
    tickers: list[str] | None = None,
    years_back: int = 6,
) -> int:
    """
    Fetch 8-K filing metadata from SEC EDGAR for all tracked tickers.

    Parameters
    ----------
    tickers    : subset of tickers; defaults to all active tickers in DB
    years_back : how many years of history to fetch (default 6 to cover
                 the full 5-year price history with margin)

    Returns
    -------
    Total number of filing rows stored.
    """
    database.initialize()
    cik_map = _load_ticker_cik_map()

    if tickers is None:
        with database.connection() as conn:
            rows = conn.execute(
                "SELECT ticker FROM stocks WHERE is_active=1"
            ).fetchall()
        tickers = [r[0] for r in rows]

    cutoff = (date.today() - timedelta(days=years_back * 365)).isoformat()
    log.info(
        "Fetching EDGAR 8-K filings for %d tickers (since %s).",
        len(tickers), cutoff,
    )

    total_stored = 0
    missing_cik  = 0

    for i, ticker in enumerate(tickers, 1):
        cik = cik_map.get(ticker)
        if not cik:
            log.debug("%s: no CIK found — skipping.", ticker)
            missing_cik += 1
            continue

        rows = _fetch_one_ticker(ticker, cik, cutoff)
        if rows:
            with database.connection() as conn:
                conn.executemany(
                    """
                    INSERT INTO edgar_filings
                        (ticker, filed_date, form_type, accession_number)
                    VALUES
                        (:ticker, :filed_date, :form_type, :accession_number)
                    ON CONFLICT(ticker, accession_number) DO NOTHING
                    """,
                    rows,
                )
            total_stored += len(rows)

        if i % 50 == 0 or i == len(tickers):
            log.info("  Progress: %d/%d tickers processed.", i, len(tickers))

    log.info(
        "EDGAR fetch complete — %d filing rows stored, %d tickers had no CIK.",
        total_stored, missing_cik,
    )
    return total_stored


# ---------------------------------------------------------------------------
# Step 3: Compute per-(ticker, date) EDGAR features → update features table
# ---------------------------------------------------------------------------

def compute_edgar_features(tickers: list[str] | None = None) -> int:
    """
    For every (ticker, date) row in the features table, compute:

      days_since_8k  — calendar days since the most recent 8-K on or before date
                       NULL if no prior 8-K exists for that ticker
      count_8k_90d   — count of 8-K filings in the 90 days on or before date

    Writes directly to the features table via batch UPDATE.
    Safe to re-run: overwrites previously computed values.

    Returns
    -------
    Number of feature rows updated.
    """
    log.info("Computing EDGAR features for features table…")

    # ── Ensure columns exist in features table ────────────────────────────
    with database.connection() as conn:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(features)")}
        for col, col_type in [("days_since_8k", "INTEGER"), ("count_8k_90d", "INTEGER")]:
            if col not in existing:
                conn.execute(f'ALTER TABLE features ADD COLUMN {col} {col_type}')
                log.info("Schema: added features.%s (%s)", col, col_type)

    # ── Load raw filing dates from edgar_filings ───────────────────────────
    with database.connection() as conn:
        filing_rows = conn.execute(
            "SELECT ticker, filed_date FROM edgar_filings WHERE form_type IN ('8-K','8-K/A')"
        ).fetchall()

    if not filing_rows:
        log.warning(
            "edgar_filings is empty — run fetch_filings() first. "
            "Skipping compute_edgar_features."
        )
        return 0

    filings_df = pd.DataFrame([dict(r) for r in filing_rows])
    filings_df["filed_date"] = pd.to_datetime(filings_df["filed_date"])

    # ── Load feature (ticker, date) pairs ─────────────────────────────────
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
        log.warning("features table is empty — run feature_engine first.")
        return 0

    feat_df = pd.DataFrame([dict(r) for r in feat_rows])
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    log.info(
        "Computing EDGAR features for %d (ticker, date) rows across %d tickers.",
        len(feat_df), feat_df["ticker"].nunique(),
    )

    all_updates: list[dict] = []

    # ── Process ticker by ticker ───────────────────────────────────────────
    ticker_groups = feat_df.groupby("ticker")
    for ticker, grp in ticker_groups:
        # Get sorted 8-K dates for this ticker
        tkr_filings = filings_df[filings_df["ticker"] == ticker]["filed_date"].sort_values()
        filing_arr  = tkr_filings.values.astype("datetime64[D]")   # numpy datetime64

        feature_dates = grp["date"].sort_values().values.astype("datetime64[D]")

        if len(filing_arr) == 0:
            # No filings for this ticker — store NULL so XGBoost treats it as
            # missing data rather than an extreme outlier (9999 distorts splits).
            for d in feature_dates:
                all_updates.append({
                    "ticker":        ticker,
                    "date":          str(d)[:10],
                    "days_since_8k": None,
                    "count_8k_90d":  0,
                })
            continue

        w90 = np.timedelta64(90, "D")
        _MAX_DAYS = 180   # beyond 180 days the recency signal is gone; cap to NULL

        for d in feature_dates:
            # days_since_8k: most recent filing on or before d
            idx = np.searchsorted(filing_arr, d, side="right") - 1
            if idx < 0:
                days_since = None          # no prior filing — let XGBoost handle as NaN
            else:
                raw = int((d - filing_arr[idx]) / np.timedelta64(1, "D"))
                days_since = raw if raw <= _MAX_DAYS else None

            # count_8k_90d: count filings in [d - 90, d]
            lo  = np.searchsorted(filing_arr, d - w90, side="left")
            hi  = np.searchsorted(filing_arr, d,        side="right")
            cnt = int(hi - lo)

            all_updates.append({
                "ticker":        ticker,
                "date":          str(d)[:10],
                "days_since_8k": days_since,
                "count_8k_90d":  cnt,
            })

    # ── Batch UPDATE features table ────────────────────────────────────────
    # Positional params (?) — named params fail when column name has digits
    _UPDATE_SQL = """
    UPDATE features
    SET days_since_8k = ?,
        count_8k_90d  = ?
    WHERE ticker = ? AND date = ?
    """

    BATCH = 5_000
    updated = 0
    update_tuples = [
        (r["days_since_8k"], r["count_8k_90d"], r["ticker"], r["date"])
        for r in all_updates
    ]
    for start in range(0, len(update_tuples), BATCH):
        batch = update_tuples[start : start + BATCH]
        with database.connection() as conn:
            conn.executemany(_UPDATE_SQL, batch)
        updated += len(batch)
        if updated % 50_000 == 0 or updated == len(update_tuples):
            log.info("  Updated %d / %d rows.", updated, len(update_tuples))

    log.info("EDGAR features written: %d rows updated.", updated)
    return updated


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def run_edgar(tickers: list[str] | None = None) -> None:
    """
    Run the full EDGAR pipeline:
      1. Fetch/refresh ticker → CIK map
      2. Fetch 8-K filing metadata from SEC
      3. Compute days_since_8k and count_8k_90d in the features table
    """
    log.info("=== EDGAR pipeline start ===")
    database.initialize()

    log.info("Step 1: ticker → CIK map")
    fetch_ticker_cik_map()

    log.info("Step 2: fetch 8-K filings from SEC")
    fetch_filings(tickers=tickers)

    log.info("Step 3: compute EDGAR features")
    compute_edgar_features(tickers=tickers)

    log.info("=== EDGAR pipeline complete ===")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_edgar() -> None:
    """Print filing counts, date ranges, and feature coverage stats."""
    with database.connection() as conn:
        n_filings = conn.execute(
            "SELECT COUNT(*) FROM edgar_filings WHERE form_type IN ('8-K','8-K/A')"
        ).fetchone()[0]

        n_tickers = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM edgar_filings"
        ).fetchone()[0]

        date_range = conn.execute(
            "SELECT MIN(filed_date), MAX(filed_date) FROM edgar_filings"
        ).fetchone()

        n_feat_rows = conn.execute(
            "SELECT COUNT(*) FROM features WHERE days_since_8k IS NOT NULL"
        ).fetchone()[0]

        total_feat_rows = conn.execute(
            "SELECT COUNT(*) FROM features"
        ).fetchone()[0]

        avg_days = conn.execute(
            "SELECT AVG(days_since_8k) FROM features WHERE days_since_8k IS NOT NULL"
        ).fetchone()[0]

        cik_count = conn.execute(
            "SELECT COUNT(*) FROM ticker_cik"
        ).fetchone()[0]

    print("\n=== EDGAR audit ===")
    print(f"  ticker_cik mappings  : {cik_count}")
    print(f"  8-K filings stored   : {n_filings} across {n_tickers} tickers")
    if date_range and date_range[0]:
        print(f"  Filing date range    : {date_range[0]}  to  {date_range[1]}")
    print(f"  Feature rows updated : {n_feat_rows} / {total_feat_rows}")
    if avg_days is not None:
        print(f"  Avg days_since_8k    : {avg_days:.0f} days")

    if n_filings >= _MIN_FILINGS_FOR_COVERAGE and n_feat_rows > 0:
        print("  PASS")
    elif n_filings == 0:
        print("  FAIL — no filings stored. Run run_edgar() first.")
    else:
        print(f"  PARTIAL — {n_filings} filings stored but features not yet computed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    run_edgar()
    audit_edgar()
