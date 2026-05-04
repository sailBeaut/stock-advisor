"""
survivorship.py

Provides get_historical_tickers() which returns both current S&P 500
members (is_active=1) and tickers removed from the index since 2021
(is_active=0), closing the survivorship bias loop.

Without removed tickers the model only ever trains on companies that
survived — inflating BUY signal and hiding how bad SELL situations look.

Entry points
------------
    get_historical_tickers() -> list[dict]
    audit_survivorship()                    — prints PASS/FAIL to stdout
"""

import logging
import re
import datetime

import io

import requests
import pandas as pd

import database

# Shared requests session with a browser-like User-Agent so Wikipedia
# doesn't reject the request with 403 Forbidden.
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; stock-market-research-bot/1.0; "
        "+https://github.com/user/stock-market-project)"
    )
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies-financials/master/data/constituents-financials.csv"
)
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Only look back to the start of our price data window
HISTORY_SINCE_YEAR = 2021

# Date formats tried when parsing the changes table
_DATE_FMTS = (
    "%B %d, %Y",   # January 3, 2022
    "%b %d, %Y",   # Jan 3, 2022
    "%Y-%m-%d",    # 2022-01-03
    "%d %B %Y",    # 3 January 2022
    "%B %Y",       # January 2022  (day-less)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_ticker(t: str) -> str:
    """Strip whitespace, upper-case, convert dots → dashes (BRK.B → BRK-B)."""
    return str(t).strip().upper().replace(".", "-")


def _parse_year(date_str: str) -> int | None:
    """Return the year embedded in a date string, or None if unparseable."""
    s = str(date_str).strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.datetime.strptime(s, fmt).year
        except ValueError:
            pass
    # Last resort: grab any 4-digit year
    m = re.search(r"\b(20\d{2})\b", s)
    return int(m.group(1)) if m else None


def _flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex column headers into 'Level0|Level1' strings."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    new_cols = []
    for parts in df.columns:
        parts_clean = [
            str(p) for p in parts
            if p and not str(p).startswith("Unnamed")
        ]
        new_cols.append("|".join(parts_clean) if parts_clean else "")
    df.columns = new_cols
    return df


def _find_col(columns: list[str], *keywords: str) -> str | None:
    """Return the first column name that contains ALL of the given keywords
    (case-insensitive), or None."""
    kw_lower = [k.lower() for k in keywords]
    for col in columns:
        col_l = col.lower()
        if all(k in col_l for k in kw_lower):
            return col
    return None


# ---------------------------------------------------------------------------
# Current S&P 500 members
# ---------------------------------------------------------------------------

def _get_current_sp500() -> list[dict]:
    """
    Fetch the current S&P 500 constituent list.
    Primary:  GitHub constituents-financials CSV.
    Fallback: Wikipedia #constituents table.
    """
    # ── Primary: GitHub CSV ──────────────────────────────────────────────
    try:
        df = pd.read_csv(GITHUB_CSV_URL)
        df.columns = [c.strip() for c in df.columns]

        sym_col  = _find_col(df.columns, "symbol") or _find_col(df.columns, "ticker")
        name_col = _find_col(df.columns, "name") or _find_col(df.columns, "security")
        sec_col  = _find_col(df.columns, "sector")

        if sym_col is None:
            raise ValueError(f"No symbol/ticker column. Got: {list(df.columns)}")

        records = []
        for _, row in df.iterrows():
            t = _norm_ticker(str(row[sym_col]))
            if not t or t == "NAN":
                continue
            records.append({
                "ticker":    t,
                "name":      str(row[name_col]).strip() if name_col else "",
                "sector":    str(row[sec_col]).strip()  if sec_col  else "",
                "industry":  "",
                "is_active": 1,
            })

        log.info("Current S&P 500: %d tickers sourced from GitHub CSV.", len(records))
        return records

    except Exception as exc:
        log.warning("GitHub CSV failed (%s) — falling back to Wikipedia.", exc)

    # ── Fallback: Wikipedia #constituents table ───────────────────────────
    html = _SESSION.get(WIKI_SP500_URL, timeout=30).text
    tables = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})
    df = tables[0]
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "Symbol":            "ticker",
        "Security":          "name",
        "GICS Sector":       "sector",
        "GICS Sub-Industry": "industry",
    }
    df = df.rename(columns=rename)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    records = [
        {
            "ticker":    str(row.get("ticker", "")),
            "name":      str(row.get("name",   "")),
            "sector":    str(row.get("sector", "")),
            "industry":  str(row.get("industry", "")),
            "is_active": 1,
        }
        for _, row in df.iterrows()
    ]
    log.info("Current S&P 500: %d tickers sourced from Wikipedia fallback.", len(records))
    return records


# ---------------------------------------------------------------------------
# Historically removed tickers
# ---------------------------------------------------------------------------

def _get_removed_since(since_year: int = HISTORY_SINCE_YEAR) -> list[dict]:
    """
    Scrape the S&P 500 composition-changes table from Wikipedia and return
    tickers that were REMOVED on or after since_year, tagged is_active=0.

    The Wikipedia page carries the #changes table alongside #constituents.
    Column naming is brittle — the code searches by keyword rather than
    relying on exact names.
    """
    try:
        html = _SESSION.get(WIKI_SP500_URL, timeout=30).text
        tables = pd.read_html(io.StringIO(html), attrs={"id": "changes"})
        df = _flatten_multiindex(tables[0])
    except Exception as exc:
        log.warning("S&P 500 changes table unavailable: %s", exc)
        return []

    cols = list(df.columns)
    log.debug("Changes table columns after flatten: %s", cols)

    date_col           = _find_col(cols, "date")
    removed_ticker_col = (
        _find_col(cols, "removed", "ticker")
        or _find_col(cols, "removed", "symbol")
    )
    removed_name_col   = (
        _find_col(cols, "removed", "security")
        or _find_col(cols, "removed", "name")
    )

    if removed_ticker_col is None:
        log.warning(
            "Could not locate 'Removed Ticker' column in changes table. "
            "Columns seen: %s", cols
        )
        return []

    removed: list[dict] = []
    seen: set[str] = set()

    for _, row in df.iterrows():
        # ── Filter by year ────────────────────────────────────────────────
        year = _parse_year(str(row.get(date_col, "")) if date_col else "")
        if year is None or year < since_year:
            continue

        t = _norm_ticker(str(row.get(removed_ticker_col, "")))
        if not t or t in ("", "NAN", "-") or t in seen:
            continue

        name = ""
        if removed_name_col:
            raw_name = str(row.get(removed_name_col, ""))
            name = "" if raw_name in ("nan", "NaN", "-") else raw_name.strip()

        seen.add(t)
        removed.append({
            "ticker":    t,
            "name":      name,
            "sector":    "",   # not available in the changes table
            "industry":  "",
            "is_active": 0,
        })

    log.info(
        "S&P 500 removals since %d: %d unique tickers.", since_year, len(removed)
    )
    return removed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fetch_universe(conn) -> list[dict]:
    """
    Active tickers that should be fetched daily.

    Reads is_active = 1 from the stocks table. Confirmed delistings (which
    have been marked is_active = 0) are excluded from the daily fetch but
    their historical rows stay in the database for survivorship-bias-correct
    backtesting.
    """
    rows = conn.execute(
        "SELECT ticker, name, sector, industry FROM stocks WHERE is_active = 1"
    ).fetchall()
    return [dict(r) for r in rows]


def get_historical_tickers() -> list[dict]:
    """
    Return the full historical S&P 500 universe:
      - Current members  →  is_active = 1
      - Removed since 2021 (not re-added) →  is_active = 0

    This prevents survivorship bias: the model trains on companies that
    were eventually removed (acquisitions, bankruptcies, poor performance)
    as well as the ones that stayed.
    """
    current  = _get_current_sp500()
    removed  = _get_removed_since(since_year=HISTORY_SINCE_YEAR)

    current_set = {r["ticker"] for r in current}

    combined: list[dict] = list(current)
    seen_removed: set[str] = set()

    for r in removed:
        t = r["ticker"]
        # Skip if the ticker was later re-added to the index
        if t not in current_set and t not in seen_removed:
            combined.append(r)
            seen_removed.add(t)

    n_active   = sum(1 for r in combined if r["is_active"] == 1)
    n_inactive = sum(1 for r in combined if r["is_active"] == 0)
    log.info(
        "Historical universe: %d total  (%d active, %d removed/inactive).",
        len(combined), n_active, n_inactive,
    )
    return combined


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_survivorship() -> None:
    """
    Query the stocks table and report active vs inactive counts.

    PASS  if inactive > 0  — removed tickers are present; survivorship bias
                             is being addressed.
    FAIL  if inactive == 0 — only current survivors exist; the model will
                             never learn what a failing company looks like.
    """
    with database.connection() as conn:
        active   = conn.execute(
            "SELECT COUNT(*) FROM stocks WHERE is_active = 1"
        ).fetchone()[0]
        inactive = conn.execute(
            "SELECT COUNT(*) FROM stocks WHERE is_active = 0"
        ).fetchone()[0]

    total = active + inactive
    print(f"  Total tickers in database  : {total}")
    print(f"  Active   (is_active = 1)   : {active}")
    print(f"  Inactive (is_active = 0)   : {inactive}")

    if inactive == 0:
        print()
        print("  WARN: No inactive/removed tickers found.")
        print("  Only current S&P 500 survivors are in the database.")
        print("  Run run_full_download() to include removed constituents.")
        print()
        print("  FAIL")
    else:
        pct = inactive / total * 100 if total else 0
        print(f"  Removed-ticker share       : {pct:.1f}%")
        print()
        print(f"  PASS: {inactive} delisted/removed tickers included.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    log.info("Fetching historical S&P 500 ticker universe…")
    tickers = get_historical_tickers()

    n_active   = sum(1 for t in tickers if t["is_active"] == 1)
    n_inactive = sum(1 for t in tickers if t["is_active"] == 0)
    print(f"\nHistorical universe: {len(tickers)} tickers  "
          f"(active: {n_active}, removed: {n_inactive})\n")

    print("=== SURVIVORSHIP AUDIT (current DB state) ===")
    audit_survivorship()
