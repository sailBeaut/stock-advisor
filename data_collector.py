"""
data_collector.py

Fetches S&P 500 price and fundamental data from Yahoo Finance and stores
everything in trading.db via database.py.

Two public entry points:
    run_full_download()        – 5 years of history for all tickers
    run_incremental_update()   – last 5 trading days only
"""

import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf

import database
from survivorship import get_historical_tickers

# ---------------------------------------------------------------------------
# Logging
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
WIKI_SP500_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
BATCH_SIZE = 50
BATCH_DELAY = 3          # seconds between batches
FULL_PERIOD = "5y"
INCREMENTAL_DAYS = 5


# ---------------------------------------------------------------------------
# Ticker list
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[dict]:
    """
    Scrape the S&P 500 table from Wikipedia.

    Returns a list of dicts with keys: ticker, name, sector, industry.
    BRK.B / BRK.A dots are converted to dashes to match Yahoo Finance.
    """
    log.info("Fetching S&P 500 ticker list from Wikipedia…")
    tables = pd.read_html(WIKI_SP500_URL, attrs={"id": "constituents"})
    df = tables[0]

    # Normalise column names across Wikipedia's occasional layout changes
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "industry",
    }
    df = df.rename(columns=col_map)[list(col_map.values())]

    # Yahoo Finance uses "-" not "." for class shares
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    records = df.to_dict("records")
    log.info("Retrieved %d tickers.", len(records))
    return records


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def _upsert_stocks(conn, rows: list[dict]) -> None:
    """Insert or update the stocks master table."""
    conn.executemany(
        """
        INSERT INTO stocks (ticker, name, sector, industry, is_active)
        VALUES (:ticker, :name, :sector, :industry, :is_active)
        ON CONFLICT(ticker) DO UPDATE SET
            name      = excluded.name,
            sector    = excluded.sector,
            industry  = excluded.industry,
            is_active = excluded.is_active
        """,
        [{**r, "is_active": r.get("is_active", 1)} for r in rows],
    )


def _upsert_prices(conn, rows: list[dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO prices (ticker, date, open, high, low, close, volume)
        VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
        ON CONFLICT(ticker, date) DO UPDATE SET
            open   = excluded.open,
            high   = excluded.high,
            low    = excluded.low,
            close  = excluded.close,
            volume = excluded.volume
        """,
        rows,
    )


def _upsert_fundamentals(conn, row: dict) -> None:
    """Write fundamental snapshot to fundamental_metadata (not the features table).

    All values come from yf.Ticker.info which returns 2026 live data —
    they are NOT point-in-time.  is_point_in_time=0 flags this explicitly
    so bias guards in trainer.py / feature_engine.py can block accidental use.
    """
    conn.execute(
        """
        INSERT INTO fundamental_metadata
            (ticker, fetch_date, pe_ratio, pb_ratio, debt_equity, roe,
             gross_margin, operating_margin, net_margin,
             market_cap, revenue_growth, is_point_in_time)
        VALUES
            (:ticker, :fetch_date, :pe_ratio, :pb_ratio, :debt_equity, :roe,
             :gross_margin, :operating_margin, :net_margin,
             :market_cap, :revenue_growth, 0)
        ON CONFLICT(ticker) DO UPDATE SET
            fetch_date       = excluded.fetch_date,
            pe_ratio         = excluded.pe_ratio,
            pb_ratio         = excluded.pb_ratio,
            debt_equity      = excluded.debt_equity,
            roe              = excluded.roe,
            gross_margin     = excluded.gross_margin,
            operating_margin = excluded.operating_margin,
            net_margin       = excluded.net_margin,
            market_cap       = excluded.market_cap,
            revenue_growth   = excluded.revenue_growth,
            is_point_in_time = 0
        """,
        row,
    )


def _update_market_cap(conn, ticker: str, market_cap) -> None:
    conn.execute(
        "UPDATE stocks SET market_cap = ? WHERE ticker = ?",
        (market_cap, ticker),
    )


# ---------------------------------------------------------------------------
# Price download (batch)
# ---------------------------------------------------------------------------

def _download_price_batch(
    tickers: list[str],
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for a batch of tickers in one yf.download() call.

    Returns {ticker: DataFrame} with columns open/high/low/close/volume.
    """
    joined = " ".join(tickers)
    try:
        raw = yf.download(
            joined,
            period=period,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as exc:
        log.error("yf.download batch failed: %s", exc)
        return {}

    if raw.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}

    # Single ticker → flat columns; multiple tickers → MultiIndex columns
    if len(tickers) == 1:
        ticker = tickers[0]
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna(how="all")
        if not df.empty:
            results[ticker] = df
    else:
        for ticker in tickers:
            try:
                df = raw[ticker].copy()
            except KeyError:
                continue
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna(how="all")
            if not df.empty:
                results[ticker] = df

    return results


def _df_to_price_rows(ticker: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "ticker": ticker,
                "date": idx.strftime("%Y-%m-%d"),
                "open": _safe_float(row.get("open")),
                "high": _safe_float(row.get("high")),
                "low": _safe_float(row.get("low")),
                "close": _safe_float(row.get("close")),
                "volume": _safe_int(row.get("volume")),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Fundamental data (one ticker at a time – no bulk API available)
# ---------------------------------------------------------------------------

def _fetch_fundamentals(ticker: str) -> dict | None:
    """
    Pull fundamental metrics from yfinance Ticker.info.

    Stores: PE, PB, debt/equity, ROE, gross/operating/net margins.
    Also returns market_cap and revenue_growth for caller use.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        log.warning("fundamentals fetch failed for %s: %s", ticker, exc)
        return None

    today = date.today().isoformat()

    return {
        "ticker":           ticker,
        "fetch_date":       today,
        "pe_ratio":         _safe_float(info.get("trailingPE")),
        "pb_ratio":         _safe_float(info.get("priceToBook")),
        "debt_equity":      _safe_float(info.get("debtToEquity")),
        "roe":              _safe_float(info.get("returnOnEquity")),
        "gross_margin":     _safe_float(info.get("grossMargins")),
        "operating_margin": _safe_float(info.get("operatingMargins")),
        "net_margin":       _safe_float(info.get("profitMargins")),
        "market_cap":       _safe_float(info.get("marketCap")),
        "revenue_growth":   _safe_float(info.get("revenueGrowth")),
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _batches(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _collect(
    tickers_meta: list[dict],
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    fetch_fundamentals: bool = True,
) -> None:
    """
    Download prices (batched) and optionally fundamentals for all tickers,
    then persist everything to the database.
    """
    database.initialize()

    tickers = [row["ticker"] for row in tickers_meta]
    total = len(tickers)

    price_ok: list[str] = []
    price_fail: list[str] = []
    fund_ok: list[str] = []
    fund_fail: list[str] = []

    # -- 1. Upsert stock metadata -----------------------------------------
    with database.connection() as conn:
        _upsert_stocks(conn, tickers_meta)
    log.info("Upserted %d stock records.", total)

    # -- 2. Price download in batches ------------------------------------
    ticker_batches = list(_batches(tickers, BATCH_SIZE))
    log.info(
        "Downloading prices in %d batches of up to %d…",
        len(ticker_batches),
        BATCH_SIZE,
    )

    for batch_num, batch in enumerate(ticker_batches, 1):
        log.info(
            "Price batch %d/%d  (%d tickers)",
            batch_num,
            len(ticker_batches),
            len(batch),
        )

        batch_data = _download_price_batch(
            batch, period=period, start=start, end=end
        )

        with database.connection() as conn:
            for ticker in batch:
                if ticker in batch_data:
                    rows = _df_to_price_rows(ticker, batch_data[ticker])
                    _upsert_prices(conn, rows)
                    price_ok.append(ticker)
                    log.debug(
                        "  %s: %d rows stored.", ticker, len(rows)
                    )
                else:
                    price_fail.append(ticker)
                    log.warning("  %s: no price data returned.", ticker)

        if batch_num < len(ticker_batches):
            log.debug("Sleeping %ds before next batch…", BATCH_DELAY)
            time.sleep(BATCH_DELAY)

    log.info(
        "Prices – success: %d  |  failed: %d",
        len(price_ok),
        len(price_fail),
    )
    if price_fail:
        log.warning("Price failures: %s", ", ".join(price_fail))

    # -- 3. Fundamental data (only on full download or explicitly requested) --
    if not fetch_fundamentals:
        return

    log.info("Fetching fundamentals for %d tickers…", total)

    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            log.info("  Fundamentals progress: %d/%d", i, total)

        fund = _fetch_fundamentals(ticker)
        if fund is None:
            fund_fail.append(ticker)
            continue

        with database.connection() as conn:
            _upsert_fundamentals(conn, fund)
            if fund.get("market_cap") is not None:
                _update_market_cap(conn, ticker, fund["market_cap"])

        fund_ok.append(ticker)

        # Light throttle: 1 request per ~0.3 s avoids hitting rate limits
        time.sleep(0.3)

    log.info(
        "Fundamentals – success: %d  |  failed: %d",
        len(fund_ok),
        len(fund_fail),
    )
    if fund_fail:
        log.warning("Fundamentals failures: %s", ", ".join(fund_fail))


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_full_download() -> None:
    """
    Download 5 years of OHLCV history and current fundamentals for the full
    historical S&P 500 universe (current members + removed since 2021).
    Safe to re-run; existing rows are upserted.
    """
    log.info("=== FULL DOWNLOAD (5 years, survivorship-bias-free) ===")
    tickers_meta = get_historical_tickers()
    _collect(tickers_meta, period=FULL_PERIOD, fetch_fundamentals=True)
    log.info("=== FULL DOWNLOAD COMPLETE ===")


def run_incremental_update() -> None:
    """
    Fetch the last 5 trading days of price data for all active S&P 500
    tickers already in the database, plus refresh fundamentals.
    """
    log.info("=== INCREMENTAL UPDATE (last %d days) ===", INCREMENTAL_DAYS)

    end_date = date.today()
    # Go back extra calendar days to guarantee 5 trading days
    start_date = end_date - timedelta(days=INCREMENTAL_DAYS + 4)

    # Use whatever is already active in the database; fall back to Wikipedia
    with database.connection() as conn:
        rows = conn.execute(
            "SELECT ticker, name, sector, industry FROM stocks WHERE is_active = 1"
        ).fetchall()

    if rows:
        tickers_meta = [dict(r) for r in rows]
        log.info(
            "Updating %d active tickers from database.", len(tickers_meta)
        )
    else:
        log.info("No active tickers in DB; falling back to Wikipedia list.")
        tickers_meta = get_sp500_tickers()

    _collect(
        tickers_meta,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        fetch_fundamentals=True,
    )
    log.info("=== INCREMENTAL UPDATE COMPLETE ===")


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def audit_fundamentals_safety() -> None:
    """
    Verify that fundamental columns are NOT leaking into the features table.

    PASS  — no fundamental column names appear in the features table schema.
    FAIL  — at least one fundamental column was found in features, meaning
            look-ahead bias could reach the model through FEATURE_COLS.
    """
    with database.connection() as conn:
        fund_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(fundamental_metadata)")
        }
        feat_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(features)")
        }

    # Exclude structural columns that are intentionally shared
    _exclude = {"ticker", "fetch_date", "date", "is_point_in_time"}
    fund_cols -= _exclude

    overlap = fund_cols & feat_cols

    print(f"  fundamental_metadata columns checked : {len(fund_cols)}")
    print(f"  features table columns               : {len(feat_cols)}")
    print(f"  Overlapping columns                  : {sorted(overlap) or 'none'}")

    if overlap:
        print()
        print("  FAIL: fundamental columns present in features table.")
        print("  These columns must NOT appear in FEATURE_COLS:")
        for col in sorted(overlap):
            print(f"    - {col}")
    else:
        print()
        print("  PASS: no fundamental columns found in features table.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "incremental":
        run_incremental_update()
    else:
        run_full_download()
