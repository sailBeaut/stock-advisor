"""
validation/test_paper_trading.py

1. Insert a fake recommendation (via signals + prices in DB)
2. Call paper_trading.record_today()
3. Call paper_trading.mark_to_market_today() using existing prices
4. Call paper_trading.forward_performance_report()
5. Assert report has required keys
Prints PASS or FAIL.
"""

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import database
import paper_trading

TEST_TICKER = "PTXXX"
SPY_TICKER  = "SPY"


def _setup(conn, yesterday: str, today: str) -> None:
    """Insert minimal test fixtures."""
    conn.execute(
        "INSERT OR IGNORE INTO stocks (ticker, name, sector, is_active) VALUES (?, 'PaperTest Co', 'Technology', 1)",
        (TEST_TICKER,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO stocks (ticker, name, sector, is_active) VALUES (?, 'S&P 500 ETF', 'ETF', 1)",
        (SPY_TICKER,),
    )
    for date, close in [(yesterday, 100.0), (today, 103.0)]:
        conn.execute(
            "INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (TEST_TICKER, date, close, close, close, close, 1_000_000),
        )
    for date, close in [(yesterday, 450.0), (today, 451.5)]:
        conn.execute(
            "INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (SPY_TICKER, date, close, close, close, close, 50_000_000),
        )
    conn.execute(
        "INSERT OR REPLACE INTO signals (ticker, date, signal, confidence, probabilities) "
        "VALUES (?, ?, 'BUY', 0.85, '{\"SELL\":0.05,\"HOLD\":0.10,\"BUY\":0.85}')",
        (TEST_TICKER, yesterday),
    )


def _teardown(conn, yesterday: str, today: str) -> None:
    conn.execute("DELETE FROM paper_portfolio WHERE as_of_date IN (?, ?)", (yesterday, today))
    conn.execute("DELETE FROM paper_nav WHERE date IN (?, ?)", (yesterday, today))


def main() -> None:
    database.initialize()

    today     = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    with database.connection() as conn:
        _teardown(conn, yesterday, today)
        _setup(conn, yesterday, today)

    # A) Record recommendation from signals
    n = paper_trading.record_today(as_of_date=yesterday)
    assert n >= 0, f"record_today returned {n}, expected >= 0"

    # B) Mark-to-market using prices already in DB
    nav_row = paper_trading.mark_to_market_today(as_of_date=yesterday)
    assert nav_row is not None, "mark_to_market_today returned None"
    assert "nav_usd" in nav_row, f"nav_row missing 'nav_usd': {nav_row}"

    # C) Performance report
    report = paper_trading.forward_performance_report()
    required = {"cumulative_return", "spy_return", "alpha"}
    missing  = required - set(report.keys())
    assert not missing, f"forward_performance_report missing keys: {missing}"

    print("PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"FAIL: {exc}")
        sys.exit(1)
