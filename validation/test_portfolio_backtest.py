"""
validation/test_portfolio_backtest.py

Smoke-test for portfolio_backtest.PortfolioBacktest.

Runs a 60-trading-day backtest with the saved ranker, asserts structural
correctness of the output, and verifies that SPY NAV is real (not zero-filled).
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database
from ranker_trainer import UniversalRanker
from portfolio_backtest import PortfolioBacktest


def main() -> None:
    # ---- Load ranker -------------------------------------------------------
    rkr = UniversalRanker.load()

    # ---- Determine a 60-trading-day window ending at latest prices date ----
    with database.connection() as conn:
        end_date = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        cal_rows = conn.execute(
            "SELECT DISTINCT date FROM prices ORDER BY date"
        ).fetchall()

    trading_days = [r[0] for r in cal_rows]

    if len(trading_days) < 60:
        raise AssertionError(
            f"Not enough trading days in prices table: {len(trading_days)} < 60"
        )

    start_date = trading_days[-60]
    print(f"Test window: {start_date} to {end_date}  ({len(trading_days[-60:])} trading days)")

    # ---- Run backtest ------------------------------------------------------
    pb = PortfolioBacktest(
        model=rkr,
        scoring_mode="ranker",
        starting_capital=1_000.0,
        top_n=5,
        rebalance_days=20,
    )
    nav_df = pb.run(start_date, end_date)

    # ---- Structural assertions on nav_df -----------------------------------
    assert len(nav_df) > 30, f"nav_df has only {len(nav_df)} rows (expected > 30)"

    final_nav = nav_df["nav"].iloc[-1]
    assert not math.isnan(final_nav), "final nav is NaN"
    assert final_nav > 0, f"final nav is non-positive: {final_nav}"

    required_cols = {"nav", "daily_return", "drawdown", "n_holdings", "turnover", "cost_drag"}
    missing = required_cols - set(nav_df.columns)
    assert not missing, f"nav_df missing columns: {missing}"

    # ---- metrics() assertions ----------------------------------------------
    m = PortfolioBacktest.metrics(nav_df["nav"])
    for key in ["cagr", "sharpe", "max_dd", "sortino", "calmar"]:
        assert key in m, f"metrics() missing key: {key!r}"

    # ---- SPY NAV assertions ------------------------------------------------
    spy_nav = pb._load_spy_nav(start_date, end_date)
    assert len(spy_nav) > 30, f"spy_nav has only {len(spy_nav)} rows (expected > 30)"
    assert not (spy_nav == 0).all(), "spy_nav is all zeros — SPY data not loaded"
    assert not spy_nav.isna().all(), "spy_nav is all NaN"

    # ---- vs_benchmark() assertions -----------------------------------------
    bench = PortfolioBacktest.vs_benchmark(nav_df["nav"], spy_nav)
    for key in ["alpha_annual", "ci_lower", "ci_upper"]:
        assert key in bench, f"vs_benchmark() missing key: {key!r}"

    # ---- Holdings assertions -----------------------------------------------
    holds = nav_df["n_holdings"]
    assert holds.max() > 0, "n_holdings is 0 throughout — no positions ever taken"
    assert holds[holds > 0].max() <= 5, (
        f"n_holdings exceeds top_n=5: max={holds.max()}"
    )

    # ---- Realistic metric range check -------------------------------------
    cagr   = m.get("cagr", float("nan"))
    sharpe = m.get("sharpe", float("nan"))
    max_dd = m.get("max_dd", float("nan"))

    if not math.isnan(cagr):
        assert abs(cagr) < 20.0, (
            f"CAGR={cagr:.1%} is outside realistic range — likely a compounding bug"
        )
    if not math.isnan(sharpe):
        assert abs(sharpe) < 10.0, (
            f"Sharpe={sharpe:.2f} is outside realistic range — check for look-ahead bias"
        )
    if not math.isnan(max_dd):
        assert max_dd <= 0.0, f"max_dd={max_dd:.2%} should be <= 0"

    print("test_portfolio_backtest: PASS")


if __name__ == "__main__":
    main()
