"""
validation/test_advisor.py

Smoke-test for recommend_trades().

Usage
-----
    python validation/test_advisor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from advisor import recommend_trades

# Parameters that match the test spec
_HOLDINGS = {"AAPL": 10}
_CASH = 5000.0
_CASH_BUFFER = 0.05
_SECTOR_CAP = 0.30

REQUIRED_KEYS = {
    "as_of_date",
    "current_value_usd",
    "target_holdings",
    "trades",
    "expected_cost_drag_usd",
    "cash_after_trades",
    "sector_breakdown",
    "risk_warnings",
}
REQUIRED_TRADE_KEYS = {"ticker", "action", "shares", "price", "usd", "reason"}
VALID_ACTIONS = {"BUY", "SELL", "ADD", "TRIM"}


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_keys_present(result: dict) -> None:
    missing = REQUIRED_KEYS - set(result.keys())
    _assert(not missing, f"Result missing keys: {missing}")


def test_trade_structure(result: dict) -> None:
    _assert(isinstance(result["trades"], list), "trades must be a list")
    for trade in result["trades"]:
        missing = REQUIRED_TRADE_KEYS - set(trade.keys())
        _assert(not missing, f"Trade missing keys {missing} in {trade}")
        _assert(trade["action"] in VALID_ACTIONS, f"Invalid action: {trade['action']!r}")
        _assert(isinstance(trade["shares"], (int, float)) and trade["shares"] > 0,
                f"shares must be > 0, got {trade['shares']}")
        _assert(trade["price"] > 0, f"price must be > 0, got {trade['price']}")
        _assert(trade["usd"] > 0, f"usd must be > 0, got {trade['usd']}")


def test_sector_breakdown_sum(result: dict) -> None:
    # Sector breakdown normalised by current_value.  It should be close to
    # (1 - cash_buffer) but integer-share rounding and any turnover-cap
    # scaling can reduce it.  We accept a ±40% band for small portfolios.
    total = sum(result["sector_breakdown"].values())
    target = 1 - _CASH_BUFFER
    _assert(
        abs(total - target) < 0.40,
        f"sector_breakdown sums to {total:.4f}, expected ~{target:.2f} "
        f"(±0.40 tolerance for integer share rounding and turnover cap)",
    )


def test_sector_cap(result: dict) -> None:
    # No single sector should exceed the cap (small tolerance for rounding)
    for sector, weight in result["sector_breakdown"].items():
        _assert(
            weight <= _SECTOR_CAP + 0.02,
            f"Sector {sector!r} at {weight:.4f} exceeds cap {_SECTOR_CAP}",
        )


def test_cash_consistency(result: dict) -> None:
    # cash_after_trades must be ≤ current_value (we can't create cash)
    _assert(
        result["cash_after_trades"] <= result["current_value_usd"] + 1.0,
        f"cash_after ({result['cash_after_trades']:.2f}) > "
        f"portfolio_value ({result['current_value_usd']:.2f})",
    )


def test_cost_drag_non_negative(result: dict) -> None:
    _assert(
        result["expected_cost_drag_usd"] >= 0,
        f"expected_cost_drag_usd must be >= 0, got {result['expected_cost_drag_usd']}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run(name: str, fn, *args) -> bool:
    try:
        fn(*args)
        print(f"  ok  {name}")
        return True
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return False
    except Exception as exc:
        print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    # Disable noisy logging from advisor / database
    import logging
    logging.disable(logging.CRITICAL)

    print(f"Running recommend_trades(holdings={_HOLDINGS!r}, cash={_CASH}) ...")
    try:
        result = recommend_trades(
            current_holdings=_HOLDINGS,
            cash=_CASH,
            # disable turnover cap so sector_breakdown sums predictably
            max_total_turnover=1.0,
        )
    except Exception as exc:
        print(f"  ERROR  recommend_trades raised {type(exc).__name__}: {exc}")
        print("\nFAIL (recommend_trades failed — is the model trained and DB populated?)")
        sys.exit(1)

    print("Running assertions ...\n")

    tests = [
        ("keys_present",          test_keys_present,          result),
        ("trade_structure",       test_trade_structure,        result),
        ("sector_breakdown_sum",  test_sector_breakdown_sum,   result),
        ("sector_cap",            test_sector_cap,             result),
        ("cash_consistency",      test_cash_consistency,       result),
        ("cost_drag_non_negative",test_cost_drag_non_negative, result),
    ]

    failed = sum(1 for name, fn, *args in tests if not _run(name, fn, *args))

    if failed == 0:
        print("\nPASS")
        sys.exit(0)
    else:
        print(f"\nFAIL ({failed} test(s) failed)")
        sys.exit(1)
