"""
validation/test_bounds_enforcement.py

Verify that validate_feature_ranges clips out-of-range values and records
violations correctly.

Usage
-----
    python validation/test_bounds_enforcement.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from prediction_guard import validate_feature_ranges


def test_clips_extreme_and_records_violation():
    # Training bounds: rsi_14 saw values in [10, 90]
    # 10% buffer: lower = 10 - 8 = 2.0, upper = 90 + 8 = 98.0
    feature_bounds = {"rsi_14": (10.0, 90.0)}

    df = pd.DataFrame([
        {"ticker": "AAPL", "date": "2026-01-01", "rsi_14": 999.0},  # extreme → clip
        {"ticker": "MSFT", "date": "2026-01-01", "rsi_14":  50.0},  # normal  → unchanged
    ])

    clipped_df, violations = validate_feature_ranges(df, feature_bounds)

    rng            = 90.0 - 10.0       # 80
    expected_upper = 90.0 + 0.10 * rng  # 98.0

    clipped_val = clipped_df.at[0, "rsi_14"]
    assert abs(clipped_val - expected_upper) < 1e-9, (
        f"Expected extreme value clipped to {expected_upper}, got {clipped_val}"
    )

    normal_val = clipped_df.at[1, "rsi_14"]
    assert normal_val == 50.0, (
        f"Expected normal value unchanged at 50.0, got {normal_val}"
    )

    assert len(violations) == 1, (
        f"Expected 1 violation, got {len(violations)}"
    )
    v = violations[0]
    assert v["ticker"]  == "AAPL",         f"Wrong ticker: {v['ticker']}"
    assert v["feature"] == "rsi_14",        f"Wrong feature: {v['feature']}"
    assert v["raw_value"] == 999.0,         f"Wrong raw_value: {v['raw_value']}"
    assert abs(v["clipped_to"] - expected_upper) < 1e-9, (
        f"Wrong clipped_to: {v['clipped_to']}"
    )


def test_clips_below_lower_bound():
    feature_bounds = {"rsi_14": (10.0, 90.0)}
    # lower = 10 - 8 = 2.0
    expected_lower = 10.0 - 0.10 * (90.0 - 10.0)  # 2.0

    df = pd.DataFrame([
        {"ticker": "TSLA", "date": "2026-01-02", "rsi_14": -50.0},
    ])

    clipped_df, violations = validate_feature_ranges(df, feature_bounds)

    assert abs(clipped_df.at[0, "rsi_14"] - expected_lower) < 1e-9, (
        f"Expected {expected_lower}, got {clipped_df.at[0, 'rsi_14']}"
    )
    assert len(violations) == 1
    assert violations[0]["raw_value"] == -50.0
    assert abs(violations[0]["clipped_to"] - expected_lower) < 1e-9


def test_no_violations_when_in_range():
    feature_bounds = {"rsi_14": (10.0, 90.0)}
    df = pd.DataFrame([
        {"ticker": "GOOG", "date": "2026-01-03", "rsi_14": 55.0},
    ])

    clipped_df, violations = validate_feature_ranges(df, feature_bounds)

    assert clipped_df.at[0, "rsi_14"] == 55.0
    assert violations == [], f"Expected no violations, got {violations}"


def test_missing_column_skipped():
    feature_bounds = {"nonexistent_col": (0.0, 1.0)}
    df = pd.DataFrame([{"ticker": "X", "date": "2026-01-01", "rsi_14": 50.0}])

    # Should not raise; just warns and skips the missing column
    clipped_df, violations = validate_feature_ranges(df, feature_bounds)
    assert violations == []


if __name__ == "__main__":
    tests = [
        test_clips_extreme_and_records_violation,
        test_clips_below_lower_bound,
        test_no_violations_when_in_range,
        test_missing_column_skipped,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except AssertionError as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR  {t.__name__}: {exc}")
            failed += 1

    if failed == 0:
        print("\nPASS")
        sys.exit(0)
    else:
        print(f"\nFAIL ({failed} test(s) failed)")
        sys.exit(1)
