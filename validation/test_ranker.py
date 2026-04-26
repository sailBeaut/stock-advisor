"""
Smoke test for UniversalRanker.

Fits on a 5-ticker subset so the test runs in minutes rather than hours.
Does NOT save the model — this is purely a correctness/integration check.

Requires trading.db with features + labels for at least some of the TICKERS.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranker_trainer import UniversalRanker

TICKERS = ["AAPL", "MSFT", "JNJ", "TSLA", "KO"]

# ---------------------------------------------------------------------------
# 1. Load data for the subset and verify it's non-empty
# ---------------------------------------------------------------------------
ranker = UniversalRanker()
df = ranker.load_data(tickers=TICKERS)

if df.empty:
    print("FAIL: load_data() returned an empty DataFrame for test tickers.")
    sys.exit(1)

if "rank_label" not in df.columns:
    print("FAIL: rank_label column missing from load_data() output.")
    sys.exit(1)

# Verify group-alignment invariant on the full dataset before splitting
df_sorted = df.sort_values(["date", "ticker"]).reset_index(drop=True)
groups_check = df_sorted.groupby("date", sort=False).size().tolist()
assert sum(groups_check) == len(df_sorted), (
    f"Pre-fit group size mismatch: {sum(groups_check)} != {len(df_sorted)}"
)

# ---------------------------------------------------------------------------
# 2. Fit on the subset
# ---------------------------------------------------------------------------
ranker.fit(df=df)

# ---------------------------------------------------------------------------
# 3. Model must be set after fit
# ---------------------------------------------------------------------------
if ranker.model is None:
    print("FAIL: ranker.model is None after fit().")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. predict() output shape must match input row count
# ---------------------------------------------------------------------------
scores = ranker.predict(df)
if scores.shape[0] != len(df):
    print(
        f"FAIL: predict() returned {scores.shape[0]} scores "
        f"for {len(df)} input rows."
    )
    sys.exit(1)

# Scores must be finite real numbers (no NaN / Inf)
import numpy as np
if not np.all(np.isfinite(scores)):
    print("FAIL: predict() returned non-finite scores (NaN or Inf).")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. evaluate_topk on the test split must return a non-empty DataFrame
# ---------------------------------------------------------------------------
_, _, test_df, _ = ranker.chronological_split(df)

if test_df.empty:
    print("FAIL: test split is empty — not enough dates for a 70/15/10/5 split.")
    sys.exit(1)

topk_df = ranker.evaluate_topk(test_df, k=3)   # k=3 is safe for 5-ticker universe

if topk_df.empty:
    print("FAIL: evaluate_topk() returned an empty DataFrame.")
    sys.exit(1)

required_cols = {"topk_return", "bah_return", "edge", "n_stocks"}
missing = required_cols - set(topk_df.columns)
if missing:
    print(f"FAIL: evaluate_topk() DataFrame missing columns: {missing}")
    sys.exit(1)

# ---------------------------------------------------------------------------
print("PASS")
