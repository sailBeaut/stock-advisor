"""
Validation test for volatility-normalized labeling.

Requires trading.db to exist and contain price data for at least some of the
TICKERS list.  Run `python labeler.py` first if the database is empty.
"""
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database
import labeler

TICKERS = ["AAPL", "MSFT", "JNJ", "TSLA", "KO"]

# ---------------------------------------------------------------------------
# 1. Regenerate labels for the subset using the current labeler
# ---------------------------------------------------------------------------
labeler.generate_labels(tickers=TICKERS)

# ---------------------------------------------------------------------------
# 2. Query results
# ---------------------------------------------------------------------------
with database.connection() as conn:
    # -- schema check --------------------------------------------------------
    schema_cols = {row[1] for row in conn.execute("PRAGMA table_info(labels)")}
    if "forward_zscore" not in schema_cols:
        print(f"FAIL: forward_zscore column missing from labels table. Columns: {schema_cols}")
        sys.exit(1)

    rows = conn.execute(
        f"SELECT label, forward_return, forward_zscore "
        f"FROM labels "
        f"WHERE ticker IN ({','.join('?' * len(TICKERS))}) "
        f"  AND forward_zscore IS NOT NULL",
        TICKERS,
    ).fetchall()

if not rows:
    print("FAIL: no rows with forward_zscore found in labels table for test tickers.")
    sys.exit(1)

n = len(rows)

# ---------------------------------------------------------------------------
# 3. forward_zscore column has values (already filtered IS NOT NULL above)
# ---------------------------------------------------------------------------
zscores = [r["forward_zscore"] for r in rows]
assert all(z is not None for z in zscores), "forward_zscore contains unexpected NULLs"

# ---------------------------------------------------------------------------
# 4. Label distribution — each class must be >= 15 % of rows
# ---------------------------------------------------------------------------
label_counts = {0: 0, 1: 0, 2: 0}
for r in rows:
    label_counts[r["label"]] += 1

names = {0: "SELL", 1: "HOLD", 2: "BUY"}
for label_id, name in names.items():
    pct = label_counts[label_id] / n
    if pct < 0.15:
        print(
            f"FAIL: {name} label is only {pct:.1%} of rows (threshold >= 15%). "
            f"Counts: {label_counts}  total: {n}"
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
# 5. forward_return must be raw (not z-scored)
#    Raw 30-day returns for large-cap stocks have stddev ~5–15 %, well below
#    the stddev ~1.0 of a z-score distribution.
# ---------------------------------------------------------------------------
fwd_returns = [r["forward_return"] for r in rows]
fwd_stdev   = statistics.stdev(fwd_returns)
if fwd_stdev >= 0.5:
    print(
        f"FAIL: forward_return std={fwd_stdev:.3f} >= 0.5, "
        "which suggests z-scores were stored instead of raw returns."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
print("PASS")
