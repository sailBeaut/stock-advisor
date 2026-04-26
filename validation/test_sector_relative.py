"""
Validation test for sector-relative features in the features table.

Assumes feature_engine.py has already been run so the features table is
populated.  Does NOT re-run the engine — it inspects existing data only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database

VS_COLS = [
    "rsi_14_vs_sector",
    "return_5d_vs_sector",
    "return_20d_vs_sector",
    "macd_hist_vs_sector",
    "vol_20d_vs_sector",
    "dist_sma50_vs_sector",
]

# ---------------------------------------------------------------------------
# 1. Schema check — all six columns must exist
# ---------------------------------------------------------------------------
with database.connection() as conn:
    schema_cols = {row[1] for row in conn.execute("PRAGMA table_info(features)")}

missing = set(VS_COLS) - schema_cols
if missing:
    print(f"FAIL: missing columns in features table: {missing}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Coverage — at least 80 % of rows must have non-NULL values in every col
# ---------------------------------------------------------------------------
with database.connection() as conn:
    total = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]

if total == 0:
    print("FAIL: features table is empty — run feature_engine.py first.")
    sys.exit(1)

with database.connection() as conn:
    for col in VS_COLS:
        non_null = conn.execute(
            f"SELECT COUNT(*) FROM features WHERE {col} IS NOT NULL"
        ).fetchone()[0]
        pct = non_null / total
        if pct < 0.80:
            print(
                f"FAIL: {col} has only {pct:.1%} non-NULL rows "
                f"({non_null}/{total}); threshold is 80 %."
            )
            sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Arithmetic cross-check for one (date, sector_encoded) pair
#    Find the most recent date+sector that has >= 10 tickers with valid rsi_14.
#    Manually recompute the sector mean and verify stored rsi_14_vs_sector.
# ---------------------------------------------------------------------------
with database.connection() as conn:
    pair = conn.execute("""
        SELECT date, sector_encoded, COUNT(*) AS n
        FROM features
        WHERE rsi_14 IS NOT NULL
          AND sector_encoded IS NOT NULL
        GROUP BY date, sector_encoded
        HAVING n >= 10
        ORDER BY date DESC
        LIMIT 1
    """).fetchone()

if not pair:
    print("FAIL: no (date, sector_encoded) pair with >= 10 non-NULL rsi_14 rows found.")
    sys.exit(1)

test_date   = pair["date"]
test_sector = pair["sector_encoded"]

with database.connection() as conn:
    rows = conn.execute("""
        SELECT ticker, rsi_14, rsi_14_vs_sector
        FROM features
        WHERE date = ?
          AND sector_encoded = ?
          AND rsi_14 IS NOT NULL
    """, (test_date, test_sector)).fetchall()

# Recompute sector mean from the same set of non-NULL rsi_14 values
rsi_vals     = [r["rsi_14"] for r in rows]
sector_mean  = sum(rsi_vals) / len(rsi_vals)

for r in rows:
    stored = r["rsi_14_vs_sector"]
    expected = r["rsi_14"] - sector_mean

    if stored is None:
        # Only NULL if the group had < 10 valid rows — but we selected a
        # group with >= 10, so NULL here means a logic error.
        print(
            f"FAIL: rsi_14_vs_sector is NULL for {r['ticker']} on {test_date} "
            f"(sector {test_sector}) despite group size {len(rows)} >= 10."
        )
        sys.exit(1)

    if abs(stored - expected) > 1e-6:
        print(
            f"FAIL: rsi_14_vs_sector mismatch for {r['ticker']} on {test_date}. "
            f"Expected {expected:.8f}, stored {stored:.8f} "
            f"(diff {abs(stored - expected):.2e})."
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
print("PASS")
