"""
compare_models.py

Thin driver: loads both models, runs honest portfolio-level NAV backtest on
the sealed holdout window, and picks the production model by decision rule.

The holdout window starts at the LATER of the two models' test_end dates so
neither model sees data it was tuned on.  Both backtests are evaluated on the
same window.

NAV is built from realized daily close-to-close returns in the prices table.
forward_return from the labels table is NOT used for NAV simulation.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
import database
from ranker_trainer import UniversalRanker
from trainer import UniversalStockModel
from portfolio_backtest import PortfolioBacktest

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("Loading models…")
clf = UniversalStockModel.load()
rkr = UniversalRanker.load()

# ---------------------------------------------------------------------------
# Determine holdout window
# Use the LATER of the two test_end dates so neither model sees tuned data.
# ---------------------------------------------------------------------------
clf_end = pd.to_datetime(clf.split_dates.get("test_end"))
rkr_end = pd.to_datetime(rkr.split_dates.get("test_end"))
later_end = max(clf_end, rkr_end)
holdout_start = (later_end + pd.Timedelta(days=1)).date().isoformat()

with database.connection() as conn:
    end_date = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]

print(f"Classifier test_end : {clf_end.date()}")
print(f"Ranker    test_end  : {rkr_end.date()}")
print(f"Holdout window      : {holdout_start} → {end_date}")

# Guard: empty or degenerate window
holdout_start_dt = pd.to_datetime(holdout_start)
end_date_dt      = pd.to_datetime(end_date)
if holdout_start_dt >= end_date_dt:
    print(f"ERROR: holdout_start ({holdout_start}) >= end_date ({end_date}).")
    print("No holdout window available — retrain or extend price history.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Run both backtests
# ---------------------------------------------------------------------------
results: dict = {}

for name, model, mode in [
    ("classifier", clf, "classifier"),
    ("ranker",     rkr, "ranker"),
]:
    print(f"\n{'='*60}")
    print(f"Running {name.upper()} backtest…")
    print("="*60)

    pb      = PortfolioBacktest(model=model, scoring_mode=mode)
    nav_df  = pb.run(holdout_start, end_date)
    spy_nav = pb._load_spy_nav(holdout_start, end_date)
    bench   = PortfolioBacktest.vs_benchmark(nav_df["nav"], spy_nav)
    metrics = PortfolioBacktest.metrics(nav_df["nav"])

    combined          = {**metrics, **bench}
    combined["verdict"] = PortfolioBacktest.assign_verdict(combined)

    print(pb.summary_text(nav_df["nav"], spy_nav, combined))
    results[name] = combined

# ---------------------------------------------------------------------------
# Sanity check: flag contamination if metrics are out of realistic range
# ---------------------------------------------------------------------------
for name, m in results.items():
    cagr   = m.get("cagr", float("nan"))
    sharpe = m.get("sharpe", float("nan"))
    import math
    if not math.isnan(cagr) and abs(cagr) > 10.0:          # > 1000% CAGR
        print(f"\n*** WARNING [{name}]: CAGR={cagr:.1%} is outside realistic range. "
              "A contamination bug may still be present. ***")
    if not math.isnan(sharpe) and abs(sharpe) > 5.0:
        print(f"\n*** WARNING [{name}]: Sharpe={sharpe:.2f} > 5 — check for look-ahead bias. ***")

# ---------------------------------------------------------------------------
# Decision rule
# Both PASS/STRONG → higher ci_lower wins (more statistically defensible alpha).
# Exactly one passes → pick it.
# Neither passes   → NONE (exit 2).
# ---------------------------------------------------------------------------
def _passing(v: str) -> bool:
    return v in ("PASS", "STRONG")

clf_ok = _passing(results["classifier"]["verdict"])
rkr_ok = _passing(results["ranker"]["verdict"])

if clf_ok and rkr_ok:
    import math as _math
    clf_ci = results["classifier"].get("ci_lower", float("nan"))
    rkr_ci = results["ranker"].get("ci_lower", float("nan"))
    clf_ci = clf_ci if not _math.isnan(clf_ci) else -999.0
    rkr_ci = rkr_ci if not _math.isnan(rkr_ci) else -999.0
    winner = "ranker" if rkr_ci > clf_ci else "classifier"
elif clf_ok:
    winner = "classifier"
elif rkr_ok:
    winner = "ranker"
else:
    winner = "NONE"

# ---------------------------------------------------------------------------
# Persist decision
# ---------------------------------------------------------------------------
import math as _math2  # already imported above but keep explicit for clarity

Path("models").mkdir(exist_ok=True)
Path("models/production_model.json").write_text(
    json.dumps(
        {
            "production_model": winner,
            "holdout_start":    holdout_start,
            "holdout_end":      end_date,
            "classifier":       results["classifier"],
            "ranker":           results["ranker"],
        },
        indent=2,
        default=str,
    )
)

print(f"\n{'='*60}")
print(f"Production decision: {winner}")
print("="*60)
print("Decision written → models/production_model.json")

if winner == "NONE":
    print("\nNeither model achieved PASS or STRONG on the honest holdout.")
    print("This is the system working correctly. Do NOT trade either model.")
    sys.exit(2)
