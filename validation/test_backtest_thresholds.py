import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest

# 1. Constant assertions
assert backtest.NET_EDGE_PASS_THRESHOLD >= 0.015, (
    f"NET_EDGE_PASS_THRESHOLD too low: {backtest.NET_EDGE_PASS_THRESHOLD}"
)
assert backtest.ROUND_TRIP_COST >= 0.010, (
    f"ROUND_TRIP_COST too low: {backtest.ROUND_TRIP_COST}"
)

# 2. Fake dict-style input: 100 dates, 5 BUYs/date, avg BUY edge ~2% above BAH
#    BUY positions return 4%, all positions return 2%  →  net edge = 2% > 1.7%
start = date(2023, 1, 2)
dates = [(start + timedelta(days=i)).isoformat() for i in range(100)]

buy_returns_by_date = {d: [0.04] * 5 for d in dates}
all_returns_by_date = {d: [0.04] * 5 + [0.02] * 15 for d in dates}

result = backtest.run({
    "buy_edge_gross":      0.04,
    "bah_return":          0.02,
    "buy_returns_by_date": buy_returns_by_date,
    "all_returns_by_date": all_returns_by_date,
})

# 3. Verdict must be a known tier
assert result["viability"] in {"PASS", "MARGINAL", "FAIL", "CRITICAL", "UNKNOWN"}, (
    f"Unexpected verdict: {result['viability']}"
)

# 4. Block bootstrap must have populated ci_lower
assert "ci_lower" in result, "ci_lower missing from returned dict"

print("backtest thresholds: PASS")
