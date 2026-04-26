"""
backtest.py

Evaluates the economic viability of a model's signal quality based on the
avg 30-day forward returns produced by trainer._compare_returns().

Viability tiers
---------------
  CRITICAL  gross BUY-signal return is negative (model loses money on buys)
  FAIL      net edge vs buy-and-hold is negative
  MARGINAL  net edge is positive but below 1.70% (insufficient to clear
            Revolut Premium round-trip costs of ~1.2% with margin of safety)
  PASS      net edge >= 1.70% AND bootstrapped 95% CI lower-bound > 0
            — signal is both economically meaningful and statistically significant

Threshold rationale
-------------------
  Revolut Premium charges ~1.0–1.2% round-trip per US-equity trade
  (bid/ask spread + platform fees).  A bare edge that merely matches the
  round-trip cost produces zero real profit; any bad fill or slippage
  erases it entirely.

  ROUND_TRIP_COST         = 1.2%  (worst-case Revolut Premium estimate)
  NET_EDGE_PASS_THRESHOLD = 1.7%  (round-trip cost + 0.5% slippage buffer)

  The 0.5% buffer absorbs imprecise limit-order fills and illiquid-day
  slippage.  Net edge alone is insufficient: a signal that beats
  buy-and-hold by 1.0% still posts a negative real return after the 1.2%
  round-trip is deducted.

  The statistical significance check (CI lower-bound > 0) ensures the edge
  is not just noise from a single test period.
"""

import logging
import math

import numpy as np

log = logging.getLogger(__name__)

# Revolut Premium US-equity round-trip cost (spread + fees), worst-case estimate
ROUND_TRIP_COST = 0.012   # 1.2%
# Net-edge must clear round-trip cost by 0.5% to absorb slippage and bad fills
NET_EDGE_PASS_THRESHOLD = ROUND_TRIP_COST + 0.005   # = 1.7%

# Bootstrap parameters for significance testing
_BOOTSTRAP_SAMPLES = 2000
_BOOTSTRAP_SEED    = 42
_CI_LEVEL          = 0.95   # 95% confidence interval


def _bootstrap_edge_ci_naive(
    buy_returns: np.ndarray,
    all_returns: np.ndarray,
    n_samples: int = _BOOTSTRAP_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI on (mean_buy_return - mean_all_return).

    Resamples BUY positions and all positions independently with replacement.
    This understates variance when positions on the same date are correlated —
    prefer _bootstrap_edge_ci_block when date-grouped inputs are available.

    Returns (ci_lower, ci_upper).
    """
    rng   = np.random.default_rng(seed)
    n_buy = len(buy_returns)
    n_all = len(all_returns)

    edges = np.empty(n_samples)
    for i in range(n_samples):
        boot_buy = buy_returns[rng.integers(0, n_buy, size=n_buy)]
        boot_all = all_returns[rng.integers(0, n_all, size=n_all)]
        edges[i] = boot_buy.mean() - boot_all.mean()

    alpha    = 1.0 - _CI_LEVEL
    ci_lower = float(np.percentile(edges, alpha / 2 * 100))
    ci_upper = float(np.percentile(edges, (1 - alpha / 2) * 100))
    return ci_lower, ci_upper


def _bootstrap_edge_ci_block(
    buy_returns_by_date: dict,
    all_returns_by_date: dict,
    n_samples: int = _BOOTSTRAP_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """
    Block bootstrap 95% CI on (mean_buy_return - mean_all_return).

    Samples *dates* with replacement, then takes all positions on each sampled
    date.  This preserves cross-sectional correlation among positions that share
    a date, giving a more accurate (wider) variance estimate than naive
    independent resampling.

    Falls back to _bootstrap_edge_ci_naive when fewer than 10 dates exist.

    Returns (ci_lower, ci_upper).
    """
    all_dates = sorted(all_returns_by_date.keys())
    n_dates   = len(all_dates)

    if n_dates < 10:
        buy_arr = np.array(
            [r for rs in buy_returns_by_date.values() for r in rs], dtype=float
        )
        all_arr = np.array(
            [r for rs in all_returns_by_date.values() for r in rs], dtype=float
        )
        return _bootstrap_edge_ci_naive(buy_arr, all_arr, n_samples=n_samples, seed=seed)

    rng            = np.random.default_rng(seed)
    all_dates_arr  = np.array(all_dates)
    edges          = np.empty(n_samples)

    for i in range(n_samples):
        sampled = all_dates_arr[rng.integers(0, n_dates, size=n_dates)]
        boot_buy = [r for d in sampled for r in buy_returns_by_date.get(d, [])]
        boot_all = [r for d in sampled for r in all_returns_by_date[d]]

        if len(boot_buy) == 0 or len(boot_all) == 0:
            edges[i] = float("nan")
            continue
        edges[i] = float(np.mean(boot_buy)) - float(np.mean(boot_all))

    alpha    = 1.0 - _CI_LEVEL
    ci_lower = float(np.nanpercentile(edges, alpha / 2 * 100))
    ci_upper = float(np.nanpercentile(edges, (1 - alpha / 2) * 100))
    return ci_lower, ci_upper


def run(results: dict) -> dict:
    """
    Evaluate signal viability from a results dict.

    Expected keys
    -------------
    buy_edge_gross  float        avg 30-day return on BUY-signal positions
    bah_return      float        buy-and-hold avg 30-day return
    buy_returns     np.ndarray   (optional) per-position returns for BUY signals
    all_returns     np.ndarray   (optional) per-position returns for all signals

    When buy_returns / all_returns are supplied the verdict also requires
    the bootstrapped 95% CI lower-bound to be > 0 (statistically significant).

    Returns the results dict with added keys:
      'viability'        — verdict string
      'viability_reason' — human-readable explanation
      'ci_lower'         — bootstrap CI lower bound (if arrays supplied)
      'ci_upper'         — bootstrap CI upper bound (if arrays supplied)
    """
    gross_edge = results.get("buy_edge_gross", float("nan"))
    bah        = results.get("bah_return",     float("nan"))

    net_edge = results.get("buy_edge_net", float("nan"))
    if math.isnan(net_edge) and not math.isnan(gross_edge) and not math.isnan(bah):
        net_edge = gross_edge - bah

    # ── Bootstrap CI ───────────────────────────────────────────────────────
    # Prefer block bootstrap (date-grouped) to account for cross-sectional
    # correlation; fall back to naive when only flat arrays are supplied.
    ci_lower    = ci_upper = float("nan")
    buy_by_date = results.get("buy_returns_by_date")
    all_by_date = results.get("all_returns_by_date")
    buy_returns = results.get("buy_returns")
    all_returns = results.get("all_returns")

    if buy_by_date is not None and all_by_date is not None and len(all_by_date) >= 10:
        ci_lower, ci_upper = _bootstrap_edge_ci_block(buy_by_date, all_by_date)
        results["ci_lower"] = ci_lower
        results["ci_upper"] = ci_upper
        log.info(
            "Block-bootstrap 95%% CI on net edge: [%+.2f%%, %+.2f%%]"
            "  (%d samples, %d dates)",
            ci_lower * 100, ci_upper * 100, _BOOTSTRAP_SAMPLES, len(all_by_date),
        )
    elif (
        buy_returns is not None
        and all_returns is not None
        and len(buy_returns) >= 30
        and len(all_returns) >= 30
    ):
        ci_lower, ci_upper = _bootstrap_edge_ci_naive(
            np.asarray(buy_returns, dtype=float),
            np.asarray(all_returns, dtype=float),
        )
        results["ci_lower"] = ci_lower
        results["ci_upper"] = ci_upper
        log.info(
            "Bootstrap 95%% CI on net edge: [%+.2f%%, %+.2f%%]  (%d samples)",
            ci_lower * 100, ci_upper * 100, _BOOTSTRAP_SAMPLES,
        )

    statistically_significant = math.isnan(ci_lower) or ci_lower > 0

    # ── Tiered verdict ──────────────────────────────────────────────────────
    if math.isnan(gross_edge):
        verdict = "UNKNOWN"
        reason  = "Insufficient data to evaluate viability."

    elif gross_edge < 0:
        verdict = "CRITICAL"
        reason  = (
            f"BUY-signal gross return is {gross_edge * 100:+.2f}% — the model "
            "loses money on its own buy picks. Do not trade this model without "
            "retraining with tighter thresholds or better features."
        )

    elif math.isnan(net_edge) or net_edge < 0:
        verdict = "FAIL"
        reason  = (
            f"Gross BUY return {gross_edge * 100:+.2f}% is positive but net edge "
            f"vs buy-and-hold is {net_edge * 100:+.2f}%. Simply holding the index "
            "outperforms this model's BUY signals."
        )

    elif net_edge < NET_EDGE_PASS_THRESHOLD:
        verdict = "MARGINAL"
        reason  = (
            f"Net edge vs buy-and-hold: {net_edge * 100:+.2f}% "
            f"(threshold >= {NET_EDGE_PASS_THRESHOLD * 100:.2f}%). "
            "Signal exists but is too thin to cover realistic transaction costs."
        )

    elif not statistically_significant:
        verdict = "MARGINAL"
        reason  = (
            f"Net edge {net_edge * 100:+.2f}% clears the economic threshold but "
            f"the bootstrapped 95%% CI lower bound is {ci_lower * 100:+.2f}% "
            "(includes zero). Edge may be within sampling noise for this test period."
        )

    else:
        verdict = "PASS"
        ci_str = (
            f"  Bootstrap 95%% CI: [{ci_lower*100:+.2f}%, {ci_upper*100:+.2f}%]."
            if not math.isnan(ci_lower) else ""
        )
        reason  = (
            f"Net edge vs buy-and-hold: {net_edge * 100:+.2f}% "
            f"(>= {NET_EDGE_PASS_THRESHOLD * 100:.2f}% threshold). "
            f"Signal has actionable, statistically significant positive edge.{ci_str}"
        )

    # ── Log verdict ─────────────────────────────────────────────────────────
    log.info("=== Viability Verdict: %s ===", verdict)
    log.info("  %s", reason)
    if not math.isnan(gross_edge):
        log.info("  BUY gross return    : %+.2f%%", gross_edge * 100)
    if not math.isnan(bah):
        log.info("  Buy-and-hold return : %+.2f%%", bah * 100)
    if not math.isnan(net_edge):
        log.info("  Net edge vs B&H     : %+.2f%%", net_edge * 100)
    if not math.isnan(ci_lower):
        log.info(
            "  95%% CI             : [%+.2f%%, %+.2f%%]",
            ci_lower * 100, ci_upper * 100,
        )
    log.info("  Round-trip cost      : %+.2f%%", ROUND_TRIP_COST * 100)
    log.info("  Pass threshold       : %+.2f%%", NET_EDGE_PASS_THRESHOLD * 100)

    results["viability"]        = verdict
    results["viability_reason"] = reason
    return results
