"""
compare_models.py

A/B comparison of UniversalStockModel (classifier) vs UniversalRanker on the
sealed holdout slice.  Computes full portfolio-level metrics, assigns verdict
tiers per assessment section 7.3, applies the decision rules, and writes the
chosen model to models/production_model.json.

Actual model paths
------------------
  Classifier : models/universal_model.joblib   (UniversalStockModel / XGBClassifier)
  Ranker     : models/universal_ranker.joblib  (UniversalRanker / XGBRanker)
"""

import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import database
from ranker_trainer import UniversalRanker
from trainer import LABEL_BUY, UniversalStockModel

# ---------------------------------------------------------------------------
ROUND_TRIP_COST = 0.012   # Revolut Premium round-trip (Prompt 2 spec)
TOP_N_RANKER    = 20

MODEL_DIR  = Path("models")
CLF_PATH   = MODEL_DIR / "universal_model.joblib"    # classifier default path
RKR_PATH   = MODEL_DIR / "universal_ranker.joblib"   # ranker default path


# ---------------------------------------------------------------------------
# NAV simulation
# ---------------------------------------------------------------------------

def simulate_nav(
    buy_returns_by_date: dict,
    all_dates: list,
    starting_capital: float = 10_000,
    round_trip_cost: float = 0.012,
) -> pd.DataFrame:
    """
    Simulate a portfolio NAV over *all_dates*.

    buy_returns_by_date : {date_str: [fwd_return per BUY position]}
    all_dates           : sorted list of every holdout date
    Equal-weight the BUY basket on each date that has buys.
    Apply round_trip_cost on every rebalance (every date with buys).
    Dates without buys → cash (0 % return, no cost).

    Returns DataFrame with columns: date, nav, period_return.
    """
    nav  = starting_capital
    rows = []
    for date in all_dates:
        rets = buy_returns_by_date.get(date, [])
        if rets:
            period_ret = float(np.mean(rets)) - round_trip_cost
        else:
            period_ret = 0.0
        nav *= 1.0 + period_ret
        rows.append({"date": date, "nav": nav, "period_return": period_ret})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SPY benchmark NAV
# ---------------------------------------------------------------------------

def build_spy_nav(all_dates: list, starting_capital: float = 10_000) -> pd.DataFrame:
    """
    Build SPY NAV from the prices table over the holdout window.
    Forward-fills on dates with no SPY close (market holiday, missing row).
    Falls back to flat (0 % return) if SPY is absent from the table.
    """
    try:
        with database.connection() as conn:
            rows = conn.execute(
                "SELECT date, close FROM prices "
                "WHERE ticker IN ('SPY', '^SPY') ORDER BY date"
            ).fetchall()
        if not rows:
            raise ValueError("SPY ticker not found in prices table")

        spy = pd.DataFrame([dict(r) for r in rows])
        spy["date"] = spy["date"].astype(str)
        spy = spy.sort_values("date").set_index("date")
        close_map = spy["close"].to_dict()

        nav       = starting_capital
        prev_close: float | None = None
        result    = []
        for date in all_dates:
            close = close_map.get(date)
            if close is None:
                period_ret = 0.0
            elif prev_close is None:
                period_ret = 0.0
                prev_close = close
            else:
                period_ret = (close - prev_close) / prev_close
                prev_close = close
            nav *= 1.0 + period_ret
            result.append({"date": date, "nav": nav, "period_return": period_ret})
        return pd.DataFrame(result)

    except Exception as exc:
        print(f"  [SPY benchmark unavailable: {exc} — using flat 0 % benchmark]")
        return pd.DataFrame(
            [{"date": d, "nav": starting_capital, "period_return": 0.0} for d in all_dates]
        )


# ---------------------------------------------------------------------------
# Block bootstrap CI
# ---------------------------------------------------------------------------

def _block_bootstrap_ci(
    series: np.ndarray,
    block_size: int = 20,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Block bootstrap 95 % CI on the mean of *series*.

    Samples contiguous blocks of length *block_size* with replacement to
    preserve serial correlation (as required by assessment anti-bias notes).
    Falls back to iid bootstrap when series is shorter than block_size.
    """
    n = len(series)
    if n < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)

    if n < block_size:
        boot = np.array(
            [np.mean(rng.choice(series, size=n, replace=True)) for _ in range(n_resamples)]
        )
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    starts   = np.arange(n - block_size + 1)
    n_blocks = math.ceil(n / block_size)
    boot     = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen  = rng.choice(starts, size=n_blocks, replace=True)
        sample  = np.concatenate([series[s : s + block_size] for s in chosen])[:n]
        boot[i] = np.mean(sample)

    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def portfolio_metrics(
    nav_df: pd.DataFrame,
    spy_nav_df: pd.DataFrame,
    n_rebalances: int,
) -> dict:
    """
    Compute full metric set for one model's NAV series.

    Returns keys:
        cagr, sharpe (rf=0), max_dd, calmar,
        alpha_annualized, alpha_ci_lower_95, alpha_ci_upper_95 (block bootstrap),
        turnover_annualized
    """
    nan_result = {k: float("nan") for k in [
        "cagr", "sharpe", "max_dd", "calmar",
        "alpha_annualized", "alpha_ci_lower_95", "alpha_ci_upper_95",
        "turnover_annualized",
    ]}
    if nav_df.empty or len(nav_df) < 2:
        return nan_result

    # Calendar years over the full holdout window
    try:
        start_dt = pd.to_datetime(nav_df["date"].iloc[0])
        end_dt   = pd.to_datetime(nav_df["date"].iloc[-1])
        years    = max((end_dt - start_dt).days, 1) / 365.25
    except Exception:
        years = max(len(nav_df) / 252, 1e-6)

    final_nav       = float(nav_df["nav"].iloc[-1])
    cagr            = (final_nav / 10_000.0) ** (1.0 / years) - 1.0

    rets            = nav_df["period_return"].values.astype(float)
    n_periods       = len(rets)
    periods_per_yr  = n_periods / years

    mean_r = float(np.mean(rets))
    std_r  = float(np.std(rets, ddof=1)) if n_periods > 1 else 1e-12
    sharpe = (mean_r / (std_r + 1e-12)) * math.sqrt(periods_per_yr)

    nav_arr     = nav_df["nav"].values.astype(float)
    peak        = np.maximum.accumulate(nav_arr)
    drawdowns   = (nav_arr - peak) / peak
    max_dd      = float(np.min(drawdowns))

    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    # ── Alpha vs SPY (date-aligned excess returns, block bootstrap CI) ───────
    m_ret = dict(zip(nav_df["date"].astype(str), rets))
    s_ret = dict(zip(
        spy_nav_df["date"].astype(str),
        spy_nav_df["period_return"].values.astype(float),
    ))
    common = sorted(set(m_ret) & set(s_ret))

    if len(common) < 4:
        alpha_ann = ci_lo = ci_hi = float("nan")
    else:
        excess    = np.array([m_ret[d] - s_ret[d] for d in common])
        alpha_ann = float(np.mean(excess)) * periods_per_yr
        ci_lo_raw, ci_hi_raw = _block_bootstrap_ci(excess, block_size=20, n_resamples=1000)
        ci_lo = ci_lo_raw * periods_per_yr
        ci_hi = ci_hi_raw * periods_per_yr

    turnover = 2.0 * n_rebalances / years if years > 0 else float("nan")

    return {
        "cagr":               cagr,
        "sharpe":             sharpe,
        "max_dd":             max_dd,
        "calmar":             calmar,
        "alpha_annualized":   alpha_ann,
        "alpha_ci_lower_95":  ci_lo,
        "alpha_ci_upper_95":  ci_hi,
        "turnover_annualized": turnover,
    }


# ---------------------------------------------------------------------------
# Verdict tiers (assessment section 7.3)
# ---------------------------------------------------------------------------

def assign_verdict(m: dict) -> str:
    """
    STRONG   alpha > 0 AND ci_lower > 0 AND sharpe > 0.7 AND max_dd > -0.20
    PASS     alpha > 0 AND ci_lower > 0 AND sharpe > 0.4
    MARGINAL alpha > 0 AND ci_lower <= 0  (or ci unknown but alpha positive)
    FAIL     alpha <= 0
    """
    alpha  = m.get("alpha_annualized", float("nan"))
    ci_lo  = m.get("alpha_ci_lower_95", float("nan"))
    sharpe = m.get("sharpe", float("nan"))
    max_dd = m.get("max_dd", float("nan"))

    if math.isnan(alpha) or alpha <= 0:
        return "FAIL"

    ci_ok  = (not math.isnan(ci_lo)) and ci_lo > 0
    sh_ok  = (not math.isnan(sharpe))

    if ci_ok and sh_ok and sharpe > 0.7 and (not math.isnan(max_dd)) and max_dd > -0.20:
        return "STRONG"
    if ci_ok and sh_ok and sharpe > 0.4:
        return "PASS"
    return "MARGINAL"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float, d: int = 2) -> str:
    return f"{v * 100:+.{d}f}%" if not math.isnan(v) else "N/A"

def _f2(v: float) -> str:
    return f"{v:.2f}" if not math.isnan(v) else "N/A"

def _ci_str(lo: float, hi: float) -> str:
    if math.isnan(lo) or math.isnan(hi):
        return "N/A"
    return f"[{lo * 100:+.1f}%, {hi * 100:+.1f}%]"


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def apply_decision_rules(
    metrics_clf: dict,
    metrics_rkr: dict,
    verdict_clf: str,
    verdict_rkr: str,
) -> tuple[str, str]:
    """
    Apply decision rules in priority order.  Returns (chosen, rule_id).

    chosen : 'classifier' | 'ranker'
    rule   : 'R1' | 'R2' | 'R3' | 'R4' | 'R5'
    """
    TIER = {"STRONG": 3, "PASS": 2, "MARGINAL": 1, "FAIL": 0}
    clf_tier = TIER[verdict_clf]
    rkr_tier = TIER[verdict_rkr]

    # R1 — exactly one model is viable (PASS/STRONG)
    if (clf_tier >= 2) != (rkr_tier >= 2):
        chosen = "classifier" if clf_tier >= 2 else "ranker"
        return chosen, "R1"

    # Both viable (PASS or STRONG)
    if clf_tier >= 2 and rkr_tier >= 2:
        # R5 — both STRONG but materially different max drawdown (>10 pp)
        if verdict_clf == "STRONG" and verdict_rkr == "STRONG":
            dd_clf = abs(metrics_clf.get("max_dd", 0.0))
            dd_rkr = abs(metrics_rkr.get("max_dd", 0.0))
            if abs(dd_clf - dd_rkr) > 0.10:
                chosen = "classifier" if dd_clf < dd_rkr else "ranker"
                return chosen, "R5"

        # R2 — higher Sharpe wins
        s_clf = metrics_clf.get("sharpe", float("nan"))
        s_rkr = metrics_rkr.get("sharpe", float("nan"))
        s_clf = s_clf if not math.isnan(s_clf) else -999.0
        s_rkr = s_rkr if not math.isnan(s_rkr) else -999.0

        if abs(s_clf - s_rkr) > 0.05:
            chosen = "classifier" if s_clf > s_rkr else "ranker"
        else:
            # Tiebreaker 1: higher alpha CI lower bound
            ci_clf = metrics_clf.get("alpha_ci_lower_95", float("nan"))
            ci_rkr = metrics_rkr.get("alpha_ci_lower_95", float("nan"))
            ci_clf = ci_clf if not math.isnan(ci_clf) else -999.0
            ci_rkr = ci_rkr if not math.isnan(ci_rkr) else -999.0
            if abs(ci_clf - ci_rkr) > 1e-6:
                chosen = "classifier" if ci_clf > ci_rkr else "ranker"
            else:
                # Tiebreaker 2: RANKER (architectural alignment, assessment §4.2.1)
                chosen = "ranker"
        return chosen, "R2"

    # R3 — both MARGINAL
    if clf_tier == 1 and rkr_tier == 1:
        return "ranker", "R3"

    # R4 — both FAIL (or mixed FAIL/MARGINAL where neither is viable)
    return "ranker", "R4"


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    # ── load both models ─────────────────────────────────────────────────────
    print("Loading models…")
    clf = UniversalStockModel.load()
    rkr = UniversalRanker.load()

    # ── load holdout slice ───────────────────────────────────────────────────
    split_dates   = clf.split_dates
    holdout_start = split_dates.get("test_end")
    print(f"Using holdout start (> test_end): {holdout_start}")

    all_feat_cols = list(dict.fromkeys(clf.feature_cols + rkr.feature_cols))
    feat_select   = ", ".join(f"f.{c}" for c in all_feat_cols)
    sql = f"""
        SELECT f.ticker, f.date, {feat_select},
               l.label, l.forward_return
        FROM   features f
        JOIN   labels   l ON l.ticker = f.ticker AND l.date = f.date
        WHERE  f.date > ?
        ORDER  BY f.date, f.ticker
    """
    with database.connection() as conn:
        rows = conn.execute(sql, (holdout_start,)).fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    print(f"Holdout rows: {len(df)}")
    if df.empty:
        print("No holdout data — abort.")
        sys.exit(1)

    holdout_dates = sorted(df["date"].astype(str).unique())
    n_dates       = len(holdout_dates)
    print(f"Holdout date range: {holdout_dates[0]} → {holdout_dates[-1]}  ({n_dates} dates)")

    fwd        = df["forward_return"].values.astype(float)
    mask_valid = ~np.isnan(fwd)

    # ── CLASSIFIER buy_returns_by_date ────────────────────────────────────────
    print("\nScoring classifier…")
    proba_clf = clf.predict_proba(df)
    preds_clf = UniversalStockModel._apply_buy_percentile(proba_clf, clf.buy_top_fraction)
    buy_clf   = preds_clf == LABEL_BUY

    clf_buy_by_date: dict[str, list] = {}
    for d, ret, is_buy, is_valid in zip(df["date"].astype(str), fwd, buy_clf, mask_valid):
        if is_buy and is_valid:
            clf_buy_by_date.setdefault(d, []).append(float(ret))

    # ── RANKER buy_returns_by_date ────────────────────────────────────────────
    print("Scoring ranker…")
    df_h           = df.copy()
    df_h["_score"] = rkr.model.predict(df_h[rkr.feature_cols].values.astype(float))
    df_h["_rank"]  = df_h.groupby("date")["_score"].rank(method="first", ascending=False)
    buy_rkr        = df_h["_rank"].values <= TOP_N_RANKER

    rkr_buy_by_date: dict[str, list] = {}
    for d, ret, is_buy, is_valid in zip(df["date"].astype(str), fwd, buy_rkr, mask_valid):
        if is_buy and is_valid:
            rkr_buy_by_date.setdefault(d, []).append(float(ret))

    # ── NAV simulations ───────────────────────────────────────────────────────
    print("Building NAV series…")
    spy_nav_df = build_spy_nav(holdout_dates)
    clf_nav_df = simulate_nav(clf_buy_by_date, holdout_dates, round_trip_cost=ROUND_TRIP_COST)
    rkr_nav_df = simulate_nav(rkr_buy_by_date, holdout_dates, round_trip_cost=ROUND_TRIP_COST)

    n_clf_reb = len(clf_buy_by_date)
    n_rkr_reb = len(rkr_buy_by_date)

    # ── portfolio metrics ─────────────────────────────────────────────────────
    print("Computing portfolio metrics (bootstrap CIs take a moment)…")
    metrics_clf = portfolio_metrics(clf_nav_df, spy_nav_df, n_clf_reb)
    metrics_rkr = portfolio_metrics(rkr_nav_df, spy_nav_df, n_rkr_reb)

    verdict_clf = assign_verdict(metrics_clf)
    verdict_rkr = assign_verdict(metrics_rkr)

    avg_buy_clf = float(np.mean([len(v) for v in clf_buy_by_date.values()])) if clf_buy_by_date else 0.0
    avg_buy_rkr = float(np.mean([len(v) for v in rkr_buy_by_date.values()])) if rkr_buy_by_date else 0.0

    # ── comparison table ──────────────────────────────────────────────────────
    W = 19
    print()
    print("=" * 65)
    print(f"  {'METRIC':<22} {'CLASSIFIER':>{W}}  {'RANKER':>{W}}")
    print("-" * 65)
    print(f"  {'Holdout dates':<22} {n_dates:>{W}}  {n_dates:>{W}}")
    print(f"  {'BUY positions/date':<22} {avg_buy_clf:>{W}.1f}  {avg_buy_rkr:>{W}.1f}")
    print(f"  {'CAGR':<22} {_pct(metrics_clf['cagr']):>{W}}  {_pct(metrics_rkr['cagr']):>{W}}")
    print(f"  {'Sharpe':<22} {_f2(metrics_clf['sharpe']):>{W}}  {_f2(metrics_rkr['sharpe']):>{W}}")
    print(f"  {'Max DD':<22} {_pct(metrics_clf['max_dd']):>{W}}  {_pct(metrics_rkr['max_dd']):>{W}}")
    print(f"  {'Alpha (ann)':<22} {_pct(metrics_clf['alpha_annualized']):>{W}}  {_pct(metrics_rkr['alpha_annualized']):>{W}}")
    clf_ci = _ci_str(metrics_clf["alpha_ci_lower_95"], metrics_clf["alpha_ci_upper_95"])
    rkr_ci = _ci_str(metrics_rkr["alpha_ci_lower_95"], metrics_rkr["alpha_ci_upper_95"])
    print(f"  {'Alpha 95% CI':<22} {clf_ci:>{W}}  {rkr_ci:>{W}}")
    print(f"  {'Turnover (ann)':<22} {_f2(metrics_clf['turnover_annualized']):>{W}}  {_f2(metrics_rkr['turnover_annualized']):>{W}}")
    print(f"  {'Verdict':<22} {verdict_clf:>{W}}  {verdict_rkr:>{W}}")
    print("=" * 65)

    # ── decision ──────────────────────────────────────────────────────────────
    chosen, rule = apply_decision_rules(metrics_clf, metrics_rkr, verdict_clf, verdict_rkr)

    reason_map = {
        "R1": "one model PASS/STRONG, other MARGINAL/FAIL — pick the viable model",
        "R2": "both viable — higher Sharpe wins (alpha CI lower bound as tiebreaker)",
        "R3": "both MARGINAL — prefer ranker (objective aligned with cross-sectional inference)",
        "R4": "both FAIL — prefer ranker (same alignment reason); NO demonstrated edge on holdout",
        "R5": "both STRONG but max drawdown differs >10 pp — prefer lower-drawdown model",
    }

    if rule == "R4":
        print()
        print("!" * 65)
        print("  WARNING: BOTH MODELS FAIL on the holdout — no demonstrated edge.")
        print("  Do NOT proceed to real-money trading.  Paper-trade only (Prompt 13).")
        print("  Revisit sector-relative / volatility-normalized features (Prompt 5).")
        print("!" * 65)

    print()
    print("=" * 65)
    print(f"  PRODUCTION MODEL: {chosen.upper()}  (rule {rule})")
    print(f"  Reason: {reason_map[rule]}")
    print("=" * 65)

    # ── persist decision ──────────────────────────────────────────────────────
    chosen_path = str(CLF_PATH) if chosen == "classifier" else str(RKR_PATH)
    # normalise to forward-slash relative path
    chosen_path = chosen_path.replace("\\", "/")

    decision = {
        "chosen":              chosen,
        "chosen_path":         chosen_path,
        "decision_rule":       rule,
        "decided_at":          datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "holdout_metrics": {
            "classifier": metrics_clf,
            "ranker":      metrics_rkr,
        },
        "verdict_classifier":  verdict_clf,
        "verdict_ranker":      verdict_rkr,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "production_model.json"
    with open(out_path, "w") as fh:
        json.dump(decision, fh, indent=2)
    print(f"\nDecision written → {out_path}")


if __name__ == "__main__":
    main()
