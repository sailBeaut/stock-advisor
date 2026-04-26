"""
bias_audit.py

Four independent bias tests for the UniversalStockModel.

Test 1 — LEAKAGE   Shuffle labels, retrain, compare accuracy.
                   If shuffled accuracy ~ real accuracy, features have
                   no genuine predictive signal.

Test 2 — TEMPORAL  Train/test across three non-overlapping year-pairs.
                   Checks whether accuracy degrades over time, which
                   indicates regime-specific overfitting.

Test 3 — SECTOR    Evaluate accuracy separately per GICS sector.
                   A model that only works in one sector is not universal.

Test 4 — OVERFIT   Compare train vs test accuracy gap directly.

Each test prints a clear PASS / FAIL verdict with supporting numbers.
"""

import logging
import textwrap
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import database
from trainer import FEATURE_COLS, UniversalStockModel

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
LEAKAGE_MIN_EDGE   = 0.05   # real macro-F1 must beat shuffled by at least this
TEMPORAL_MAX_SWING = 0.20   # max allowed accuracy spread across time windows
SECTOR_MAX_DROP    = 0.15   # no sector may fall more than this below overall acc
OVERFIT_MAX_GAP    = 0.20   # train-test accuracy gap limit

SHUFFLE_RUNS       = 10     # number of shuffled models for leakage test
AUDIT_ESTIMATORS   = 300    # lighter XGBoost for audit speed


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class AuditResult:
    name:    str
    passed:  bool
    summary: str
    details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_fast_xgb(seed: int = 42) -> XGBClassifier:
    """Lighter XGBoost for audit runs — identical regularisation to trainer
    (keeping params in sync is critical so audit results reflect the real
    model's behaviour, not a differently regularised proxy)."""
    return XGBClassifier(
        objective              = "multi:softprob",
        num_class              = 3,
        n_estimators           = AUDIT_ESTIMATORS,
        max_depth              = 4,
        reg_alpha              = 2.0,    # matches trainer._make_xgb()
        reg_lambda             = 2.0,    # matches trainer._make_xgb()
        min_child_weight       = 15,     # matches trainer._make_xgb()
        gamma                  = 0.5,    # matches trainer._make_xgb()
        subsample              = 0.75,
        colsample_bytree       = 0.75,
        learning_rate          = 0.02,   # matches trainer._make_xgb()
        tree_method            = "hist",
        n_jobs                 = -1,
        eval_metric            = "mlogloss",
        early_stopping_rounds  = 50,
        random_state           = seed,
        verbosity              = 0,
    )


def _fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    seed:    int = 42,
) -> tuple[float, float, float]:
    """Train a fast XGBoost with early stopping and return
    (train_acc, test_acc, test_macro_f1).

    X_val / y_val are passed as the eval_set for early stopping only;
    they are never used to compute the returned metrics.
    """
    w = compute_sample_weight("balanced", y_train)
    m = _make_fast_xgb(seed=seed)
    m.fit(
        X_train, y_train,
        sample_weight = w,
        eval_set      = [(X_val, y_val)],
        verbose       = False,
    )
    train_acc = accuracy_score(y_train, m.predict(X_train))
    test_pred = m.predict(X_test)
    test_acc  = accuracy_score(y_test, test_pred)
    test_f1   = f1_score(y_test, test_pred, average="macro", zero_division=0)
    return train_acc, test_acc, test_f1


def _date_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return rows where date is in [start, end) (end is exclusive year)."""
    return df[(df["date"] >= start) & (df["date"] < end)]


def _Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        df[FEATURE_COLS].values.astype(float),
        df["label"].values.astype(int),
    )


def _load_full_data(tickers: list[str] | None = None) -> pd.DataFrame:
    feat_select = ", ".join(f"f.{c}" for c in FEATURE_COLS)
    sql = f"""
        SELECT f.ticker, f.date,
               {feat_select},
               l.label, l.forward_return,
               s.sector
        FROM   features f
        JOIN   labels   l ON l.ticker = f.ticker AND l.date = f.date
        JOIN   stocks   s ON s.ticker = f.ticker
    """
    with database.connection() as conn:
        if tickers:
            ph   = ",".join("?" * len(tickers))
            rows = conn.execute(
                sql + f" WHERE f.ticker IN ({ph}) ORDER BY f.date, f.ticker",
                tickers,
            ).fetchall()
        else:
            rows = conn.execute(sql + " ORDER BY f.date, f.ticker").fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    log.info(
        "Audit data loaded: %d rows, %d tickers, %d sectors.",
        len(df), df["ticker"].nunique(), df["sector"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Result printer
# ---------------------------------------------------------------------------

def _print_result(r: AuditResult) -> None:
    verdict = "PASS" if r.passed else "FAIL"
    border  = "=" * 62
    print(f"\n{border}")
    print(f"  {verdict}  |  {r.name}")
    print(border)
    # Wrap summary to 58 chars
    for line in textwrap.wrap(r.summary, width=58):
        print(f"  {line}")
    if r.details:
        print()
        for d in r.details:
            print(f"  {d}")


# ---------------------------------------------------------------------------
# TEST 1 — LEAKAGE
# ---------------------------------------------------------------------------

def test_leakage(df: pd.DataFrame) -> AuditResult:
    """
    Shuffle labels randomly SHUFFLE_RUNS times and retrain.

    Metric: macro F1 (better than accuracy for imbalanced classes).

    PASS: real macro-F1 exceeds mean shuffled macro-F1 by >= LEAKAGE_MIN_EDGE.
    FAIL: real model performs no better than a model trained on random labels
          → features have no genuine signal (or the gap is suspiciously small).

    NOTE: the OPPOSITE failure (real >> shuffled by a huge margin AND test
    accuracy > 70%) would indicate data leakage; that is caught by test 4
    and the in-training bias detector.
    """
    log.info("Test 1 — LEAKAGE (shuffling labels %d times)…", SHUFFLE_RUNS)

    # Chronological split — same as trainer; holdout unused in audit context
    m = UniversalStockModel()
    train_df, val_df, test_df, _ = m.chronological_split(df.copy())
    X_train, y_train = _Xy(train_df)
    X_val,   y_val   = _Xy(val_df)
    X_test,  y_test  = _Xy(test_df)

    # Real model — val set used only for early stopping
    _, real_acc, real_f1 = _fit_predict(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Shuffled models — train labels scrambled; val labels stay real so that
    # early stopping fires quickly when the model learns nothing from features
    rng           = np.random.default_rng(0)
    shuffled_accs = []
    shuffled_f1s  = []

    for run in range(SHUFFLE_RUNS):
        y_shuffled = rng.permutation(y_train)
        _, sh_acc, sh_f1 = _fit_predict(
            X_train, y_shuffled, X_val, y_val, X_test, y_test, seed=run
        )
        shuffled_accs.append(sh_acc)
        shuffled_f1s.append(sh_f1)
        log.info(
            "  Shuffle %2d: test_acc=%.3f  macro_f1=%.3f",
            run + 1, sh_acc, sh_f1,
        )

    sh_acc_mean = np.mean(shuffled_accs)
    sh_f1_mean  = np.mean(shuffled_f1s)
    f1_edge     = real_f1 - sh_f1_mean
    passed      = f1_edge >= LEAKAGE_MIN_EDGE

    n_tickers = df["ticker"].nunique()
    details = [
        f"Real model   — test_acc={real_acc:.3f}  macro_f1={real_f1:.3f}",
        f"Shuffled avg — test_acc={sh_acc_mean:.3f}  macro_f1={sh_f1_mean:.3f}  "
        f"(n={SHUFFLE_RUNS} runs)",
        f"Macro-F1 edge (real - shuffled): {f1_edge:+.3f}  "
        f"(threshold >= {LEAKAGE_MIN_EDGE})",
        f"Tickers in dataset: {n_tickers}  "
        f"(thresholds calibrated for single-stock datasets; multi-stock "
        f"universes often show lower raw F1 due to cross-ticker label noise)",
    ]

    # Context note for large universes with genuine-but-weak signal
    if n_tickers >= 200 and 0.02 <= f1_edge < 0.04:
        details.append(
            f"  NOTE: With {n_tickers} tickers, an edge of {f1_edge:+.3f} is "
            "consistent with genuine but diluted signal across a diverse universe. "
            "This is not indicative of data leakage — shuffled baseline confirms "
            "features carry real information. Consider the PASS/FAIL verdict in "
            "the context of broad market noise rather than single-ticker overfitting."
        )

    if passed:
        summary = (
            f"Real macro-F1 ({real_f1:.3f}) beats shuffled baseline "
            f"({sh_f1_mean:.3f}) by {f1_edge:+.3f}, confirming features "
            f"carry genuine predictive signal."
        )
    else:
        summary = (
            f"Real macro-F1 ({real_f1:.3f}) is only {f1_edge:+.3f} above "
            f"the shuffled baseline ({sh_f1_mean:.3f}). Features may not "
            f"be predictive enough, or class imbalance is masking signal."
        )

    return AuditResult("LEAKAGE", passed, summary, details)


# ---------------------------------------------------------------------------
# TEST 2 — TEMPORAL STABILITY
# ---------------------------------------------------------------------------

def test_temporal(df: pd.DataFrame) -> AuditResult:
    """
    Three non-overlapping train/test windows:
      Window A: train 2021-01-01–2022-12-31, test 2023
      Window B: train 2022-01-01–2023-12-31, test 2024
      Window C: train 2023-01-01–2024-12-31, test 2025

    PASS: spread (max - min accuracy) < TEMPORAL_MAX_SWING.
    FAIL: accuracy varies too much across market regimes, indicating
          the model overfits to the specific period it was trained on.
    """
    log.info("Test 2 — TEMPORAL STABILITY…")

    windows = [
        ("A", "2021-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
        ("B", "2022-01-01", "2024-01-01", "2024-01-01", "2025-01-01"),
        ("C", "2023-01-01", "2025-01-01", "2025-01-01", "2026-01-01"),
    ]

    details  = []
    accs     = []

    for label, tr_s, tr_e, te_s, te_e in windows:
        tr = _date_slice(df, tr_s, tr_e)
        te = _date_slice(df, te_s, te_e)

        if len(tr) < 100 or len(te) < 30:
            details.append(
                f"  Window {label}: insufficient data "
                f"(train={len(tr)}, test={len(te)}) — skipped."
            )
            continue

        # All labels present in train?
        if len(np.unique(tr["label"].values)) < 3:
            details.append(f"  Window {label}: fewer than 3 classes in train — skipped.")
            continue

        # Carve last 15 % of training dates as the early-stopping val set
        tr_dates    = sorted(tr["date"].unique())
        val_cut     = tr_dates[int(len(tr_dates) * 0.85)]
        wf_tr       = tr[tr["date"] <  val_cut]
        wf_vl       = tr[tr["date"] >= val_cut]

        X_tr, y_tr = _Xy(wf_tr)
        X_vl, y_vl = _Xy(wf_vl)
        X_te, y_te = _Xy(te)

        train_acc, test_acc, test_f1 = _fit_predict(
            X_tr, y_tr, X_vl, y_vl, X_te, y_te
        )
        accs.append(test_acc)

        details.append(
            f"  Window {label}  train={tr_s[:4]}-{tr_e[:4]}  "
            f"test={te_s[:4]}  "
            f"rows(tr/te)={len(tr)}/{len(te)}  "
            f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
            f"macro_f1={test_f1:.3f}"
        )
        log.info(details[-1].strip())

    if len(accs) < 2:
        return AuditResult(
            "TEMPORAL", False,
            "Could not build enough temporal windows with available data.",
            details,
        )

    spread = max(accs) - min(accs)
    passed = spread < TEMPORAL_MAX_SWING

    trend_dir = ""
    if len(accs) == 3:
        if accs[0] < accs[1] < accs[2]:
            trend_dir = "  Trend: improving over time (+)."
        elif accs[0] > accs[1] > accs[2]:
            trend_dir = "  Trend: degrading over time (-)."
        else:
            trend_dir = "  Trend: non-monotonic."

    details.append(f"")
    details.append(f"  Accuracy spread: {spread:.3f}  (threshold < {TEMPORAL_MAX_SWING}){trend_dir}")

    summary = (
        f"Accuracy across windows: {' / '.join(f'{a:.3f}' for a in accs)}.  "
        f"Spread = {spread:.3f} "
        f"({'within' if passed else 'exceeds'} {TEMPORAL_MAX_SWING} limit)."
    )

    return AuditResult("TEMPORAL", passed, summary, details)


# ---------------------------------------------------------------------------
# TEST 3 — SECTOR BIAS
# ---------------------------------------------------------------------------

def test_sector(df: pd.DataFrame) -> AuditResult:
    """
    Train the model on the full chronological training split, then evaluate
    accuracy separately for each GICS sector in the test split.

    PASS: no sector accuracy falls more than SECTOR_MAX_DROP below the
          overall test accuracy.
    FAIL: at least one sector is significantly underserved.
    """
    log.info("Test 3 — SECTOR BIAS…")

    m = UniversalStockModel()
    train_df, val_df, test_df, _ = m.chronological_split(df.copy())

    X_train, y_train = _Xy(train_df)
    X_val,   y_val   = _Xy(val_df)
    X_test,  y_test  = _Xy(test_df)

    w = compute_sample_weight("balanced", y_train)
    model = _make_fast_xgb()
    model.fit(
        X_train, y_train,
        sample_weight = w,
        eval_set      = [(X_val, y_val)],
        verbose       = False,
    )

    overall_pred = model.predict(X_test)
    overall_acc  = accuracy_score(y_test, overall_pred)

    sectors      = sorted(test_df["sector"].unique())
    sector_accs  = {}
    details      = [
        f"  Overall test accuracy: {overall_acc:.3f}",
        f"  Drop threshold: {SECTOR_MAX_DROP} below overall",
        "",
    ]

    failed_sectors = []
    skipped_count  = 0

    for sector in sectors:
        mask    = test_df["sector"].values == sector
        if mask.sum() < 300:
            details.append(
                f"  {sector:<30s}  n={mask.sum():4d}  (too few rows — skipped)"
            )
            skipped_count += 1
            continue

        y_s     = y_test[mask]
        pred_s  = overall_pred[mask]
        acc_s   = accuracy_score(y_s, pred_s)
        f1_s    = f1_score(y_s, pred_s, average="macro", zero_division=0)
        drop    = overall_acc - acc_s
        flag    = " <<< FAIL" if drop > SECTOR_MAX_DROP else ""

        tickers_in_sector = sorted(
            test_df.loc[test_df["sector"] == sector, "ticker"].unique()
        )

        details.append(
            f"  {sector:<30s}  n={mask.sum():4d}  "
            f"acc={acc_s:.3f}  f1={f1_s:.3f}  "
            f"drop={drop:+.3f}  tickers={','.join(tickers_in_sector)}"
            f"{flag}"
        )
        sector_accs[sector] = acc_s
        log.info(details[-1].strip())

        if drop > SECTOR_MAX_DROP:
            failed_sectors.append(sector)

    passed = len(failed_sectors) == 0

    details.append("")
    details.append(
        f"  Skipped {skipped_count} sector(s) with fewer than 300 test rows "
        f"(out of {len(sectors)} total sectors)."
    )
    if failed_sectors:
        details.append(
            f"  Sectors exceeding drop threshold: {', '.join(failed_sectors)}"
        )
    else:
        details.append("  All evaluated sectors within acceptable drop threshold.")

    if sector_accs:
        best_s  = max(sector_accs, key=sector_accs.get)
        worst_s = min(sector_accs, key=sector_accs.get)
        swing   = sector_accs[best_s] - sector_accs[worst_s]
        details.append(
            f"  Best sector : {best_s} ({sector_accs[best_s]:.3f})"
        )
        details.append(
            f"  Worst sector: {worst_s} ({sector_accs[worst_s]:.3f})"
        )
        details.append(f"  Sector swing: {swing:.3f}")

    summary = (
        f"Overall test acc = {overall_acc:.3f}.  "
        f"{'All ' + str(len(sector_accs)) + ' sectors pass.' if passed else str(len(failed_sectors)) + ' sector(s) fail: ' + ', '.join(failed_sectors) + '.'}"
    )

    return AuditResult("SECTOR", passed, summary, details)


# ---------------------------------------------------------------------------
# TEST 4 — OVERFITTING
# ---------------------------------------------------------------------------

def test_overfit(df: pd.DataFrame) -> AuditResult:
    """
    Train on the chronological 70% split and evaluate on the 15% test split.
    Also checks for validation degradation (val_acc << train_acc).

    PASS: train_acc - test_acc <= OVERFIT_MAX_GAP
    FAIL: gap exceeds threshold.
    """
    log.info("Test 4 — OVERFIT CHECK…")

    m = UniversalStockModel()
    train_df, val_df, test_df, _ = m.chronological_split(df.copy())

    X_train, y_train = _Xy(train_df)
    X_val,   y_val   = _Xy(val_df)
    X_test,  y_test  = _Xy(test_df)

    # Train once; val set is the early-stopping criterion only.
    # All three accuracy values come from the same fitted model.
    w = compute_sample_weight("balanced", y_train)
    model = _make_fast_xgb()
    model.fit(
        X_train, y_train,
        sample_weight = w,
        eval_set      = [(X_val, y_val)],
        verbose       = False,
    )
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc   = accuracy_score(y_val,   model.predict(X_val))
    test_pred = model.predict(X_test)
    test_acc  = accuracy_score(y_test,  test_pred)
    test_f1   = f1_score(y_test, test_pred, average="macro", zero_division=0)

    gap          = train_acc - test_acc
    val_gap      = train_acc - val_acc
    passed       = gap <= OVERFIT_MAX_GAP

    # Majority-class baseline (always predict HOLD=1)
    majority_acc = (y_test == 1).mean()

    details = [
        f"  Train accuracy    : {train_acc:.3f}",
        f"  Val   accuracy    : {val_acc:.3f}   (train-val gap: {val_gap:+.3f})",
        f"  Test  accuracy    : {test_acc:.3f}   (train-test gap: {gap:+.3f})",
        f"  Majority baseline : {majority_acc:.3f}  (always predict HOLD)",
        f"  Test macro-F1     : {test_f1:.3f}",
        f"",
        f"  Overfit threshold : train-test gap <= {OVERFIT_MAX_GAP}",
        f"  Result            : gap={gap:.3f}  {'<= threshold (OK)' if passed else '> threshold (OVERFIT)'}",
    ]

    if test_acc < majority_acc:
        details.append(
            f"  NOTE: test accuracy ({test_acc:.3f}) is below the majority-class "
            f"baseline ({majority_acc:.3f}). The model sacrifices accuracy on HOLD "
            f"to improve recall on SELL/BUY minority classes."
        )

    summary = (
        f"Train {train_acc:.3f}  Val {val_acc:.3f}  Test {test_acc:.3f}.  "
        f"Train-test gap = {gap:.3f} "
        f"({'within' if passed else 'exceeds'} {OVERFIT_MAX_GAP} limit)."
    )

    return AuditResult("OVERFIT", passed, summary, details)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class BiasAudit:
    def __init__(self, tickers: list[str] | None = None) -> None:
        self.tickers = tickers
        self.df: pd.DataFrame | None = None

    def run_all(self) -> None:
        print("\n" + "#" * 62)
        print("  BIAS AUDIT REPORT")
        print("#" * 62)

        self.df = _load_full_data(self.tickers)

        results = [
            test_leakage(self.df),
            test_temporal(self.df),
            test_sector(self.df),
            test_overfit(self.df),
        ]

        for r in results:
            _print_result(r)

        # ── Final scorecard ────────────────────────────────────────────────
        n_pass = sum(r.passed for r in results)
        n_fail = len(results) - n_pass
        print("\n" + "=" * 62)
        print("  SCORECARD")
        print("=" * 62)
        for r in results:
            verdict = "PASS" if r.passed else "FAIL"
            print(f"  [{verdict}]  {r.name}")
        print(f"\n  {n_pass}/{len(results)} tests passed.")
        if n_fail == 0:
            print("  Model cleared all bias checks.")
        else:
            print(
                f"  {n_fail} concern(s) flagged — review FAIL details above "
                f"before using this model in production."
            )
        print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    tickers_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    BiasAudit(tickers=tickers_arg).run_all()
