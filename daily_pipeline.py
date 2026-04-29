"""
daily_pipeline.py

Runs the complete daily update and prediction sequence in order.
Each step is timed and error-isolated — a failure in one step does not
crash the pipeline; subsequent steps still run.

Usage
-----
    python daily_pipeline.py                 # full pipeline
    python daily_pipeline.py --skip-fetch    # skip data fetch steps 1-3
    python daily_pipeline.py --retrain       # retrain model before predict
    python daily_pipeline.py --skip-fetch --retrain
"""

import argparse
import sys
import time
import traceback
from datetime import datetime


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

class _StepResult:
    def __init__(self, name: str) -> None:
        self.name    = name
        self.passed  = False
        self.elapsed = 0.0
        self.error   = ""


def _run_step(name: str, fn) -> _StepResult:
    result = _StepResult(name)
    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()

    print(f"\n{'─'*68}")
    print(f"  STEP: {name}")
    print(f"  Started : {ts_start}")
    print(f"{'─'*68}")

    try:
        fn()
        result.passed = True
    except Exception:
        result.error = traceback.format_exc()
        print(f"\n  ERROR in '{name}':\n{result.error}")

    result.elapsed = time.perf_counter() - t0
    ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "OK" if result.passed else "FAILED"
    print(f"  Finished: {ts_end}  [{status}]  elapsed: {result.elapsed:.1f}s")

    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(skip_fetch: bool = False, retrain: bool = False) -> None:
    pipeline_start = time.perf_counter()
    results: list[_StepResult] = []

    print("\n" + "=" * 68)
    print(f"  DAILY PIPELINE  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  skip-fetch={skip_fetch}  retrain={retrain}")
    print("=" * 68)

    # ── Step 0: fundamentals safety audit (abort on failure) ────────────────
    import data_collector
    _audit = _run_step(
        "0. Fundamentals safety audit",
        data_collector.audit_fundamentals_safety,
    )
    results.append(_audit)
    if not _audit.passed:
        print("\n  ABORT: fundamentals safety audit FAILED — pipeline halted.")
        print("  Fix the schema overlap before running the pipeline.\n")
        return

    # ── Steps 1–3: data fetch (skippable) ───────────────────────────────────

    if not skip_fetch:
        results.append(_run_step(
            "1. Price update (incremental)",
            data_collector.run_incremental_update,
        ))

        import macro
        results.append(_run_step(
            "2. Macro features (VIX / rates / CPI)",
            macro.fetch_macro,
        ))

        try:
            import sentiment_collector
            results.append(_run_step(
                "3. Sentiment (last 7 days)",
                lambda: sentiment_collector.collect_sentiment(days_back=7),
            ))
        except ImportError:
            print("\n  SKIP: sentiment_collector not found.")
    else:
        print("\n  --skip-fetch: steps 1-3 skipped.")

    # ── Step 4: technical features ───────────────────────────────────────────
    import feature_engine
    results.append(_run_step(
        "4. Feature engine (incremental)",
        feature_engine.compute_incremental,
    ))

    # ── Step 5: EDGAR features ───────────────────────────────────────────────
    import edgar_collector
    results.append(_run_step(
        "5. EDGAR features",
        edgar_collector.compute_edgar_features,
    ))

    # ── Step 6: earnings features (optional) ─────────────────────────────────
    try:
        import earnings_collector
        results.append(_run_step(
            "6. Earnings features",
            earnings_collector.compute_earnings_features,
        ))
    except ImportError:
        print("\n  SKIP: earnings_collector not found.")

    # ── Step 7 (optional): retrain ───────────────────────────────────────────
    if retrain:
        from trainer import UniversalStockModel
        def _retrain():
            m = UniversalStockModel()
            m.fit()
            m.save()

        results.append(_run_step(
            "7. Model retrain",
            _retrain,
        ))

    # ── Step 8: predict ──────────────────────────────────────────────────────
    import predict
    results.append(_run_step(
        "8. Predict (signals output)",
        lambda: predict.run(skip_update=True, save_signals=True),
    ))

    # ── Summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - pipeline_start
    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass

    print("\n" + "=" * 68)
    print("  PIPELINE SUMMARY")
    print("=" * 68)
    for r in results:
        status = "OK    " if r.passed else "FAILED"
        print(f"  [{status}]  {r.name:<42}  {r.elapsed:6.1f}s")
        if not r.passed and r.error:
            # Print first line of traceback for quick diagnosis
            first_line = [l for l in r.error.splitlines() if l.strip()][-1]
            print(f"           -> {first_line[:60]}")

    print(f"\n  Steps passed: {n_pass}/{len(results)}")
    print(f"  Total time  : {total_elapsed:.1f}s")
    if n_fail == 0:
        print("  Status      : ALL STEPS PASSED")
    else:
        print(f"  Status      : {n_fail} STEP(S) FAILED — review errors above")
    print("=" * 68 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run the complete daily update and prediction pipeline."
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetch steps (prices, macro, sentiment). "
             "Useful when prices were already updated.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the model before generating predictions.",
    )
    args = parser.parse_args()

    run_pipeline(skip_fetch=args.skip_fetch, retrain=args.retrain)
