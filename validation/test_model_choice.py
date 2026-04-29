"""
validation/test_model_choice.py

Validates that compare_models.py produced a valid production_model.json
and that model_loader.load_production_model() works end-to-end.

Exit codes: 0 = PASS, 1 = FAIL
"""

import json
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

REQUIRED_METRIC_KEYS = {"cagr", "sharpe", "max_dd", "alpha_annualized", "alpha_ci_lower_95"}
VALID_VERDICTS       = {"STRONG", "PASS", "MARGINAL", "FAIL"}


def run_tests() -> bool:
    errors: list[str] = []

    # ── 1. JSON exists and is valid ─────────────────────────────────────────
    json_path = Path("models/production_model.json")
    if not json_path.exists():
        print(f"  FAIL: {json_path} does not exist — run `python compare_models.py` first")
        return False

    try:
        with open(json_path) as fh:
            cfg = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  FAIL: invalid JSON in {json_path} — {exc}")
        return False

    # ── 2. chosen ────────────────────────────────────────────────────────────
    if cfg.get("chosen") not in ("ranker", "classifier"):
        errors.append(
            f"cfg['chosen'] must be 'ranker' or 'classifier', got: {cfg.get('chosen')!r}"
        )

    # ── 3. chosen_path points to an existing file ────────────────────────────
    chosen_path_str = cfg.get("chosen_path", "")
    chosen_path     = Path(chosen_path_str)
    if not chosen_path.exists():
        errors.append(f"cfg['chosen_path'] does not exist on disk: {chosen_path}")

    # ── 4. holdout_metrics contains both models with required keys ───────────
    hm = cfg.get("holdout_metrics", {})
    for model_key in ("classifier", "ranker"):
        if model_key not in hm:
            errors.append(f"cfg['holdout_metrics'] is missing key: '{model_key}'")
        else:
            missing = REQUIRED_METRIC_KEYS - set(hm[model_key].keys())
            if missing:
                errors.append(
                    f"cfg['holdout_metrics']['{model_key}'] is missing keys: {sorted(missing)}"
                )

    # ── 5. verdict strings are valid ─────────────────────────────────────────
    for field in ("verdict_classifier", "verdict_ranker"):
        val = cfg.get(field)
        if val not in VALID_VERDICTS:
            errors.append(
                f"cfg['{field}'] = {val!r} is not one of {sorted(VALID_VERDICTS)}"
            )

    # ── 6. load_production_model() returns a live model ─────────────────────
    try:
        from model_loader import load_production_model
        model, loaded_cfg = load_production_model()
        if model is None:
            errors.append("load_production_model() returned None as the model")
        elif loaded_cfg.get("chosen") != cfg.get("chosen"):
            errors.append(
                "load_production_model() returned cfg['chosen'] that does not match "
                "the JSON on disk"
            )
    except Exception as exc:
        errors.append(f"load_production_model() raised an exception: {exc}")

    # ── result ────────────────────────────────────────────────────────────────
    if errors:
        for msg in errors:
            print(f"  FAIL: {msg}")
        print("\nFAIL")
        return False

    print(f"  chosen            : {cfg['chosen']}")
    print(f"  chosen_path       : {cfg['chosen_path']}")
    print(f"  decision_rule     : {cfg.get('decision_rule', 'N/A')}")
    print(f"  decided_at        : {cfg.get('decided_at', 'N/A')}")
    print(f"  verdict_classifier: {cfg['verdict_classifier']}")
    print(f"  verdict_ranker    : {cfg['verdict_ranker']}")
    for key in ("classifier", "ranker"):
        m = hm.get(key, {})
        sharpe = m.get("sharpe", float("nan"))
        alpha  = m.get("alpha_annualized", float("nan"))
        try:
            print(
                f"  {key:<12} sharpe={sharpe:.3f}  alpha={alpha * 100:+.2f}%"
                f"  ci_lo={m.get('alpha_ci_lower_95', float('nan')) * 100:+.2f}%"
            )
        except (TypeError, ValueError):
            print(f"  {key:<12} (metrics contain NaN — model may not have enough holdout data)")
    print("\nPASS")
    return True


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
