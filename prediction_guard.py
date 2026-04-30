"""
prediction_guard.py

Runtime safety checks before serving predictions.

verify_saved_model()
    Loads the saved model file and confirms that feature_bounds was stored.
    If feature_bounds is absent the model was saved by an older version of
    trainer.py that did not compute bounds — retraining is required to
    produce a model that can validate inference-time inputs.

    Returns True  → model is valid, safe to serve predictions.
    Returns False → model is missing feature_bounds; caller should retrain.

validate_feature_ranges()
    Clips inference features to training-time bounds (plus 10% buffer) and
    records any values that were out of range.  Clipping is correct because
    XGBoost extrapolates badly outside its training distribution; logging the
    violations makes data corruption visible rather than silent.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


def verify_saved_model(path: str | Path | None = None) -> bool:
    """
    Check whether the saved model contains feature_bounds.

    Parameters
    ----------
    path : optional path override; defaults to trainer.MODEL_PATH

    Returns
    -------
    True if the model exists and contains feature_bounds, False otherwise.
    """
    from trainer import MODEL_PATH

    load_path = Path(path) if path else MODEL_PATH

    if not load_path.exists():
        log.warning(
            "verify_saved_model: no model file found at %s — "
            "model has not been trained yet.",
            load_path,
        )
        return False

    try:
        payload = joblib.load(load_path)
    except Exception as exc:
        log.warning("verify_saved_model: failed to load %s — %s", load_path, exc)
        return False

    if "feature_bounds" not in payload or payload["feature_bounds"] is None:
        log.warning(
            "verify_saved_model: 'feature_bounds' missing from %s. "
            "Model was saved without bounds — retraining required.",
            load_path,
        )
        return False

    feature_cols = payload.get("feature_cols")
    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        log.warning(
            "verify_saved_model: 'feature_cols' missing or empty in %s — retraining required.",
            load_path,
        )
        return False

    if payload.get("model") is None:
        log.warning(
            "verify_saved_model: 'model' key is None in %s — retraining required.",
            load_path,
        )
        return False

    log.info(
        "verify_saved_model: model at %s is valid "
        "(feature_bounds present for %d features, model loaded).",
        load_path,
        len(payload["feature_bounds"]),
    )
    return True


def validate_feature_ranges(
    df: "pd.DataFrame",
    feature_bounds: dict,
) -> "tuple[pd.DataFrame, list[dict]]":
    """
    Clip inference features to training-time bounds (plus 10% extrapolation
    buffer) and record every value that required clipping.

    Parameters
    ----------
    df            : DataFrame with one row per (ticker, date)
    feature_bounds: {col: (min_val, max_val)} as stored by trainer/ranker

    Returns
    -------
    (clipped_df, violations)
        clipped_df : copy of df with out-of-range values clipped to bounds
        violations : list of dicts
                     {ticker, date, feature, raw_value, clipped_to}
    """
    import pandas as pd  # noqa: PLC0415 — lazy to keep module-level import light

    df = df.copy()
    violations: list[dict] = []

    for col, bounds in feature_bounds.items():
        if col not in df.columns:
            log.warning("validate_feature_ranges: column %r not in DataFrame — skipping", col)
            continue

        lo, hi = float(bounds[0]), float(bounds[1])
        rng    = hi - lo
        lower  = lo - 0.10 * rng
        upper  = hi + 0.10 * rng

        below_mask = df[col] < lower
        above_mask = df[col] > upper

        # Record below-bound violations before clipping
        if below_mask.any():
            sub = df.loc[below_mask, ["ticker", "date", col]] if "ticker" in df.columns else df.loc[below_mask, [col]]
            for _, row in sub.iterrows():
                violations.append({
                    "ticker":     str(row.get("ticker", "")),
                    "date":       str(row.get("date",   "")),
                    "feature":    col,
                    "raw_value":  float(row[col]),
                    "clipped_to": float(lower),
                })

        # Record above-bound violations before clipping
        if above_mask.any():
            sub = df.loc[above_mask, ["ticker", "date", col]] if "ticker" in df.columns else df.loc[above_mask, [col]]
            for _, row in sub.iterrows():
                violations.append({
                    "ticker":     str(row.get("ticker", "")),
                    "date":       str(row.get("date",   "")),
                    "feature":    col,
                    "raw_value":  float(row[col]),
                    "clipped_to": float(upper),
                })

        df[col] = df[col].clip(lower=lower, upper=upper)

    return df, violations
