"""
prediction_guard.py

Runtime safety checks for the UniversalStockModel before serving predictions.

verify_saved_model()
    Loads the saved model file and confirms that feature_bounds was stored.
    If feature_bounds is absent the model was saved by an older version of
    trainer.py that did not compute bounds — retraining is required to
    produce a model that can validate inference-time inputs.

    Returns True  → model is valid, safe to serve predictions.
    Returns False → model is missing feature_bounds; caller should retrain.
"""

import logging
from pathlib import Path

import joblib

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
