"""
model_loader.py

Single entry point for loading the production model chosen by compare_models.py.
All downstream scripts (predict.py, advisor.py, FastAPI, paper_trading.py)
should call load_production_model() rather than loading either model class directly.

Usage
-----
    from model_loader import load_production_model
    model, cfg = load_production_model()
    # cfg['chosen'] == 'ranker' | 'classifier'
    # cfg['chosen_path'] == relative path used
"""

import json
from pathlib import Path


_BASE = Path(__file__).parent
_CFG_PATH = _BASE / "models" / "production_model.json"


def load_production_model():
    """
    Load the production model and its decision config.

    Returns
    -------
    (model, cfg) where:
        model : UniversalRanker or UniversalStockModel instance
        cfg   : dict parsed from models/production_model.json

    Raises
    ------
    FileNotFoundError  if production_model.json or the model file is missing.
    ValueError         if cfg['chosen'] is not 'ranker' or 'classifier'.
    """
    if not _CFG_PATH.exists():
        raise FileNotFoundError(
            f"Production model config not found at {_CFG_PATH}. "
            "Run `python compare_models.py` first to generate it."
        )

    with open(_CFG_PATH) as fh:
        cfg = json.load(fh)

    chosen      = cfg.get("chosen")
    chosen_path = _BASE / cfg.get("chosen_path", "")

    if not chosen_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {chosen_path}. "
            "Retrain or verify the path in models/production_model.json."
        )

    if chosen == "ranker":
        from ranker_trainer import UniversalRanker
        model = UniversalRanker.load(chosen_path)
    elif chosen == "classifier":
        from trainer import UniversalStockModel
        model = UniversalStockModel.load(chosen_path)
    else:
        raise ValueError(
            f"cfg['chosen'] must be 'ranker' or 'classifier', got: {chosen!r}"
        )

    return model, cfg
