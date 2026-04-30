"""
production_config.py

Config-driven model loader for paper-trading.

Reads models/production_config.json to determine which model is the current
candidate under paper-trading evaluation.  This is separate from
model_loader.py / production_model.json, which records the model that has
*earned* production status from forward evidence.

The candidate is a judgment call made explicitly at selection time.
The production model is determined by data after paper-trading matures.

Usage
-----
    from production_config import load_production_model
    model, model_type = load_production_model()
    # model_type is 'ranker' or 'classifier'
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "models" / "production_config.json"


def load_production_model():
    """Load whichever model is currently designated for paper-trading."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Candidate model config not found at {CONFIG_PATH}. "
            "Create models/production_config.json with a 'candidate_model' key."
        )
    cfg = json.loads(CONFIG_PATH.read_text())
    candidate = cfg.get("candidate_model")
    if candidate == "ranker":
        from ranker_trainer import UniversalRanker
        return UniversalRanker.load(), "ranker"
    elif candidate == "classifier":
        from trainer import UniversalStockModel
        return UniversalStockModel.load(), "classifier"
    raise ValueError(f"Unknown candidate_model: {candidate!r}")
