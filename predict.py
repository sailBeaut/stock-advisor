"""
predict.py

Inference script — fetches today's data, computes features, and outputs
ranked BUY signals from the model designated in models/production_config.json.

The active model is controlled by production_config.json ('ranker' or
'classifier').  Change the JSON to switch candidates; no code edits needed.

Usage
-----
    python predict.py                 # fetch latest data, then predict
    python predict.py --skip-update   # predict from existing DB data only
    python predict.py --no-save       # predict but do not write to signals table
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import database
import prediction_guard
from production_config import load_production_model
from trainer import FEATURE_COLS, LABEL_BUY, LABEL_HOLD, LABEL_SELL

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LABEL_NAMES = {LABEL_SELL: "SELL", LABEL_HOLD: "HOLD", LABEL_BUY: "BUY"}

_CFG_PATH = Path(__file__).parent / "models" / "production_config.json"

# ---------------------------------------------------------------------------
# Data update helpers
# ---------------------------------------------------------------------------

def _update_data() -> None:
    """Run the full incremental data pipeline to bring the DB up to date."""
    import data_collector
    import edgar_collector
    import feature_engine
    import macro

    log.info("--- Step 1/5: incremental price update ---")
    data_collector.run_incremental_update()

    log.info("--- Step 2/5: macro features ---")
    macro.fetch_macro()

    log.info("--- Step 3/5: technical features ---")
    feature_engine.compute_incremental()

    log.info("--- Step 4/5: EDGAR features ---")
    edgar_collector.compute_edgar_features()

    try:
        import earnings_collector
        log.info("--- Step 5a/5: earnings features ---")
        earnings_collector.compute_earnings_features()
    except ImportError:
        pass

    try:
        import sentiment_collector
        log.info("--- Step 5b/5: sentiment (7 days back) ---")
        sentiment_collector.collect_sentiment(days_back=7)
    except ImportError:
        pass

    log.info("--- Data update complete ---")


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _load_latest_features() -> pd.DataFrame:
    """
    Load the most recent feature row per ticker from the DB.
    Joins stocks table to include sector.
    """
    feat_select = ", ".join(f"f.{c}" for c in FEATURE_COLS)
    sql = f"""
        SELECT f.ticker, f.date,
               {feat_select},
               COALESCE(s.sector, '') AS sector
        FROM   features f
        LEFT JOIN stocks s ON s.ticker = f.ticker
        WHERE  f.date = (SELECT MAX(date) FROM features)
        ORDER  BY f.ticker
    """
    with database.connection() as conn:
        rows = conn.execute(sql).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    log.info(
        "Loaded %d feature rows for date %s.",
        len(df), df["date"].iloc[0],
    )
    return df


# ---------------------------------------------------------------------------
# Feature violations persistence
# ---------------------------------------------------------------------------

_VIOLATIONS_INSERT = """
INSERT INTO feature_violations (ticker, date, feature, raw_value, clipped_to, run_at)
VALUES (:ticker, :date, :feature, :raw_value, :clipped_to, :run_at)
"""


def _persist_violations(violations: list[dict]) -> None:
    run_at  = datetime.datetime.utcnow().isoformat(timespec="seconds")
    records = [{**v, "run_at": run_at} for v in violations]
    with database.connection() as conn:
        conn.executemany(_VIOLATIONS_INSERT, records)
    log.info("Persisted %d feature violations to DB.", len(records))


# ---------------------------------------------------------------------------
# Signal persistence
# ---------------------------------------------------------------------------

_SIGNAL_UPSERT = """
INSERT INTO signals (ticker, date, signal, confidence, probabilities)
VALUES (:ticker, :date, :signal, :confidence, :probabilities)
ON CONFLICT(ticker, date) DO UPDATE SET
    signal        = excluded.signal,
    confidence    = excluded.confidence,
    probabilities = excluded.probabilities
"""


def _save_signals(df: pd.DataFrame, proba: np.ndarray, preds: np.ndarray) -> int:
    """Upsert all predictions into the signals table. Returns row count."""
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        label_id  = int(preds[i])
        signal    = LABEL_NAMES[label_id]
        conf      = float(proba[i, label_id])
        prob_json = json.dumps({
            "SELL": round(float(proba[i, LABEL_SELL]), 4),
            "HOLD": round(float(proba[i, LABEL_HOLD]), 4),
            "BUY":  round(float(proba[i, LABEL_BUY]),  4),
        })
        records.append({
            "ticker":        row["ticker"],
            "date":          row["date"],
            "signal":        signal,
            "confidence":    conf,
            "probabilities": prob_json,
        })

    with database.connection() as conn:
        conn.executemany(_SIGNAL_UPSERT, records)

    log.info("Saved %d signal rows to DB.", len(records))
    return len(records)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_signal_table(
    label: str,
    df: pd.DataFrame,
    proba: np.ndarray,
    mask: np.ndarray,
    sort_col: int,
    ascending: bool = False,
) -> None:
    """Print a ranked signal table for one signal type (BUY or SELL)."""
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    if not mask.any():
        print(f"  No {label.split()[0]} signals for this date.")
        return

    header = f"{'Rank':>4}  {'Ticker':<7}  {'Sector':<25}  {'BUY':>7}  {'SELL':>7}  {'HOLD':>7}"
    print(header)
    print("-" * 72)

    indices   = np.where(mask)[0]
    sort_probs = proba[indices, sort_col]
    order      = np.argsort(sort_probs)
    if not ascending:
        order = order[::-1]
    sorted_idx = indices[order]

    for rank, idx in enumerate(sorted_idx, 1):
        ticker       = df.iloc[idx]["ticker"]
        sector_short = (df.iloc[idx]["sector"] or "")[:25]
        b = float(proba[idx, LABEL_BUY])
        s = float(proba[idx, LABEL_SELL])
        h = float(proba[idx, LABEL_HOLD])
        print(
            f"{rank:>4}  {ticker:<7}  {sector_short:<25}  "
            f"{b:>7.4f}  {s:>7.4f}  {h:>7.4f}"
        )


def _print_signals(df: pd.DataFrame, proba: np.ndarray, preds: np.ndarray) -> None:
    """Print ranked BUY and SELL signal tables plus summary stats."""
    date_str  = df["date"].iloc[0] if len(df) else "N/A"
    n_total   = len(df)

    buy_mask  = preds == LABEL_BUY
    sell_mask = preds == LABEL_SELL
    hold_mask = preds == LABEL_HOLD

    n_buy  = int(buy_mask.sum())
    n_sell = int(sell_mask.sum())
    n_hold = int(hold_mask.sum())

    # ---- BUY signals: sorted by P(BUY) descending -------------------------
    _print_signal_table(
        f"BUY SIGNALS — {date_str}",
        df, proba, buy_mask,
        sort_col=LABEL_BUY, ascending=False,
    )

    # ---- SELL signals: sorted by P(BUY) ascending (least bullish first) ---
    _print_signal_table(
        f"SELL / AVOID SIGNALS — {date_str}",
        df, proba, sell_mask,
        sort_col=LABEL_BUY, ascending=True,
    )

    # ---- summary stats -----------------------------------------------------
    print(f"\n{'─'*72}")
    print(f"  Total tickers analyzed : {n_total}")
    print(f"  BUY  signals           : {n_buy:4d}  ({n_buy/n_total*100:5.1f}%)")
    print(f"  SELL signals           : {n_sell:4d}  ({n_sell/n_total*100:5.1f}%)")
    print(f"  HOLD signals           : {n_hold:4d}  ({n_hold/n_total*100:5.1f}%)")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(skip_update: bool = False, save_signals: bool = True) -> None:
    """
    Full prediction pipeline.

    Parameters
    ----------
    skip_update  : if True, skip the data-fetch step and predict from DB
    save_signals : if True, upsert predictions into the signals table
    """
    # ---- 0. Determine candidate model and its on-disk path ----------------
    _candidate = "classifier"
    if _CFG_PATH.exists():
        _candidate = json.loads(_CFG_PATH.read_text()).get("candidate_model", "classifier")

    if _candidate == "ranker":
        from ranker_trainer import RANKER_PATH as _model_path
    else:
        from trainer import MODEL_PATH as _model_path

    # ---- 1. Verify model integrity ----------------------------------------
    if not prediction_guard.verify_saved_model(_model_path):
        log.error(
            "Model failed integrity check — feature_bounds missing. "
            "Run: python trainer.py    to retrain and resave."
        )
        sys.exit(1)

    # ---- 2. Load model ----------------------------------------------------
    model, model_type = load_production_model()
    log.info("Candidate model: %s", model_type)

    # ---- 3. Fetch latest data ---------------------------------------------
    if not skip_update:
        _update_data()
    else:
        log.info("--skip-update: using existing data in DB.")

    # ---- 4. Load today's feature rows -------------------------------------
    df = _load_latest_features()
    if df.empty:
        log.error("No feature rows found in DB — run feature_engine.py first.")
        sys.exit(1)

    # ---- 5. Validate feature bounds (clip + record violations) ------------
    if model.feature_bounds:
        df, violations = prediction_guard.validate_feature_ranges(df, model.feature_bounds)
        if violations:
            log.warning("FEATURE BOUNDS: %d violations clipped", len(violations))
            for v in violations[:10]:
                log.warning(
                    "  %s  %s  %s: raw=%.4f  clipped_to=%.4f",
                    v["ticker"], v["date"], v["feature"],
                    v["raw_value"], v["clipped_to"],
                )
            _persist_violations(violations)
            n_cells = len(df) * len(model.feature_bounds)
            if len(violations) > 0.05 * n_cells:
                log.error(
                    "FEATURE BOUNDS: %.1f%% of cells violated (>5%%) — "
                    "data may be corrupted. Aborting.",
                    100.0 * len(violations) / n_cells,
                )
                sys.exit(1)

    # ---- 6. Predict -------------------------------------------------------
    if model_type == "ranker":
        # Ranker returns relevance scores; top-20 by score → BUY, rest → HOLD
        scores     = model.predict(df)                      # (n,) float
        top_n      = 20
        order      = np.argsort(scores)[::-1]
        preds      = np.full(len(df), LABEL_HOLD, dtype=int)
        preds[order[:top_n]] = LABEL_BUY
        score_range = scores.max() - scores.min()
        scores_norm = (scores - scores.min()) / (score_range + 1e-9)
        # Pseudo-probability matrix so display/save functions work unchanged:
        # SELL=0, HOLD=1−score_norm, BUY=score_norm
        proba = np.stack([
            np.zeros(len(df)),
            1.0 - scores_norm,
            scores_norm,
        ], axis=1)
    else:
        from trainer import UniversalStockModel
        proba = model.predict_proba(df)
        preds = UniversalStockModel._apply_buy_percentile(proba, model.buy_top_fraction)

    # ---- 7. Display results -----------------------------------------------
    _print_signals(df, proba, preds)

    # ---- 8. Persist -------------------------------------------------------
    if save_signals:
        _save_signals(df, proba, preds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Generate BUY/HOLD/SELL signals from the trained model."
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="Skip data fetch; predict from existing DB features.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write predictions to the signals table.",
    )
    args = parser.parse_args()

    run(skip_update=args.skip_update, save_signals=not args.no_save)
