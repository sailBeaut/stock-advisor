"""
encoders.py

Persisted sklearn encoders for categorical features.

SECTOR_ENCODER_PATH — path to the fitted LabelEncoder joblib file.

Public API
----------
fit_sector_encoder()                    → LabelEncoder
    Fit on all DISTINCT sectors in the stocks table, add 'Unknown',
    save to disk, and return.  Logs a WARNING when sector classes change
    vs a previous fit (encoder is refitted and saved).

get_sector_encoder()                    → LabelEncoder
    Load from SECTOR_ENCODER_PATH if it exists; otherwise call
    fit_sector_encoder().

encode_sector(sector_name, encoder)     → int
    Encode one sector name.  Unknown names fall back to 'Unknown'.
"""

import logging
from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder

import database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SECTOR_ENCODER_PATH = Path(__file__).parent / "models" / "sector_encoder.joblib"


def fit_sector_encoder() -> LabelEncoder:
    """
    Read all DISTINCT non-NULL sectors from the stocks table, union with
    {'Unknown'}, fit a LabelEncoder, persist to SECTOR_ENCODER_PATH,
    and return the encoder.

    If a previous encoder file exists and the class set has changed, a
    WARNING is logged before the file is overwritten — the encoder is
    intentionally frozen after the initial fit and should only be refitted
    when the universe of sectors genuinely changes.
    """
    with database.connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT sector FROM stocks WHERE sector IS NOT NULL"
        ).fetchall()

    classes = sorted({r[0] for r in rows} | {"Unknown"})

    encoder = LabelEncoder()
    encoder.fit(classes)

    SECTOR_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Warn when sector classes differ from a previous fit
    if SECTOR_ENCODER_PATH.exists():
        try:
            old_enc = joblib.load(SECTOR_ENCODER_PATH)
            old_set = set(old_enc.classes_)
            new_set = set(encoder.classes_)
            if old_set != new_set:
                added   = sorted(new_set - old_set)
                removed = sorted(old_set - new_set)
                log.warning(
                    "Sector encoder classes changed from previous fit — "
                    "refitting and saving.  Added: %s  Removed: %s",
                    added, removed,
                )
        except Exception:
            pass  # corrupt or incompatible old file — silently overwrite

    joblib.dump(encoder, SECTOR_ENCODER_PATH)
    log.info(
        "Sector encoder fitted (%d classes) and saved → %s",
        len(encoder.classes_), SECTOR_ENCODER_PATH,
    )
    return encoder


def get_sector_encoder() -> LabelEncoder:
    """
    Return a fitted LabelEncoder for sector names.

    Loads from SECTOR_ENCODER_PATH when the file exists; otherwise calls
    fit_sector_encoder() to create it.  The encoder is intentionally
    frozen after the initial fit — call fit_sector_encoder() explicitly
    to refit when the sector universe changes.
    """
    if SECTOR_ENCODER_PATH.exists():
        return joblib.load(SECTOR_ENCODER_PATH)
    return fit_sector_encoder()


def encode_sector(sector_name: str, encoder: LabelEncoder | None = None) -> int:
    """
    Encode *sector_name* to a stable integer using the persisted LabelEncoder.

    Unknown names (not in encoder.classes_) fall back silently to the
    'Unknown' class so XGBoost can route them via its native missing-value
    handling rather than raising an error.

    Parameters
    ----------
    sector_name : str
        GICS sector name, e.g. 'Information Technology'.
    encoder : LabelEncoder, optional
        Pre-loaded encoder.  When None, get_sector_encoder() is called.
        Pass an already-loaded encoder in hot paths to avoid repeated
        disk I/O.

    Returns
    -------
    int
    """
    if encoder is None:
        encoder = get_sector_encoder()
    if sector_name not in encoder.classes_:
        sector_name = "Unknown"
    return int(encoder.transform([sector_name])[0])
