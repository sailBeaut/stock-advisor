import os
from pathlib import Path

CLOUD_DB_PATH = "/tmp/trading.db"
LOCAL_DB_PATH = "trading.db"


def is_cloud_environment() -> bool:
    """True if we should treat R2 as the source of truth."""
    return bool(os.environ.get("RENDER")) or bool(os.environ.get("USE_R2_DB"))


def get_db_path() -> str:
    """Return the path the app should open for trading.db."""
    if is_cloud_environment():
        return CLOUD_DB_PATH
    return LOCAL_DB_PATH


def ensure_db_present() -> str:
    """
    If running in cloud: download from R2 if missing or stale.
    Returns the path the app should use.
    """
    path = get_db_path()
    if not is_cloud_environment():
        return path

    from r2_client import download_db, is_remote_newer

    if not Path(path).exists():
        print(f"[db_path] no local DB at {path}, downloading from R2")
        download_db(path)
    elif is_remote_newer(path):
        print(f"[db_path] R2 has newer DB, re-downloading to {path}")
        download_db(path)
    else:
        print(f"[db_path] local DB at {path} is up to date")
    return path
