import os
import boto3
from datetime import datetime, timezone
from pathlib import Path
from botocore.client import Config
from botocore.exceptions import ClientError

DB_KEY = "trading.db"
LASTMOD_KEY = "trading.db.lastmod"


def _get_client():
    endpoint = os.environ.get("ENDPOINTS_FOR_S3_CLIENTS")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("SECRET_ACCESS_KEY")

    assert endpoint, "ENDPOINTS_FOR_S3_CLIENTS not set in environment"
    assert access_key, "R2_ACCESS_KEY_ID not set in environment"
    assert secret_key, "SECRET_ACCESS_KEY not set in environment"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _bucket():
    bucket = os.environ.get("R2_BUCKET_NAME")
    assert bucket, "R2_BUCKET_NAME not set in environment"
    return bucket


def upload_db(local_path: str) -> str:
    """Upload local DB to R2, update lastmod marker, return ISO timestamp."""
    local_path = Path(local_path)
    assert local_path.exists(), f"Local DB not found: {local_path}"
    assert local_path.stat().st_size > 0, f"Local DB is empty: {local_path}"

    client = _get_client()
    bucket = _bucket()

    print(f"[r2] uploading {local_path} ({local_path.stat().st_size / 1e6:.1f} MB) to {bucket}/{DB_KEY}")
    client.upload_file(str(local_path), bucket, DB_KEY)

    now_iso = datetime.now(timezone.utc).isoformat()
    client.put_object(Bucket=bucket, Key=LASTMOD_KEY, Body=now_iso.encode("utf-8"))
    print(f"[r2] uploaded, lastmod marker set to {now_iso}")
    return now_iso


def download_db(local_path: str) -> None:
    """Download DB from R2 to local_path."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    bucket = _bucket()

    print(f"[r2] downloading {bucket}/{DB_KEY} to {local_path}")
    client.download_file(bucket, DB_KEY, str(local_path))
    assert local_path.exists(), f"Download failed, file missing: {local_path}"
    assert local_path.stat().st_size > 0, f"Download failed, empty file: {local_path}"
    print(f"[r2] downloaded {local_path.stat().st_size / 1e6:.1f} MB")


def get_remote_lastmod() -> str | None:
    """Return ISO timestamp of remote DB, or None if marker missing."""
    client = _get_client()
    bucket = _bucket()
    try:
        resp = client.get_object(Bucket=bucket, Key=LASTMOD_KEY)
        return resp["Body"].read().decode("utf-8").strip()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def get_local_lastmod(local_path: str) -> str | None:
    """Return ISO timestamp of local DB based on file mtime, or None if missing."""
    p = Path(local_path)
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()


def is_remote_newer(local_path: str) -> bool:
    """True if R2's copy is newer than local (or local doesn't exist)."""
    if not Path(local_path).exists():
        return True
    remote = get_remote_lastmod()
    if remote is None:
        return False
    local = get_local_lastmod(local_path)
    return remote > local  # ISO timestamps sort lexicographically
