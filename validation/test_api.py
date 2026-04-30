"""
validation/test_api.py

Smoke-tests for the FastAPI server using FastAPI's TestClient.

Usage
-----
    python validation/test_api.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.disable(logging.CRITICAL)

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app, raise_server_exceptions=False)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    body = r.json()
    assert body.get("status") == "ok", f"Expected status='ok', got {body}"


def test_signals_latest():
    r = client.get("/signals/latest")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    body = r.json()
    assert isinstance(body, list), f"Expected list, got {type(body).__name__}"


def test_recommend():
    r = client.post(
        "/recommend",
        json={"positions": {"AAPL": 10}, "cash": 5000},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert "trades" in body, f"Response missing 'trades' key: {list(body.keys())}"


def test_portfolio_history():
    r = client.get("/portfolio/history")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    body = r.json()
    assert isinstance(body, list), f"Expected list, got {type(body).__name__}"


TESTS = [
    ("GET /health → 200 + status='ok'", test_health),
    ("GET /signals/latest → 200 + list", test_signals_latest),
    ("POST /recommend {AAPL:10, cash:5000} → 200 + 'trades' key", test_recommend),
    ("GET /portfolio/history → 200 + list", test_portfolio_history),
]


if __name__ == "__main__":
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR  {name}: {type(exc).__name__}: {exc}")
            failed += 1

    print()
    if failed == 0:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL ({failed} test(s) failed)")
        sys.exit(1)
