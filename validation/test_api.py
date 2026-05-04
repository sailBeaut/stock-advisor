"""
validation/test_api.py

Smoke-tests for the FastAPI server using FastAPI's TestClient.

Usage
-----
    python validation/test_api.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.disable(logging.CRITICAL)

import database
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


def test_stock_detail():
    # Resolve a valid ticker from existing signals; fall back to 404-only check
    signals_r = client.get("/signals/latest")
    has_data = signals_r.status_code == 200 and bool(signals_r.json())

    if has_data:
        ticker = signals_r.json()[0]["ticker"]
        r = client.get(f"/stock/{ticker}")
        assert r.status_code == 200, f"Expected 200 for {ticker}, got {r.status_code}: {r.text}"
        body = r.json()
        required = ("ticker", "meta", "price_history", "signal", "features_latest", "top_contributions")
        for field in required:
            assert field in body, f"Response missing '{field}': {list(body.keys())}"
        assert body["ticker"] == ticker
        assert isinstance(body["price_history"], list)
        assert isinstance(body["top_contributions"], list)

    # 404 for a nonsense ticker is always exercised
    r404 = client.get("/stock/ZZZNOTREAL999")
    assert r404.status_code == 404, f"Expected 404 for unknown ticker, got {r404.status_code}"


def test_portfolio_overview():
    r = client.get("/portfolio/overview")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    required = ("portfolio_value", "vs_spy_pct", "sharpe", "max_drawdown", "top_picks", "holdings")
    for field in required:
        assert field in body, f"Response missing '{field}': {list(body.keys())}"
    assert isinstance(body["top_picks"], list)
    assert isinstance(body["holdings"], list)


def _cleanup_holdings():
    with database.connection() as conn:
        conn.execute("DELETE FROM user_holdings")
        conn.execute("DELETE FROM user_cash")


def test_get_holdings_empty():
    _cleanup_holdings()
    r = client.get("/holdings")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    for field in ("positions", "cash", "avg_costs", "total_value_usd"):
        assert field in body, f"Missing '{field}': {list(body.keys())}"
    assert isinstance(body["positions"], dict)
    assert body["cash"] == 0.0
    assert isinstance(body["total_value_usd"], (int, float))


def test_holdings_auth_fail():
    """PUT and POST must return 401 when APP_API_KEY is set and header is wrong."""
    os.environ["APP_API_KEY"] = "secret-test-key"
    try:
        # No header at all
        r = client.put("/holdings", json={"positions": {"AAPL": 1}, "cash": 100.0})
        assert r.status_code == 401, f"Expected 401 (no header), got {r.status_code}"

        # Wrong key
        r = client.put(
            "/holdings",
            json={"positions": {"AAPL": 1}, "cash": 100.0},
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401, f"Expected 401 (wrong key), got {r.status_code}"

        r = client.post(
            "/holdings/AAPL",
            json={"shares": 5},
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401, f"Expected 401 on POST (wrong key), got {r.status_code}"
    finally:
        del os.environ["APP_API_KEY"]


def test_holdings_crud():
    """PUT replaces holdings; POST upserts a single position; GET reflects both."""
    os.environ["APP_API_KEY"] = "test-key-123"
    headers = {"X-API-Key": "test-key-123"}
    try:
        _cleanup_holdings()

        # PUT — set initial snapshot
        r = client.put(
            "/holdings",
            json={
                "positions": {"AAPL": 10.0, "MSFT": 5.0},
                "cash": 2500.0,
                "avg_costs": {"AAPL": 180.0, "MSFT": 320.0},
            },
            headers=headers,
        )
        assert r.status_code == 200, f"PUT expected 200, got {r.status_code}: {r.text}"
        assert r.json().get("ok") is True

        # GET — verify positions and cash are saved
        r = client.get("/holdings")
        assert r.status_code == 200
        body = r.json()
        assert "AAPL" in body["positions"], "AAPL missing from positions"
        assert body["positions"]["AAPL"] == 10.0
        assert body["cash"] == 2500.0
        assert body["avg_costs"].get("AAPL") == 180.0
        assert isinstance(body["total_value_usd"], (int, float))

        # POST — add a new ticker
        r = client.post(
            "/holdings/GOOGL",
            json={"shares": 3.0, "avg_cost": 150.0},
            headers=headers,
        )
        assert r.status_code == 200, f"POST expected 200, got {r.status_code}: {r.text}"
        assert r.json().get("ticker") == "GOOGL"

        # GET — verify GOOGL was added
        r = client.get("/holdings")
        body = r.json()
        assert "GOOGL" in body["positions"], "GOOGL missing after POST"
        assert body["positions"]["GOOGL"] == 3.0

        # POST — update existing ticker
        r = client.post(
            "/holdings/AAPL",
            json={"shares": 20.0, "avg_cost": 175.0},
            headers=headers,
        )
        assert r.status_code == 200
        r = client.get("/holdings")
        assert r.json()["positions"]["AAPL"] == 20.0

        # PUT — full replace wipes previous holdings
        r = client.put(
            "/holdings",
            json={"positions": {"TSLA": 2.0}, "cash": 500.0},
            headers=headers,
        )
        assert r.status_code == 200
        r = client.get("/holdings")
        body = r.json()
        assert "TSLA" in body["positions"]
        assert "AAPL" not in body["positions"], "AAPL should be gone after full replace"
        assert body["cash"] == 500.0

    finally:
        del os.environ["APP_API_KEY"]
        _cleanup_holdings()


TESTS = [
    ("GET /health -> 200 + status='ok'", test_health),
    ("GET /signals/latest -> 200 + list", test_signals_latest),
    ("POST /recommend {AAPL:10, cash:5000} -> 200 + 'trades' key", test_recommend),
    ("GET /portfolio/history -> 200 + list", test_portfolio_history),
    ("GET /stock/{ticker} -> 200 + all fields / 404 for unknown", test_stock_detail),
    ("GET /portfolio/overview -> 200 + dashboard fields", test_portfolio_overview),
    ("GET /holdings -> 200 + shape when empty", test_get_holdings_empty),
    ("PUT+POST /holdings auth=wrong -> 401", test_holdings_auth_fail),
    ("PUT+POST+GET /holdings full CRUD with auth", test_holdings_crud),
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
