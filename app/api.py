import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import advisor
import database

ENVIRONMENT = os.environ.get('ENVIRONMENT', 'local')

if ENVIRONMENT != 'cloud':
    import paper_trading

_model_cache: dict = {}


def get_model():
    if 'model' not in _model_cache:
        from trainer import UniversalStockModel
        _model_cache['model'] = UniversalStockModel.load()
    return _model_cache['model']


app = FastAPI(title="Stock Advisor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Holdings(BaseModel):
    positions: dict[str, float]  # {ticker: shares}
    cash: float


class HoldingsUpdate(BaseModel):
    positions: dict[str, float]   # {ticker: shares}
    cash: float
    avg_costs: dict[str, float] = {}  # {ticker: avg_cost_per_share}


class PositionUpdate(BaseModel):
    shares: float
    avg_cost: Optional[float] = None


def _check_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("APP_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/signals/latest")
def latest_signals():
    with database.connection() as conn:
        latest_date = conn.execute("SELECT MAX(date) FROM signals").fetchone()[0]
        if latest_date is None:
            return []
        rows = conn.execute(
            "SELECT ticker, date, signal, confidence, probabilities "
            "FROM signals WHERE date = ? "
            "ORDER BY COALESCE(json_extract(probabilities, '$.BUY'), confidence) DESC "
            "LIMIT 50",
            (latest_date,),
        ).fetchall()
    result = []
    for r in rows:
        probs = json.loads(r["probabilities"]) if r["probabilities"] else None
        result.append({
            "ticker": r["ticker"],
            "date": r["date"],
            "signal": r["signal"],
            "confidence": r["confidence"],
            "probabilities": probs,
        })
    return result


@app.post("/recommend")
def recommend(h: Holdings):
    try:
        return advisor.recommend_trades(current_holdings=h.positions, cash=h.cash)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/history")
def portfolio_history():
    with database.connection() as conn:
        rows = conn.execute(
            "SELECT id, as_of_date, payload_json, executed, executed_at "
            "FROM recommendations "
            "ORDER BY as_of_date DESC LIMIT 30",
        ).fetchall()
    result = []
    for r in rows:
        payload = json.loads(r["payload_json"])
        result.append({
            "id": r["id"],
            "as_of_date": r["as_of_date"],
            "executed": bool(r["executed"]),
            "executed_at": r["executed_at"],
            **payload,
        })
    return result


@app.get("/paper/performance")
def paper_performance(start_date: Optional[str] = None):
    if ENVIRONMENT == 'cloud':
        raise HTTPException(status_code=503, detail="Not available in cloud mode")
    try:
        return paper_trading.forward_performance_report(start_date=start_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/holdings")
def get_holdings():
    """Returns current user holdings: positions, cash, and total market value."""
    with database.connection() as conn:
        rows = conn.execute(
            "SELECT ticker, shares, avg_cost FROM user_holdings"
        ).fetchall()
        cash_row = conn.execute(
            "SELECT amount FROM user_cash WHERE id = 1"
        ).fetchone()
        cash = float(cash_row["amount"]) if cash_row else 0.0

        positions: dict[str, float] = {}
        avg_costs: dict[str, float] = {}
        total_value = cash

        for r in rows:
            ticker = r["ticker"]
            shares = float(r["shares"])
            positions[ticker] = shares
            if r["avg_cost"] is not None:
                avg_costs[ticker] = float(r["avg_cost"])
            price_row = conn.execute(
                "SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                (ticker,),
            ).fetchone()
            if price_row and price_row["close"] is not None:
                total_value += shares * float(price_row["close"])

    return {
        "positions": positions,
        "cash": cash,
        "avg_costs": avg_costs,
        "total_value_usd": round(total_value, 2),
    }


@app.put("/holdings", dependencies=[Depends(_check_api_key)])
def put_holdings(payload: HoldingsUpdate):
    """Replaces all user holdings with the provided snapshot."""
    now = datetime.now(timezone.utc).isoformat()
    with database.connection() as conn:
        conn.execute("DELETE FROM user_holdings")
        for ticker, shares in payload.positions.items():
            avg_cost = payload.avg_costs.get(ticker.upper())
            conn.execute(
                "INSERT INTO user_holdings (ticker, shares, avg_cost, added_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (ticker.upper(), shares, avg_cost, now, now),
            )
        conn.execute(
            "INSERT OR REPLACE INTO user_cash (id, amount, updated_at) VALUES (1, ?, ?)",
            (payload.cash, now),
        )
    return {"ok": True}


@app.post("/holdings/{ticker}", dependencies=[Depends(_check_api_key)])
def update_position(ticker: str, payload: PositionUpdate):
    """Add or update a single position."""
    ticker = ticker.upper().strip()
    now = datetime.now(timezone.utc).isoformat()
    with database.connection() as conn:
        existing = conn.execute(
            "SELECT added_at FROM user_holdings WHERE ticker = ?", (ticker,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE user_holdings SET shares = ?, avg_cost = ?, updated_at = ? "
                "WHERE ticker = ?",
                (payload.shares, payload.avg_cost, now, ticker),
            )
        else:
            conn.execute(
                "INSERT INTO user_holdings (ticker, shares, avg_cost, added_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (ticker, payload.shares, payload.avg_cost, now, now),
            )
    return {"ok": True, "ticker": ticker}


@app.get("/stock/{ticker}")
def stock_detail(ticker: str):
    """Returns everything the iPhone detail sheet needs in one payload."""
    ticker = ticker.upper().strip()

    from trainer import FEATURE_COLS

    with database.connection() as conn:
        row = conn.execute(
            "SELECT ticker, sector, name FROM stocks WHERE ticker = ?", (ticker,)
        ).fetchone()
        if not row:
            raise HTTPException(404, f"Ticker {ticker} not found in universe")
        meta = dict(row)

        price_rows = conn.execute(
            "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 130",
            (ticker,),
        ).fetchall()
        price_history = list(
            reversed([{"date": r["date"], "close": r["close"]} for r in price_rows])
        )

        sig_row = conn.execute(
            "SELECT date, signal, confidence, probabilities FROM signals "
            "WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        signal = dict(sig_row) if sig_row else None
        if signal and signal.get("probabilities"):
            signal["probabilities"] = json.loads(signal["probabilities"])

        feat_row = conn.execute(
            "SELECT * FROM features WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        features = dict(feat_row) if feat_row else {}

    try:
        import numpy as np
        import pandas as pd

        model = get_model()
        if features:
            X = pd.DataFrame([features]).reindex(columns=FEATURE_COLS)
            importances = model.model.feature_importances_
            feat_array = X.values[0]
            bounds = model.feature_bounds or {}
            contributions = []
            for col, imp, val in zip(FEATURE_COLS, importances, feat_array):
                is_nan = val != val  # NaN check without numpy at call site
                if col in bounds and not is_nan:
                    lo, hi = bounds[col]
                    if hi != lo:
                        norm = (val - lo) / (hi - lo)
                        norm = (norm - 0.5) * 2.0
                    else:
                        norm = 0.0
                else:
                    norm = 0.0
                contributions.append({
                    "feature": col,
                    "value": None if is_nan else float(val),
                    "importance": float(imp),
                    "contribution": float(imp * norm),
                })
            top5 = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)[:5]
        else:
            top5 = []
    except Exception:
        top5 = []

    return {
        "ticker": ticker,
        "meta": meta,
        "price_history": price_history,
        "signal": signal,
        "features_latest": {
            k: v for k, v in features.items() if k in FEATURE_COLS or k == "date"
        },
        "top_contributions": top5,
    }


@app.get("/portfolio/overview")
def portfolio_overview():
    """Single payload: NAV, vs-SPY %, Sharpe, max drawdown, top 3 picks, holdings summary."""
    import math

    with database.connection() as conn:
        nav_rows = conn.execute(
            "SELECT date, nav_usd, spy_close, n_holdings FROM paper_nav ORDER BY date ASC"
        ).fetchall()

        if not nav_rows:
            return {
                "portfolio_value": 0.0,
                "vs_spy_pct": 0.0,
                "sharpe": None,
                "max_drawdown": 0.0,
                "top_picks": [],
                "holdings": [],
                "n_days": 0,
                "as_of_date": None,
            }

        nav_vals = [float(r["nav_usd"]) for r in nav_rows]
        spy_vals = [float(r["spy_close"]) for r in nav_rows]
        portfolio_value = nav_vals[-1]
        nav0, spy0 = nav_vals[0], spy_vals[0]

        paper_total = (nav_vals[-1] / nav0 - 1.0) if nav0 > 0 else 0.0
        spy_total = (spy_vals[-1] / spy0 - 1.0) if spy0 > 0 else 0.0
        vs_spy_pct = paper_total - spy_total

        paper_daily = [
            (nav_vals[i] / nav_vals[i - 1] - 1.0) if nav_vals[i - 1] > 0 else 0.0
            for i in range(1, len(nav_vals))
        ]

        if len(paper_daily) > 1:
            mean_r = sum(paper_daily) / len(paper_daily)
            var = sum((x - mean_r) ** 2 for x in paper_daily) / (len(paper_daily) - 1)
            std = math.sqrt(var) if var > 0 else 0.0
            sharpe = round(mean_r / std * math.sqrt(252), 4) if std > 0 else None
        else:
            sharpe = None

        peak, max_dd = nav_vals[0], 0.0
        for v in nav_vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        latest_sig_date = conn.execute("SELECT MAX(date) FROM signals").fetchone()[0]
        top_picks = []
        if latest_sig_date:
            pick_rows = conn.execute(
                "SELECT ticker, signal, confidence, probabilities FROM signals "
                "WHERE date = ? AND signal = 'BUY' "
                "ORDER BY COALESCE(json_extract(probabilities, '$.BUY'), confidence) DESC "
                "LIMIT 3",
                (latest_sig_date,),
            ).fetchall()
            for r in pick_rows:
                probs = json.loads(r["probabilities"]) if r["probabilities"] else None
                top_picks.append({
                    "ticker": r["ticker"],
                    "signal": r["signal"],
                    "confidence": r["confidence"],
                    "probabilities": probs,
                })

        latest_pp_date = conn.execute(
            "SELECT MAX(as_of_date) FROM paper_portfolio"
        ).fetchone()[0]
        holdings = []
        if latest_pp_date:
            h_rows = conn.execute(
                "SELECT ticker, target_shares, target_weight, entry_price "
                "FROM paper_portfolio WHERE as_of_date = ?",
                (latest_pp_date,),
            ).fetchall()
            holdings = [dict(r) for r in h_rows]

    return {
        "portfolio_value": round(portfolio_value, 2),
        "vs_spy_pct": round(vs_spy_pct, 6),
        "sharpe": sharpe,
        "max_drawdown": round(max_dd, 6),
        "top_picks": top_picks,
        "holdings": holdings,
        "n_days": len(nav_rows),
        "as_of_date": nav_rows[-1]["date"],
    }


@app.get("/model/info")
def model_info():
    try:
        from ranker_trainer import RANKER_PATH

        model, model_type = advisor._load_model()
        info: dict = {
            "model_type": model_type,
            "split_dates": getattr(model, "split_dates", {}),
            "metrics": getattr(model, "metrics", {}),
            "trained_at": None,
        }
        try:
            mtime = os.path.getmtime(RANKER_PATH)
            info["trained_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except OSError:
            pass
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
