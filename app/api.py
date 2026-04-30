import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import advisor
import database

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
