"""
advisor.py

Portfolio recommendation engine: given current holdings + cash + today's
model predictions, recommends specific trades.

Usage
-----
    python advisor.py --holdings holdings.json --cash 1500
    where holdings.json has format {'AAPL': 10, 'MSFT': 5}
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import database
import prediction_guard
from trainer import FEATURE_COLS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(model_path=None):
    """Load the production model. Defaults to ranker; falls back to config."""
    from ranker_trainer import UniversalRanker, RANKER_PATH

    target = Path(model_path) if model_path else RANKER_PATH

    if target.exists():
        if "ranker" in target.stem.lower():
            return UniversalRanker.load(target), "ranker"
        try:
            from trainer import UniversalStockModel
            return UniversalStockModel.load(target), "classifier"
        except Exception:
            pass

    from production_config import load_production_model
    log.info("Falling back to production_config model.")
    return load_production_model()


def _load_latest_features() -> pd.DataFrame:
    """Load the most recent feature row per ticker, joined with sector."""
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
    try:
        with database.connection() as conn:
            rows = conn.execute(sql).fetchall()
    except Exception as exc:
        log.error("Failed to load features: %s", exc)
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    log.info("Loaded %d feature rows for %s.", len(df), df["date"].iloc[0])
    return df


def _load_latest_prices(tickers: list) -> dict:
    """Return {ticker: latest_close} for each requested ticker."""
    if not tickers:
        return {}
    placeholders = ",".join("?" * len(tickers))
    sql = f"""
        SELECT p.ticker, p.close
        FROM prices p
        INNER JOIN (
            SELECT ticker, MAX(date) AS max_date
            FROM prices
            WHERE ticker IN ({placeholders})
            GROUP BY ticker
        ) m ON p.ticker = m.ticker AND p.date = m.max_date
    """
    with database.connection() as conn:
        rows = conn.execute(sql, tickers).fetchall()
    return {r["ticker"]: float(r["close"]) for r in rows if r["close"] is not None}


def _compute_volatility(tickers: list, lookback: int = 30) -> dict:
    """Return {ticker: realized_vol} from the last `lookback` daily returns."""
    if not tickers:
        return {}
    placeholders = ",".join("?" * len(tickers))
    sql = f"""
        SELECT ticker, close
        FROM (
            SELECT ticker, close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker IN ({placeholders})
        )
        WHERE rn <= {lookback + 1}
        ORDER BY ticker, rn DESC
    """
    with database.connection() as conn:
        rows = conn.execute(sql, tickers).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    result: dict = {}
    if df.empty:
        return result
    for ticker, grp in df.groupby("ticker"):
        closes = grp["close"].values.astype(float)
        if len(closes) < 5:
            result[ticker] = 0.02
        else:
            rets = np.diff(closes) / closes[:-1]
            v = float(np.std(rets, ddof=1))
            result[ticker] = max(v, 1e-4)
    return result


def _compute_scores(model, model_type: str, df: pd.DataFrame) -> np.ndarray:
    """Return a 1-D score array (higher = more bullish)."""
    if model_type == "ranker":
        return model.predict(df)
    from trainer import LABEL_BUY
    proba = model.predict_proba(df)
    return proba[:, LABEL_BUY]


def _persist_recommendation(payload: dict) -> None:
    """Write the recommendation payload to the recommendations table."""
    rec_json = json.dumps(payload, default=str)
    with database.connection() as conn:
        conn.execute(
            "INSERT INTO recommendations (as_of_date, payload_json, executed, executed_at) "
            "VALUES (?, ?, 0, NULL)",
            (payload["as_of_date"], rec_json),
        )
    log.info("Persisted recommendation for %s.", payload["as_of_date"])


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def recommend_trades(
    current_holdings: dict,   # {ticker: shares}
    cash: float,
    model_path: str = None,
    top_n: int = 20,
    sector_cap: float = 0.30,
    cash_buffer: float = 0.05,
    min_trade_usd: float = 100.0,
    max_total_turnover: float = 0.50,
    round_trip_cost: float = 0.012,
) -> dict:
    """
    Returns a dict with structure:
    {
      'as_of_date': '2026-04-24',
      'current_value_usd': 12345.67,
      'target_holdings': {'AAPL': 25, 'MSFT': 10, ...},
      'trades': [
         {'ticker': 'NVDA', 'action': 'BUY',  'shares': 5,
          'price': 850.00, 'usd': 4250.00, 'reason': 'rank=12, score=0.87'},
         {'ticker': 'XOM',  'action': 'SELL', 'shares': 20,
          'price': 110.00, 'usd': 2200.00, 'reason': 'no longer in top 20'},
         {'ticker': 'AAPL', 'action': 'TRIM', 'shares': 5,
          'price': 220.00, 'usd': 1100.00, 'reason': 'sector cap reached'},
      ],
      'expected_cost_drag_usd': 47.20,
      'cash_after_trades': 615.50,
      'sector_breakdown': {'IT': 0.28, 'HC': 0.18, ...},
      'risk_warnings': ['turnover 38% — high but within cap', ...]
    }
    """
    # Ensure recommendations table exists (idempotent)
    database.initialize()

    # 0. Load model
    model, model_type = _load_model(model_path)

    # 1. Load latest features
    df = _load_latest_features()
    if df.empty:
        raise RuntimeError("No features found in DB — run feature_engine.py first.")
    as_of_date = str(df["date"].iloc[0])

    # 2. Validate feature bounds
    if model.feature_bounds:
        df, _ = prediction_guard.validate_feature_ranges(df, model.feature_bounds)

    # 3. Compute scores and sort descending
    scores = _compute_scores(model, model_type, df)
    df = df.copy()
    df["_score"] = scores
    df = df.sort_values("_score", ascending=False).reset_index(drop=True)

    score_map: dict = dict(zip(df["ticker"], df["_score"]))
    raw_sector = df["sector"].fillna("") if "sector" in df.columns else pd.Series([""] * len(df))
    sector_map: dict = {t: (s if s else "Unknown") for t, s in zip(df["ticker"], raw_sector)}

    # 4. Greedy sector-cap selection
    # Allow at most floor(sector_cap * top_n) stocks per sector (min 1)
    max_per_sector = max(1, int(sector_cap * top_n))
    selected: list = []
    sector_counts: dict = {}

    for _, row in df.iterrows():
        if len(selected) >= top_n:
            break
        ticker = row["ticker"]
        sector = sector_map.get(ticker, "Unknown")
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # 5. Load prices for all relevant tickers
    all_tickers = list(set(selected) | set(current_holdings.keys()))
    prices = _load_latest_prices(all_tickers)

    # 6. Current portfolio value
    current_value = cash + sum(
        int(current_holdings.get(t, 0)) * prices.get(t, 0)
        for t in current_holdings
    )

    # 7. Inverse-vol weights with iterative sector-cap enforcement
    tickers_with_prices = [t for t in selected if t in prices]
    vols = _compute_volatility(tickers_with_prices)
    inv_vols: dict = {t: 1.0 / max(vols.get(t, 0.02), 1e-4) for t in tickers_with_prices}

    for _ in range(8):
        total_iv = sum(inv_vols.values()) or 1.0
        sec_iv: dict = {}
        for t, iv in inv_vols.items():
            sec = sector_map.get(t, "Unknown")
            sec_iv[sec] = sec_iv.get(sec, 0) + iv
        sec_weight = {s: v / total_iv * (1 - cash_buffer) for s, v in sec_iv.items()}
        over_cap = {s for s, w in sec_weight.items() if w > sector_cap + 1e-6}
        if not over_cap:
            break
        for t in list(inv_vols.keys()):
            sec = sector_map.get(t, "Unknown")
            if sec in over_cap:
                inv_vols[t] *= sector_cap / sec_weight[sec]

    total_iv = sum(inv_vols.values()) or 1.0
    target_weights: dict = {
        t: (inv_vols[t] / total_iv) * (1 - cash_buffer)
        for t in inv_vols
    }

    # Sector weights after inverse-vol (for TRIM reason detection)
    sector_target_w: dict = {}
    for t, w in target_weights.items():
        sec = sector_map.get(t, "Unknown")
        sector_target_w[sec] = sector_target_w.get(sec, 0) + w

    # 8. Convert weights to target shares (floor to avoid overspend)
    target_shares: dict = {}
    for ticker in tickers_with_prices:
        price = prices[ticker]
        target_usd = target_weights.get(ticker, 0) * current_value
        target_shares[ticker] = int(target_usd / price)

    # Tickers not selected → target = 0 (SELL if currently held)
    for ticker in current_holdings:
        if ticker not in target_shares:
            target_shares[ticker] = 0

    # 9. Compute raw trade list
    raw_trades: list = []
    for ticker, target in target_shares.items():
        current = int(current_holdings.get(ticker, 0))
        delta = target - current
        if delta == 0:
            continue
        price = prices.get(ticker, 0)
        if not price:
            continue
        raw_trades.append({
            "ticker":  ticker,
            "delta":   delta,
            "current": current,
            "target":  target,
            "price":   price,
            "usd":     abs(delta * price),
        })

    # 10. Drop tiny trades
    raw_trades = [t for t in raw_trades if t["usd"] >= min_trade_usd]

    # 11. Turnover check + proportional scaling
    total_trade_usd = sum(t["usd"] for t in raw_trades)
    turnover = total_trade_usd / current_value if current_value > 0 else 0.0

    if turnover > max_total_turnover and raw_trades:
        scale = (max_total_turnover * current_value) / total_trade_usd
        for t in raw_trades:
            new_delta = int(t["delta"] * scale)
            if new_delta == 0:
                new_delta = 1 if t["delta"] > 0 else -1
            t["delta"] = new_delta
            t["usd"] = abs(new_delta * t["price"])
        raw_trades = [t for t in raw_trades if t["usd"] >= min_trade_usd]
        total_trade_usd = sum(t["usd"] for t in raw_trades)
        turnover = total_trade_usd / current_value if current_value > 0 else 0.0

    # 12. Classify trades
    sorted_scores = sorted(score_map.values(), reverse=True)

    trades: list = []
    cash_delta = 0.0

    for t in raw_trades:
        ticker  = t["ticker"]
        current = t["current"]
        delta   = t["delta"]
        actual_target = current + delta

        # BUY: was 0, now > 0 | SELL: was > 0, now 0 | ADD: was > 0, now larger
        # TRIM: was > 0, now smaller (still > 0)
        if current == 0:
            action = "BUY"
        elif actual_target == 0:
            action = "SELL"
        elif delta > 0:
            action = "ADD"
        else:
            action = "TRIM"

        score = score_map.get(ticker)
        reason_parts: list = []
        if score is not None:
            try:
                rank = sorted_scores.index(score) + 1
            except ValueError:
                rank = "?"
            reason_parts.append(f"rank={rank}, score={score:.4f}")

        if action == "SELL":
            reason_parts.append(f"no longer in top {top_n}")
        elif action == "TRIM":
            sec = sector_map.get(ticker, "Unknown")
            if sector_target_w.get(sec, 0) >= sector_cap - 0.01:
                reason_parts.append("sector cap reached")
            else:
                reason_parts.append("target weight reduced")

        if action in ("BUY", "ADD"):
            cash_delta -= t["usd"]
        else:
            cash_delta += t["usd"]

        trades.append({
            "ticker": ticker,
            "action": action,
            "shares": abs(delta),
            "price":  round(t["price"], 2),
            "usd":    round(t["usd"], 2),
            "reason": ", ".join(reason_parts) if reason_parts else action,
        })

    # 13. Final holdings (after actual executed trades)
    final_holdings: dict = {k: int(v) for k, v in current_holdings.items()}
    for t in raw_trades:
        ticker = t["ticker"]
        final_holdings[ticker] = final_holdings.get(ticker, 0) + t["delta"]
    final_holdings = {k: int(v) for k, v in final_holdings.items() if v > 0}

    # 14. Sector breakdown (fraction of current_value in each sector)
    sector_usd: dict = {}
    for ticker, shares in final_holdings.items():
        price = prices.get(ticker, 0)
        sec = sector_map.get(ticker, "") or "Unknown"
        sector_usd[sec] = sector_usd.get(sec, 0) + shares * price

    sector_breakdown: dict = {}
    if current_value > 0:
        sector_breakdown = {
            s: round(usd / current_value, 4)
            for s, usd in sorted(sector_usd.items(), key=lambda x: -x[1])
        }

    # 15. Cost drag and remaining cash
    expected_cost_drag = sum(tr["usd"] for tr in trades) * round_trip_cost
    cash_after = cash + cash_delta

    # 16. Risk warnings
    risk_warnings: list = []
    if turnover > 0.25:
        flag = "high but within cap" if turnover <= max_total_turnover else "EXCEEDS cap"
        risk_warnings.append(f"turnover {turnover*100:.0f}% — {flag}")
    for sec, weight in sector_breakdown.items():
        if weight > sector_cap:
            risk_warnings.append(
                f"sector {sec} at {weight:.1%} exceeds {sector_cap:.0%} cap"
            )
    if cash_after < 0:
        risk_warnings.append(
            f"insufficient cash: would need ${abs(cash_after):.2f} more"
        )

    result = {
        "as_of_date":            as_of_date,
        "current_value_usd":     round(current_value, 2),
        "target_holdings":       final_holdings,
        "trades":                trades,
        "expected_cost_drag_usd": round(expected_cost_drag, 2),
        "cash_after_trades":     round(cash_after, 2),
        "sector_breakdown":      sector_breakdown,
        "risk_warnings":         risk_warnings,
    }

    _persist_recommendation(result)
    return result


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------

def _print_recommendation(rec: dict) -> None:
    W = 78
    sep  = "=" * W
    thin = "-" * W

    print(f"\n{sep}")
    print(f"  PORTFOLIO RECOMMENDATION  —  {rec['as_of_date']}")
    print(sep)
    print(f"  Current Portfolio Value  : ${rec['current_value_usd']:>13,.2f}")
    print(f"  Estimated Cost Drag      : ${rec['expected_cost_drag_usd']:>13,.2f}")
    print(f"  Cash After Trades        : ${rec['cash_after_trades']:>13,.2f}")
    print(sep)

    trades = rec["trades"]
    if trades:
        print(f"\nTRADES  ({len(trades)})")
        print(thin)
        print(f"  {'Action':<6}  {'Ticker':<7}  {'Shares':>6}  {'Price':>10}  {'USD':>10}  Reason")
        print(thin)
        for tr in trades:
            print(
                f"  {tr['action']:<6}  {tr['ticker']:<7}  {tr['shares']:>6}  "
                f"${tr['price']:>9,.2f}  ${tr['usd']:>9,.2f}  {tr['reason']}"
            )
        print(thin)
    else:
        print(f"\n  No trades recommended.")

    sb = rec["sector_breakdown"]
    if sb:
        print(f"\nSECTOR BREAKDOWN")
        print(thin)
        for sec, w in sb.items():
            bar = "█" * int(w * 40)
            print(f"  {sec:<38}  {w:>6.1%}  {bar}")

    if rec["risk_warnings"]:
        print(f"\nRISK WARNINGS")
        print(thin)
        for w in rec["risk_warnings"]:
            print(f"  ! {w}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Portfolio recommendation engine."
    )
    parser.add_argument(
        "--holdings", required=True,
        help="Path to JSON file: {ticker: shares}",
    )
    parser.add_argument(
        "--cash", type=float, required=True,
        help="Available cash in USD.",
    )
    parser.add_argument("--top-n",    type=int,   default=20)
    parser.add_argument("--model-path",            default=None)
    parser.add_argument("--sector-cap",  type=float, default=0.30)
    parser.add_argument("--cash-buffer", type=float, default=0.05)
    args = parser.parse_args()

    holdings_path = Path(args.holdings)
    if not holdings_path.exists():
        print(f"Error: holdings file not found: {holdings_path}", file=sys.stderr)
        sys.exit(1)

    with holdings_path.open() as fh:
        holdings = json.load(fh)

    rec = recommend_trades(
        current_holdings=holdings,
        cash=args.cash,
        model_path=args.model_path,
        top_n=args.top_n,
        sector_cap=args.sector_cap,
        cash_buffer=args.cash_buffer,
    )

    _print_recommendation(rec)
