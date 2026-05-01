"""
paper_trading.py

A) record_today()          — snapshot today's BUY signals into paper_portfolio
B) mark_to_market_today()  — price the portfolio at today's closes, log to paper_nav
C) forward_performance_report() — compare cumulative NAV vs SPY
"""

import datetime
import logging
import math
from typing import Optional

import database

log = logging.getLogger(__name__)

STARTING_CAPITAL = 10_000.0
TOP_N = 20
SPY_TICKER = "SPY"


# ---------------------------------------------------------------------------
# A) Record today's recommendation
# ---------------------------------------------------------------------------

def record_today(as_of_date: Optional[str] = None) -> int:
    """
    Snapshot today's top BUY signals into paper_portfolio.
    Skips silently if already recorded for this date.
    Returns number of rows inserted.
    """
    if as_of_date is None:
        as_of_date = datetime.date.today().isoformat()

    with database.connection() as conn:
        existing = conn.execute(
            "SELECT COUNT(*) FROM paper_portfolio WHERE as_of_date = ?",
            (as_of_date,),
        ).fetchone()[0]
        if existing > 0:
            log.info("paper_portfolio already has %d rows for %s — skipping.", existing, as_of_date)
            return 0

        # Most recent signal date ≤ as_of_date (anti-lookahead)
        signal_date = conn.execute(
            "SELECT MAX(date) FROM signals WHERE date <= ?",
            (as_of_date,),
        ).fetchone()[0]
        if signal_date is None:
            log.warning("No signals found for date ≤ %s.", as_of_date)
            return 0

        rows = conn.execute(
            """
            SELECT ticker,
                   COALESCE(json_extract(probabilities, '$.BUY'), confidence) AS buy_prob
            FROM signals
            WHERE date = ? AND signal = 'BUY'
            ORDER BY buy_prob DESC
            LIMIT ?
            """,
            (signal_date, TOP_N),
        ).fetchall()

        if not rows:
            log.warning("No BUY signals on %s.", signal_date)
            return 0

        tickers = [r["ticker"] for r in rows]

        # Entry prices as of signal_date or earlier (no look-ahead)
        placeholders = ",".join("?" * len(tickers))
        price_rows = conn.execute(
            f"""
            SELECT p.ticker, p.close
            FROM prices p
            INNER JOIN (
                SELECT ticker, MAX(date) AS max_date
                FROM prices
                WHERE ticker IN ({placeholders}) AND date <= ?
                GROUP BY ticker
            ) m ON p.ticker = m.ticker AND p.date = m.max_date
            """,
            (*tickers, signal_date),
        ).fetchall()
        prices = {r["ticker"]: float(r["close"]) for r in price_rows if r["close"] is not None}

        tickers = [t for t in tickers if t in prices]
        if not tickers:
            log.warning("No prices available for any BUY signal on %s.", signal_date)
            return 0

        # Use most recent NAV as capital; fall back to STARTING_CAPITAL on first run
        nav_row = conn.execute(
            "SELECT nav_usd FROM paper_nav WHERE date <= ? ORDER BY date DESC LIMIT 1",
            (as_of_date,),
        ).fetchone()
        capital = float(nav_row["nav_usd"]) if nav_row else STARTING_CAPITAL

        n = len(tickers)
        weight = 1.0 / n

        inserted = 0
        for ticker in tickers:
            price = prices[ticker]
            target_usd = weight * capital
            target_shares = math.floor(target_usd / price) if price > 0 else 0
            conn.execute(
                """
                INSERT OR IGNORE INTO paper_portfolio
                    (as_of_date, ticker, target_shares, target_weight, entry_price)
                VALUES (?, ?, ?, ?, ?)
                """,
                (as_of_date, ticker, float(target_shares), weight, price),
            )
            inserted += 1

        log.info("Recorded %d paper_portfolio rows for %s.", inserted, as_of_date)
        return inserted


# ---------------------------------------------------------------------------
# B) Mark-to-market
# ---------------------------------------------------------------------------

def mark_to_market_today(as_of_date: Optional[str] = None) -> Optional[dict]:
    """
    Compute paper portfolio NAV as of as_of_date and log to paper_nav.
    Only uses prices from dates ≤ as_of_date (no look-ahead).
    Returns the NAV row dict, or None if no portfolio data exists yet.
    """
    if as_of_date is None:
        as_of_date = datetime.date.today().isoformat()

    with database.connection() as conn:
        existing = conn.execute(
            "SELECT nav_usd, spy_close, n_holdings FROM paper_nav WHERE date = ?",
            (as_of_date,),
        ).fetchone()
        if existing:
            log.info("paper_nav already has entry for %s.", as_of_date)
            return {"date": as_of_date, **dict(existing)}

        # Most recent portfolio snapshot ≤ as_of_date
        latest_portfolio_date = conn.execute(
            "SELECT MAX(as_of_date) FROM paper_portfolio WHERE as_of_date <= ?",
            (as_of_date,),
        ).fetchone()[0]
        if latest_portfolio_date is None:
            log.warning("No paper_portfolio data for date ≤ %s.", as_of_date)
            return None

        holdings = conn.execute(
            "SELECT ticker, target_shares FROM paper_portfolio WHERE as_of_date = ?",
            (latest_portfolio_date,),
        ).fetchall()
        if not holdings:
            return None

        tickers = [r["ticker"] for r in holdings]
        shares_map = {r["ticker"]: float(r["target_shares"]) for r in holdings}

        # Latest close per holding ≤ as_of_date (strict no look-ahead)
        placeholders = ",".join("?" * len(tickers))
        price_rows = conn.execute(
            f"""
            SELECT p.ticker, p.close
            FROM prices p
            INNER JOIN (
                SELECT ticker, MAX(date) AS max_date
                FROM prices
                WHERE ticker IN ({placeholders}) AND date <= ?
                GROUP BY ticker
            ) m ON p.ticker = m.ticker AND p.date = m.max_date
            """,
            (*tickers, as_of_date),
        ).fetchall()
        prices = {r["ticker"]: float(r["close"]) for r in price_rows if r["close"] is not None}

        nav = sum(shares_map[t] * prices.get(t, 0.0) for t in tickers)

        spy_row = conn.execute(
            "SELECT close FROM prices WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1",
            (SPY_TICKER, as_of_date),
        ).fetchone()
        spy_close = float(spy_row["close"]) if spy_row else 0.0

        n_holdings = sum(1 for t in tickers if shares_map[t] > 0 and t in prices)

        conn.execute(
            "INSERT OR REPLACE INTO paper_nav (date, nav_usd, spy_close, n_holdings) VALUES (?, ?, ?, ?)",
            (as_of_date, nav, spy_close, n_holdings),
        )

        log.info("paper_nav: %s  NAV=$%.2f  SPY=%.2f  n=%d", as_of_date, nav, spy_close, n_holdings)
        return {"date": as_of_date, "nav_usd": nav, "spy_close": spy_close, "n_holdings": n_holdings}


# ---------------------------------------------------------------------------
# C) Forward performance report
# ---------------------------------------------------------------------------

def forward_performance_report(start_date: Optional[str] = None) -> dict:
    """
    Compare paper portfolio vs SPY since start_date (or earliest paper_nav row).

    Returns dict with cumulative_return, spy_return, alpha (annualized or simple),
    tracking_error (annualized), hit_rate, monthly_buckets, and metadata.
    """
    with database.connection() as conn:
        if start_date is None:
            row = conn.execute("SELECT MIN(date) FROM paper_nav").fetchone()
            start_date = row[0] if row and row[0] else None

        if start_date is None:
            return _empty_report(start_date)

        rows = conn.execute(
            "SELECT date, nav_usd, spy_close FROM paper_nav WHERE date >= ? ORDER BY date ASC",
            (start_date,),
        ).fetchall()

    if not rows:
        return _empty_report(start_date)

    dates    = [r["date"]      for r in rows]
    nav_vals = [float(r["nav_usd"])   for r in rows]
    spy_vals = [float(r["spy_close"]) for r in rows]
    n_days   = len(rows)

    nav0 = nav_vals[0]
    spy0 = spy_vals[0]

    cumulative_return = (nav_vals[-1] / nav0 - 1.0) if nav0 > 0 else 0.0
    spy_return        = (spy_vals[-1] / spy0 - 1.0) if spy0 > 0 else 0.0

    # Daily excess returns
    paper_daily = []
    spy_daily   = []
    for i in range(1, n_days):
        pr = (nav_vals[i] / nav_vals[i - 1] - 1.0) if nav_vals[i - 1] > 0 else 0.0
        sr = (spy_vals[i] / spy_vals[i - 1] - 1.0) if spy_vals[i - 1] > 0 else 0.0
        paper_daily.append(pr)
        spy_daily.append(sr)

    # Tracking error: annualized std of daily (paper - spy)
    if len(paper_daily) > 1:
        excess  = [p - s for p, s in zip(paper_daily, spy_daily)]
        mean_ex = sum(excess) / len(excess)
        var     = sum((x - mean_ex) ** 2 for x in excess) / (len(excess) - 1)
        tracking_error = math.sqrt(var) * math.sqrt(252)
    else:
        tracking_error = 0.0

    # Alpha: annualized when ≥ 126 trading days (≈ 6 months), simple otherwise
    n_years = n_days / 252
    if n_years >= 0.5 and nav0 > 0 and spy0 > 0:
        paper_ann = (nav_vals[-1] / nav0) ** (1.0 / n_years) - 1.0
        spy_ann   = (spy_vals[-1] / spy0) ** (1.0 / n_years) - 1.0
        alpha = paper_ann - spy_ann
    else:
        alpha = cumulative_return - spy_return

    # Hit rate: fraction of trading days paper beat SPY (daily return basis)
    if paper_daily:
        hit_rate = sum(1 for p, s in zip(paper_daily, spy_daily) if p > s) / len(paper_daily)
    else:
        hit_rate = 0.0

    # Monthly buckets: compare start-of-month vs end-of-month for paper and SPY
    monthly: dict = {}
    for i, date_str in enumerate(dates):
        month_key = date_str[:7]
        if month_key not in monthly:
            monthly[month_key] = {
                "nav_start": nav_vals[i], "nav_end": nav_vals[i],
                "spy_start": spy_vals[i], "spy_end": spy_vals[i],
            }
        else:
            monthly[month_key]["nav_end"] = nav_vals[i]
            monthly[month_key]["spy_end"] = spy_vals[i]

    monthly_buckets = []
    for month_key in sorted(monthly):
        m = monthly[month_key]
        m_paper = (m["nav_end"] / m["nav_start"] - 1.0) if m["nav_start"] > 0 else 0.0
        m_spy   = (m["spy_end"] / m["spy_start"] - 1.0) if m["spy_start"] > 0 else 0.0
        monthly_buckets.append({
            "month":        month_key,
            "paper_return": round(m_paper, 6),
            "spy_return":   round(m_spy, 6),
            "alpha":        round(m_paper - m_spy, 6),
        })

    return {
        "cumulative_return": round(cumulative_return, 6),
        "spy_return":        round(spy_return, 6),
        "alpha":             round(alpha, 6),
        "tracking_error":    round(tracking_error, 6),
        "hit_rate":          round(hit_rate, 6),
        "monthly_buckets":   monthly_buckets,
        "start_date":        dates[0],
        "end_date":          dates[-1],
        "n_days":            n_days,
    }


def _empty_report(start_date: Optional[str]) -> dict:
    return {
        "cumulative_return": 0.0,
        "spy_return":        0.0,
        "alpha":             0.0,
        "tracking_error":    0.0,
        "hit_rate":          0.0,
        "monthly_buckets":   [],
        "start_date":        start_date,
        "end_date":          None,
        "n_days":            0,
    }
