"""
sentiment_collector.py

Fetches financial news sentiment from two sources and stores daily
per-ticker scores in the `sentiment` table.

Sources
-------
1. Polygon.io   — article-level pre-scored sentiment via `insights` field.
                  Fetches general market news pages (up to 1 000 articles each)
                  and extracts ticker-level sentiment from the `insights` array.
                  Budget: 150 requests/day. Safety buffer: 10. Effective: 140/day.

2. NewsAPI      — headline aggregation. Batches 10 tickers per request, scores
                  headlines using a financial keyword word-list heuristic.
                  Budget: 100 requests/day. Safety buffer: 5. Effective: 95/day.

Token management
----------------
Daily call counts are tracked in the `api_usage` table (service, date, calls_used).
Each collector checks remaining budget before every request and stops cleanly when
exhausted. A safety buffer is kept in reserve for emergencies.

Detrimentality check
--------------------
After fetching, the coverage fraction (tickers_covered / total_tracked) is compared
to MIN_COVERAGE_FRACTION (10%). If coverage is below this threshold, the data is
considered too sparse to add signal and is NOT written to the DB. This prevents
low-quality partial sentiment from distorting the model.

Entry points
------------
    collect_sentiment(days_back=30)  — run both collectors
    collect_polygon_sentiment(...)   — Polygon only
    collect_newsapi_sentiment(...)   — NewsAPI only
    audit_sentiment()                — print coverage stats
"""

import logging
import os
import re
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

import requests
from dotenv import load_dotenv

import database

load_dotenv()

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
POLYGON_KEY  = os.environ["POLYGON_API_KEY"]
NEWSAPI_KEY  = os.environ["NEWSAPI_KEY"]

# ---------------------------------------------------------------------------
# Budget configuration
# ---------------------------------------------------------------------------
POLYGON_DAILY_LIMIT    = 150
POLYGON_SAFETY_BUFFER  = 10     # keep 10 in reserve — 140 effective per day
NEWSAPI_DAILY_LIMIT    = 100
NEWSAPI_SAFETY_BUFFER  = 5      # keep 5 in reserve — 95 effective per day

POLYGON_ARTICLES_PER_PAGE = 1000   # max Polygon allows per request
NEWSAPI_BATCH_SIZE        = 10     # tickers per NewsAPI request (OR query)
NEWSAPI_PAGE_SIZE         = 100    # max articles per NewsAPI request (free tier)

# Minimum fraction of tracked tickers that must have data for a batch to be
# considered non-detrimental and worth storing.
MIN_COVERAGE_FRACTION = 0.10   # 10% of active tickers

# ---------------------------------------------------------------------------
# Polygon endpoints
# ---------------------------------------------------------------------------
POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"

# ---------------------------------------------------------------------------
# NewsAPI endpoint
# ---------------------------------------------------------------------------
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# ---------------------------------------------------------------------------
# Simple financial sentiment word-lists (used for NewsAPI headline scoring)
# ---------------------------------------------------------------------------
_POS_WORDS = frozenset([
    "beats", "beat", "surges", "surge", "jumps", "jump", "rises", "rise",
    "gains", "gain", "record", "upgrade", "upgrades", "buy", "outperform",
    "exceeds", "exceed", "strong", "growth", "profit", "revenue", "soars",
    "soar", "rallies", "rally", "positive", "bullish", "expansion", "raised",
    "raises", "boosts", "boost", "topped", "tops", "upbeat", "optimistic",
])
_NEG_WORDS = frozenset([
    "misses", "miss", "falls", "fall", "drops", "drop", "cuts", "cut",
    "downgrade", "downgrades", "sell", "underperform", "warns", "warn",
    "loss", "losses", "decline", "declines", "weak", "disappoints",
    "disappoint", "negative", "bearish", "contraction", "layoffs",
    "layoff", "investigation", "lawsuit", "recall", "charges", "slumps",
    "slump", "plunges", "plunge", "tumbles", "tumble", "missed", "lowered",
])


# ---------------------------------------------------------------------------
# Schema — api_usage table
# ---------------------------------------------------------------------------

_CREATE_API_USAGE_SQL = """
CREATE TABLE IF NOT EXISTS api_usage (
    service    TEXT    NOT NULL,
    date       TEXT    NOT NULL,
    calls_used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (service, date)
);
"""


def _ensure_api_usage_table() -> None:
    with database.connection() as conn:
        conn.executescript(_CREATE_API_USAGE_SQL)


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

def _get_calls_used(service: str, today: str) -> int:
    with database.connection() as conn:
        row = conn.execute(
            "SELECT calls_used FROM api_usage WHERE service=? AND date=?",
            (service, today),
        ).fetchone()
    return row[0] if row else 0


def _increment_calls(service: str, today: str, n: int = 1) -> None:
    with database.connection() as conn:
        conn.execute(
            """
            INSERT INTO api_usage (service, date, calls_used)
            VALUES (?, ?, ?)
            ON CONFLICT(service, date) DO UPDATE SET
                calls_used = calls_used + ?
            """,
            (service, today, n, n),
        )


def _budget_remaining(service: str, daily_limit: int, buffer: int) -> int:
    today = date.today().isoformat()
    used  = _get_calls_used(service, today)
    return max(0, daily_limit - buffer - used)


# ---------------------------------------------------------------------------
# Sentiment storage
# ---------------------------------------------------------------------------

_UPSERT_SENTIMENT_SQL = """
INSERT INTO sentiment (ticker, date, news_sentiment)
VALUES (:ticker, :date, :news_sentiment)
ON CONFLICT(ticker, date) DO UPDATE SET
    news_sentiment = excluded.news_sentiment
"""


def _save_sentiment(records: list[dict]) -> int:
    if not records:
        return 0
    with database.connection() as conn:
        conn.executemany(_UPSERT_SENTIMENT_SQL, records)
    return len(records)


def _get_tracked_tickers() -> list[str]:
    with database.connection() as conn:
        rows = conn.execute(
            "SELECT ticker FROM stocks WHERE is_active=1"
        ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_headline(text: str) -> float:
    """
    Simple financial word-list sentiment score. Range: -1.0 to +1.0.
    Returns 0.0 if no positive or negative words are found.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    total = pos + neg
    return round((pos - neg) / total, 4) if total > 0 else 0.0


def _aggregate(scores: list[float]) -> Optional[float]:
    return round(sum(scores) / len(scores), 4) if scores else None


def _is_detrimental(tickers_covered: int, total_tracked: int) -> bool:
    if total_tracked == 0:
        return True
    return (tickers_covered / total_tracked) < MIN_COVERAGE_FRACTION


# ---------------------------------------------------------------------------
# Polygon.io collector
# ---------------------------------------------------------------------------

def collect_polygon_sentiment(days_back: int = 30) -> dict:
    """
    Fetch news pages from Polygon and extract per-ticker daily sentiment.

    Strategy
    --------
    Polygon's /v2/reference/news endpoint returns up to 1 000 articles per
    request with an `insights` array that contains pre-scored sentiment per
    ticker mentioned in each article. By fetching general market news (no
    ticker filter) we can cover all 491 tickers in a single page batch —
    far more efficient than one request per ticker.

    With 140 effective requests/day × 1 000 articles = up to 140 000 articles.

    Detrimentality
    --------------
    If the Polygon `insights` field is sparse (e.g. on free tier) and fewer
    than 10% of tracked tickers get any data, the batch is flagged detrimental
    and NOT written to the DB.
    """
    _ensure_api_usage_table()
    today      = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days_back)).isoformat()

    budget = _budget_remaining("polygon", POLYGON_DAILY_LIMIT, POLYGON_SAFETY_BUFFER)
    if budget <= 0:
        log.warning(
            "Polygon: daily budget exhausted (%d/%d calls used). Skipping.",
            _get_calls_used("polygon", today), POLYGON_DAILY_LIMIT,
        )
        return {"calls_used": 0, "tickers_covered": 0, "detrimental": False,
                "skipped_reason": "budget_exhausted"}

    tracked = set(_get_tracked_tickers())
    if not tracked:
        log.warning("Polygon: no active tickers in DB. Skipping.")
        return {"calls_used": 0, "tickers_covered": 0, "detrimental": False,
                "skipped_reason": "no_tickers"}

    log.info(
        "Polygon: starting fetch (days_back=%d, budget=%d calls, %d tickers tracked).",
        days_back, budget, len(tracked),
    )

    ticker_date_scores: dict[tuple, list[float]] = defaultdict(list)
    calls_made   = 0
    next_url     = None
    errors       = 0

    base_params = {
        "limit":                POLYGON_ARTICLES_PER_PAGE,
        "published_utc.gte":    start_date,
        "published_utc.lte":    today,
        "order":                "desc",
        "sort":                 "published_utc",
    }

    while calls_made < budget:
        try:
            if next_url:
                # Polygon pagination: next_url is a full URL without apiKey
                resp = requests.get(
                    next_url,
                    params={"apiKey": POLYGON_KEY},
                    timeout=30,
                )
            else:
                resp = requests.get(
                    POLYGON_NEWS_URL,
                    params={**base_params, "apiKey": POLYGON_KEY},
                    timeout=30,
                )
            resp.raise_for_status()
            data = resp.json()

        except requests.HTTPError as exc:
            errors += 1
            log.error("Polygon HTTP %s: %s", resp.status_code, exc)
            if resp.status_code in (401, 403, 429):
                log.warning("Polygon: auth/rate error — stopping early.")
                break
            if errors >= 3:
                log.warning("Polygon: 3 consecutive errors — stopping.")
                break
            time.sleep(2.0)
            continue

        except Exception as exc:
            errors += 1
            log.error("Polygon request failed: %s", exc)
            if errors >= 3:
                break
            time.sleep(2.0)
            continue

        errors = 0  # reset on success
        _increment_calls("polygon", today)
        calls_made += 1

        articles = data.get("results", [])
        if not articles:
            log.info("Polygon: no more articles (page %d).", calls_made)
            break

        for article in articles:
            pub_date = (article.get("published_utc") or "")[:10]
            if not pub_date:
                continue

            for insight in article.get("insights", []):
                tkr = insight.get("ticker", "")
                if tkr not in tracked:
                    continue
                raw = insight.get("sentiment", "neutral")
                score = 1.0 if raw == "positive" else (-1.0 if raw == "negative" else 0.0)
                ticker_date_scores[(tkr, pub_date)].append(score)

        next_url = data.get("next_url")
        if not next_url:
            log.info("Polygon: reached last page after %d calls.", calls_made)
            break

        time.sleep(0.2)  # stay well under 5 req/min free-tier limit

    # ── Aggregate scores ───────────────────────────────────────────────────
    records         = []
    tickers_covered = set()
    for (ticker, pub_date), scores in ticker_date_scores.items():
        agg = _aggregate(scores)
        if agg is not None:
            records.append({"ticker": ticker, "date": pub_date, "news_sentiment": agg})
            tickers_covered.add(ticker)

    # ── Detrimentality check ───────────────────────────────────────────────
    detrimental  = _is_detrimental(len(tickers_covered), len(tracked))
    coverage_pct = round(len(tickers_covered) / len(tracked) * 100, 1) if tracked else 0.0

    if detrimental:
        log.warning(
            "Polygon: coverage %.1f%% < %.0f%% threshold after %d calls. "
            "Data NOT stored — too sparse to be useful. "
            "This is normal on the free tier if `insights` are not populated.",
            coverage_pct, MIN_COVERAGE_FRACTION * 100, calls_made,
        )
    else:
        saved = _save_sentiment(records)
        log.info(
            "Polygon: stored %d records for %d tickers (%.1f%% coverage, %d calls).",
            saved, len(tickers_covered), coverage_pct, calls_made,
        )

    return {
        "calls_used":      calls_made,
        "tickers_covered": len(tickers_covered),
        "coverage_pct":    coverage_pct,
        "detrimental":     detrimental,
    }


# ---------------------------------------------------------------------------
# NewsAPI collector
# ---------------------------------------------------------------------------

def collect_newsapi_sentiment(days_back: int = 30) -> dict:
    """
    Fetch headlines from NewsAPI, score with keyword heuristic, store results.

    Strategy
    --------
    NewsAPI free tier has no built-in sentiment scoring, so we use a financial
    keyword word-list. Tickers are batched NEWSAPI_BATCH_SIZE per request using
    boolean OR queries. With 95 effective requests and 491 tickers:
      ceil(491 / 10) = 50 requests to cover all tickers — well within budget.

    Limitation: free tier limited to 1 month of history. days_back is capped
    at 29 days automatically.

    Detrimentality
    --------------
    Same 10% coverage threshold as Polygon. On the free tier, this should
    comfortably pass since we can cover all tickers in ~50 requests.
    """
    _ensure_api_usage_table()
    today      = date.today().isoformat()
    # Free tier hard limit: articles no older than 1 month
    earliest   = (date.today() - timedelta(days=min(days_back, 29))).isoformat()

    budget = _budget_remaining("newsapi", NEWSAPI_DAILY_LIMIT, NEWSAPI_SAFETY_BUFFER)
    if budget <= 0:
        log.warning(
            "NewsAPI: daily budget exhausted (%d/%d calls used). Skipping.",
            _get_calls_used("newsapi", today), NEWSAPI_DAILY_LIMIT,
        )
        return {"calls_used": 0, "tickers_covered": 0, "detrimental": False,
                "skipped_reason": "budget_exhausted"}

    tracked = _get_tracked_tickers()
    if not tracked:
        log.warning("NewsAPI: no active tickers in DB. Skipping.")
        return {"calls_used": 0, "tickers_covered": 0, "detrimental": False,
                "skipped_reason": "no_tickers"}

    log.info(
        "NewsAPI: starting fetch (days_back=%d capped at 29, budget=%d calls, %d tickers).",
        days_back, budget, len(tracked),
    )

    tracked_set          = set(tracked)
    ticker_date_scores   = defaultdict(list)
    calls_made           = 0
    errors               = 0

    batches = [
        tracked[i : i + NEWSAPI_BATCH_SIZE]
        for i in range(0, len(tracked), NEWSAPI_BATCH_SIZE)
    ]

    for batch in batches:
        if calls_made >= budget:
            log.info("NewsAPI: budget reached after %d calls.", calls_made)
            break

        query = " OR ".join(batch)
        params = {
            "q":        query,
            "from":     earliest,
            "to":       today,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": NEWSAPI_PAGE_SIZE,
            "apiKey":   NEWSAPI_KEY,
        }

        try:
            resp = requests.get(NEWSAPI_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            errors += 1
            log.error("NewsAPI HTTP %s: %s", resp.status_code, exc)
            if resp.status_code in (401, 426, 429):
                log.warning("NewsAPI: auth/rate/plan error — stopping.")
                break
            if errors >= 3:
                break
            time.sleep(2.0)
            continue
        except Exception as exc:
            errors += 1
            log.error("NewsAPI request failed: %s", exc)
            if errors >= 3:
                break
            time.sleep(2.0)
            continue

        errors = 0
        _increment_calls("newsapi", today)
        calls_made += 1

        if data.get("status") != "ok":
            log.warning("NewsAPI non-ok status: %s | code: %s",
                        data.get("status"), data.get("code"))
            continue

        for article in data.get("articles", []):
            pub_date = (article.get("publishedAt") or "")[:10]
            if not pub_date:
                continue

            title       = article.get("title")       or ""
            description = article.get("description") or ""
            full_text   = (title + " " + description).upper()
            score       = _score_headline(title + " " + description)

            # Determine which tickers in this batch appear in the article text
            for tkr in batch:
                if tkr in full_text:
                    ticker_date_scores[(tkr, pub_date)].append(score)

        time.sleep(0.25)

    # ── Aggregate scores ───────────────────────────────────────────────────
    records         = []
    tickers_covered = set()
    for (ticker, pub_date), scores in ticker_date_scores.items():
        agg = _aggregate(scores)
        if agg is not None:
            records.append({"ticker": ticker, "date": pub_date, "news_sentiment": agg})
            tickers_covered.add(ticker)

    # ── Detrimentality check ───────────────────────────────────────────────
    detrimental  = _is_detrimental(len(tickers_covered), len(tracked))
    coverage_pct = round(len(tickers_covered) / len(tracked) * 100, 1) if tracked else 0.0

    if detrimental:
        log.warning(
            "NewsAPI: coverage %.1f%% < %.0f%% threshold after %d calls. "
            "Data NOT stored.",
            coverage_pct, MIN_COVERAGE_FRACTION * 100, calls_made,
        )
    else:
        saved = _save_sentiment(records)
        log.info(
            "NewsAPI: stored %d records for %d tickers (%.1f%% coverage, %d calls).",
            saved, len(tickers_covered), coverage_pct, calls_made,
        )

    return {
        "calls_used":      calls_made,
        "tickers_covered": len(tickers_covered),
        "coverage_pct":    coverage_pct,
        "detrimental":     detrimental,
    }


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def collect_sentiment(days_back: int = 30) -> None:
    """
    Run both Polygon and NewsAPI collectors.
    NewsAPI supplements any gaps left by Polygon's coverage.
    Each collector stops cleanly if its daily budget is already used.
    """
    log.info("=== Sentiment collection start (days_back=%d) ===", days_back)

    poly  = collect_polygon_sentiment(days_back=days_back)
    news  = collect_newsapi_sentiment(days_back=days_back)

    poly_status = "DETRIMENTAL-SKIPPED" if poly["detrimental"] else (
        poly.get("skipped_reason", f"OK ({poly['coverage_pct']}% coverage)")
    )
    news_status = "DETRIMENTAL-SKIPPED" if news["detrimental"] else (
        news.get("skipped_reason", f"OK ({news['coverage_pct']}% coverage)")
    )

    log.info(
        "=== Sentiment collection complete === "
        "Polygon: %d calls, %d tickers, %s | "
        "NewsAPI: %d calls, %d tickers, %s",
        poly["calls_used"], poly["tickers_covered"], poly_status,
        news["calls_used"], news["tickers_covered"], news_status,
    )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_sentiment() -> None:
    """Print sentiment table coverage and today's API usage stats."""
    with database.connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM sentiment WHERE news_sentiment IS NOT NULL"
        ).fetchone()[0]

        tickers = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM sentiment "
            "WHERE news_sentiment IS NOT NULL"
        ).fetchone()[0]

        date_range = conn.execute(
            "SELECT MIN(date), MAX(date) FROM sentiment "
            "WHERE news_sentiment IS NOT NULL"
        ).fetchone()

        today = date.today().isoformat()
        poly_used = conn.execute(
            "SELECT calls_used FROM api_usage WHERE service='polygon' AND date=?",
            (today,),
        ).fetchone()
        news_used = conn.execute(
            "SELECT calls_used FROM api_usage WHERE service='newsapi' AND date=?",
            (today,),
        ).fetchone()

    print("\n=== sentiment audit ===")
    print(f"  Total rows (news_sentiment not null): {total}")
    print(f"  Distinct tickers covered            : {tickers}")
    if date_range and date_range[0]:
        print(f"  Date range  : {date_range[0]}  to  {date_range[1]}")
    print(f"  API usage today ({today}):")
    print(f"    Polygon : {poly_used[0] if poly_used else 0} / {POLYGON_DAILY_LIMIT}  "
          f"(effective budget: {POLYGON_DAILY_LIMIT - POLYGON_SAFETY_BUFFER})")
    print(f"    NewsAPI : {news_used[0] if news_used else 0} / {NEWSAPI_DAILY_LIMIT}  "
          f"(effective budget: {NEWSAPI_DAILY_LIMIT - NEWSAPI_SAFETY_BUFFER})")
    result = "PASS" if total > 0 else "FAIL — no sentiment data stored yet"
    print(f"  {result}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    collect_sentiment(days_back=30)
    audit_sentiment()
