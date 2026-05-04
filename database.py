import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


DB_PATH = Path(os.environ.get('DB_PATH', str(Path(__file__).parent / 'trading.db')))


def get_connection(path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = -524288;
        PRAGMA mmap_size = 536870912;
        PRAGMA temp_store = MEMORY;
        PRAGMA foreign_keys = ON;
        PRAGMA auto_vacuum = INCREMENTAL;
        PRAGMA wal_autocheckpoint = 1000;
    """)


@contextmanager
def connection(path: str | Path = DB_PATH):
    conn = get_connection(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize(path: str | Path = DB_PATH) -> None:
    with connection(path) as conn:
        _create_stocks(conn)
        _create_prices(conn)
        _create_features(conn)
        _create_fundamentals(conn)
        _create_fundamental_metadata(conn)
        _create_sentiment(conn)
        _create_signals(conn)
        _create_macro_features(conn)
        _create_api_usage(conn)
        _create_ticker_cik(conn)
        _create_edgar_filings(conn)
        _create_earnings_events(conn)
        _create_feature_violations(conn)
        _create_recommendations(conn)
        _create_paper_portfolio(conn)
        _create_paper_nav(conn)
        _create_user_holdings(conn)
        _create_user_cash(conn)


def _create_stocks(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stocks (
            ticker      TEXT    NOT NULL,
            name        TEXT,
            sector      TEXT,
            industry    TEXT,
            market_cap  REAL,
            tier        TEXT,
            is_active   INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (ticker)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_stocks_sector
            ON stocks (sector);

        CREATE INDEX IF NOT EXISTS idx_stocks_tier_active
            ON stocks (tier, is_active);
    """)


def _create_prices(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            PRIMARY KEY (ticker, date),
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_prices_date
            ON prices (date);

        CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
            ON prices (ticker, date DESC);
    """)


def _create_features(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS features (
            ticker              TEXT    NOT NULL,
            date                TEXT    NOT NULL,
            sma_5               REAL,
            sma_10              REAL,
            sma_20              REAL,
            sma_50              REAL,
            sma_200             REAL,
            ema_12              REAL,
            ema_26              REAL,
            rsi_14              REAL,
            rsi_28              REAL,
            macd                REAL,
            macd_signal         REAL,
            macd_hist           REAL,
            stoch_k             REAL,
            stoch_d             REAL,
            williams_r          REAL,
            roc_10              REAL,
            atr_14              REAL,
            bb_upper            REAL,
            bb_middle           REAL,
            bb_lower            REAL,
            bb_width            REAL,
            bb_pct_b            REAL,
            obv                 REAL,
            vwap                REAL,
            volume_sma_20       REAL,
            volume_ratio        REAL,
            return_1d           REAL,
            return_5d           REAL,
            return_20d          REAL,
            PRIMARY KEY (ticker, date),
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_features_date
            ON features (date);

        CREATE INDEX IF NOT EXISTS idx_features_ticker_date
            ON features (ticker, date DESC);
    """)


def _create_fundamentals(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker              TEXT    NOT NULL,
            date                TEXT    NOT NULL,
            pe_ratio            REAL,
            pb_ratio            REAL,
            debt_equity         REAL,
            roe                 REAL,
            gross_margin        REAL,
            operating_margin    REAL,
            net_margin          REAL,
            PRIMARY KEY (ticker, date),
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_fundamentals_date
            ON fundamentals (date);

        CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_date
            ON fundamentals (ticker, date DESC);
    """)


def _create_fundamental_metadata(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fundamental_metadata (
            ticker              TEXT    PRIMARY KEY,
            fetch_date          TEXT    NOT NULL,
            pe_ratio            REAL,
            pb_ratio            REAL,
            debt_equity         REAL,
            roe                 REAL,
            gross_margin        REAL,
            operating_margin    REAL,
            net_margin          REAL,
            market_cap          REAL,
            revenue_growth      REAL,
            is_point_in_time    INTEGER DEFAULT 0,
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        );
    """)


def _create_sentiment(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sentiment (
            ticker              TEXT    NOT NULL,
            date                TEXT    NOT NULL,
            news_sentiment      REAL,
            social_sentiment    REAL,
            PRIMARY KEY (ticker, date),
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_sentiment_date
            ON sentiment (date);

        CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date
            ON sentiment (ticker, date DESC);
    """)


def _create_earnings_events(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS earnings_events (
            ticker        TEXT NOT NULL,
            report_date   TEXT NOT NULL,
            eps_actual    REAL,
            eps_estimate  REAL,
            surprise_pct  REAL,
            PRIMARY KEY (ticker, report_date),
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        );

        CREATE INDEX IF NOT EXISTS idx_earnings_ticker_date
            ON earnings_events (ticker, report_date);
    """)


def _create_ticker_cik(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ticker_cik (
            ticker  TEXT PRIMARY KEY,
            cik     TEXT NOT NULL,
            name    TEXT,
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        );
    """)


def _create_edgar_filings(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS edgar_filings (
            ticker            TEXT NOT NULL,
            filed_date        TEXT NOT NULL,
            form_type         TEXT NOT NULL,
            accession_number  TEXT NOT NULL,
            PRIMARY KEY (ticker, accession_number),
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        );

        CREATE INDEX IF NOT EXISTS idx_edgar_ticker_date
            ON edgar_filings (ticker, filed_date);

        CREATE INDEX IF NOT EXISTS idx_edgar_form_date
            ON edgar_filings (form_type, filed_date);
    """)


def _create_api_usage(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_usage (
            service    TEXT    NOT NULL,
            date       TEXT    NOT NULL,
            calls_used INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (service, date)
        );
    """)


def _create_macro_features(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS macro_features (
            date                TEXT    PRIMARY KEY,
            fed_funds_rate      REAL,
            treasury_10y        REAL,
            yield_curve_spread  REAL,
            vix                 REAL,
            vix_sma20           REAL,
            vix_regime          INTEGER,
            sp500_return_20d    REAL,
            sp500_above_sma50   INTEGER,
            unemployment_rate   REAL,
            cpi_yoy             REAL,
            spread_10y2y        REAL
        );

        CREATE INDEX IF NOT EXISTS idx_macro_date
            ON macro_features (date);
    """)


def _create_signals(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            ticker          TEXT    NOT NULL,
            date            TEXT    NOT NULL,
            signal          TEXT,
            confidence      REAL,
            probabilities   TEXT,
            PRIMARY KEY (ticker, date),
            FOREIGN KEY (ticker) REFERENCES stocks (ticker)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_signals_date
            ON signals (date);

        CREATE INDEX IF NOT EXISTS idx_signals_ticker_date
            ON signals (ticker, date DESC);

        CREATE INDEX IF NOT EXISTS idx_signals_signal_date
            ON signals (signal, date DESC);
    """)


def _create_recommendations(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of_date   TEXT    NOT NULL,
            payload_json TEXT    NOT NULL,
            executed     INTEGER NOT NULL DEFAULT 0,
            executed_at  TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_rec_date
            ON recommendations (as_of_date);

        CREATE INDEX IF NOT EXISTS idx_rec_executed
            ON recommendations (executed, as_of_date DESC);
    """)


def _create_feature_violations(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feature_violations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            feature     TEXT    NOT NULL,
            raw_value   REAL    NOT NULL,
            clipped_to  REAL    NOT NULL,
            run_at      TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_fviol_ticker_date
            ON feature_violations (ticker, date);

        CREATE INDEX IF NOT EXISTS idx_fviol_run_at
            ON feature_violations (run_at DESC);
    """)


def _create_paper_portfolio(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_portfolio (
            as_of_date    TEXT NOT NULL,
            ticker        TEXT NOT NULL,
            target_shares REAL NOT NULL,
            target_weight REAL NOT NULL,
            entry_price   REAL NOT NULL,
            PRIMARY KEY (as_of_date, ticker)
        );

        CREATE INDEX IF NOT EXISTS idx_pp_date
            ON paper_portfolio (as_of_date DESC);
    """)


def _create_paper_nav(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_nav (
            date       TEXT PRIMARY KEY,
            nav_usd    REAL NOT NULL,
            spy_close  REAL NOT NULL,
            n_holdings INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_pnav_date
            ON paper_nav (date ASC);
    """)


def _create_user_holdings(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_holdings (
            ticker      TEXT PRIMARY KEY,
            shares      REAL NOT NULL,
            avg_cost    REAL,
            added_at    TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
    """)


def _create_user_cash(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_cash (
            id          INTEGER PRIMARY KEY CHECK(id = 1),
            amount      REAL NOT NULL,
            updated_at  TEXT NOT NULL
        );
    """)


if __name__ == "__main__":
    initialize()
    print(f"Database initialized at {DB_PATH}")
    with connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for t in tables:
            count = conn.execute(
                f"SELECT COUNT(*) FROM pragma_table_info('{t['name']}')"
            ).fetchone()[0]
            print(f"  {t['name']}: {count} columns")
