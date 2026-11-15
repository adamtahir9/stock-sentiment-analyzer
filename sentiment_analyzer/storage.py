"""SQLite persistence so sentiment history accumulates across pipeline runs."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY, source TEXT, ticker TEXT, published TEXT,
    title TEXT, text TEXT, url TEXT,
    label TEXT, score REAL, p_negative REAL, p_neutral REAL, p_positive REAL,
    is_demo INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_articles_ticker_pub ON articles(ticker, published);
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
    is_demo INTEGER DEFAULT 0,
    PRIMARY KEY (ticker, date)
);
"""


class Store:
    def __init__(self, path: Path | str = config.DB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    def upsert_articles(self, df: pd.DataFrame, is_demo: bool = False) -> int:
        if df.empty:
            return 0
        d = df.copy()
        d["published"] = pd.to_datetime(d["published"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        d["is_demo"] = int(is_demo)
        cols = ["id", "source", "ticker", "published", "title", "text", "url", "label",
                "score", "p_negative", "p_neutral", "p_positive", "is_demo"]
        with self._conn() as c:
            c.executemany(f"INSERT OR REPLACE INTO articles ({','.join(cols)}) "
                          f"VALUES ({','.join('?' * len(cols))})",
                          d[cols].itertuples(index=False, name=None))
        return len(d)

    def upsert_prices(self, df: pd.DataFrame, is_demo: bool = False) -> int:
        if df.empty:
            return 0
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        d["is_demo"] = int(is_demo)
        cols = ["ticker", "date", "open", "high", "low", "close", "volume", "is_demo"]
        with self._conn() as c:
            c.executemany(f"INSERT OR REPLACE INTO prices ({','.join(cols)}) "
                          f"VALUES ({','.join('?' * len(cols))})",
                          d[cols].itertuples(index=False, name=None))
        return len(d)

    def articles(self, ticker: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM articles", ()
        if ticker:
            q, args = q + " WHERE ticker = ?", (ticker,)
        with self._conn() as c:
            df = pd.read_sql_query(q, c, params=args)
        df["published"] = pd.to_datetime(df["published"], utc=True)
        return df

    def prices(self, ticker: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM prices", ()
        if ticker:
            q, args = q + " WHERE ticker = ?", (ticker,)
        with self._conn() as c:
            df = pd.read_sql_query(q + " ORDER BY date", c, params=args)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def tickers(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute("SELECT DISTINCT ticker FROM articles UNION "
                             "SELECT DISTINCT ticker FROM prices ORDER BY 1").fetchall()
        return [r[0] for r in rows]

    def clear_demo(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM articles WHERE is_demo = 1")
            c.execute("DELETE FROM prices WHERE is_demo = 1")
