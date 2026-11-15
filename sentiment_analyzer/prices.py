"""Security price data (daily OHLCV)."""
from __future__ import annotations

import io
import logging

import pandas as pd

from .scrapers import fetch, make_session

log = logging.getLogger(__name__)

COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def _from_yfinance(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
    df["ticker"] = ticker
    return df[COLUMNS]


def _from_stooq(ticker: str) -> pd.DataFrame:
    r = fetch(make_session(), f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d")
    if r is None or not r.text.startswith("Date"):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(io.StringIO(r.text)).rename(columns=str.lower)
    df["ticker"] = ticker
    return df[COLUMNS]


def get_prices(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Daily prices from Yahoo Finance (yfinance), falling back to Stooq."""
    for loader in (lambda: _from_yfinance(ticker, period), lambda: _from_stooq(ticker)):
        try:
            df = loader()
        except Exception as exc:
            log.warning("price source failed for %s: %s", ticker, exc)
            continue
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return pd.DataFrame(columns=COLUMNS)
