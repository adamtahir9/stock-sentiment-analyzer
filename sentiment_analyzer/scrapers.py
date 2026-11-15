"""Data-collection layer: financial news sites and social-media posts.

Every scraper returns a list of plain dicts with the same schema so the rest of
the pipeline does not care where a piece of text came from:

    {source, ticker, published (tz-aware UTC Timestamp), title, text, url}
"""
from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config

log = logging.getLogger(__name__)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return s


def fetch(session: requests.Session, url: str, **kwargs) -> requests.Response | None:
    """GET with retries and exponential backoff. Returns None on failure."""
    for attempt in range(config.HTTP_RETRIES):
        try:
            r = session.get(url, timeout=config.HTTP_TIMEOUT, **kwargs)
            if r.status_code == 429:  # rate limited
                time.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            log.debug("fetch %s failed (attempt %d): %s", url, attempt + 1, exc)
            time.sleep(2 ** attempt)
    log.warning("Giving up on %s", url)
    return None


def article_id(source: str, url: str, title: str) -> str:
    return hashlib.sha1(f"{source}|{url or title}".encode()).hexdigest()[:16]


def clean_html(text: str) -> str:
    return BeautifulSoup(text or "", "html.parser").get_text(" ", strip=True)


def make_record(source, ticker, published, title, text="", url="") -> dict:
    ts = pd.Timestamp(published)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    title, text = clean_html(title), clean_html(text)
    return {
        "id": article_id(source, url, title),
        "source": source,
        "ticker": ticker.upper(),
        "published": ts,
        "title": title,
        "text": text,
        "url": url,
    }


class Scraper(ABC):
    name = "base"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or make_session()

    @abstractmethod
    def scrape(self, ticker: str) -> list[dict]:
        ...


class _RSSScraper(Scraper):
    url_template = ""

    def feed_url(self, ticker: str) -> str:
        return self.url_template.format(ticker=ticker, query=quote_plus(
            f"{config.COMPANY_NAMES.get(ticker, ticker)} {ticker} stock"))

    def scrape(self, ticker: str) -> list[dict]:
        r = fetch(self.session, self.feed_url(ticker))
        if r is None:
            return []
        feed = feedparser.parse(r.content)
        out = []
        for e in feed.entries:
            parsed = e.get("published_parsed") or e.get("updated_parsed")
            if not parsed:
                continue
            published = datetime(*parsed[:6], tzinfo=timezone.utc)
            out.append(make_record(self.name, ticker, published, e.get("title", ""),
                                   e.get("summary", ""), e.get("link", "")))
        return out


class YahooFinanceRSS(_RSSScraper):
    name = "yahoo_finance"
    url_template = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


class GoogleNewsRSS(_RSSScraper):
    name = "google_news"
    url_template = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


class FinvizNews(Scraper):
    """Parses the headline table on a Finviz quote page."""
    name = "finviz"

    def scrape(self, ticker: str) -> list[dict]:
        r = fetch(self.session, f"https://finviz.com/quote.ashx?t={ticker}&p=d")
        return self.parse(r.text, ticker) if r is not None else []

    @staticmethod
    def parse(html: str, ticker: str, today: datetime | None = None) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find(id="news-table")
        if table is None:
            return []
        today = today or datetime.now(tz=timezone.utc)
        out, current_date = [], None
        for row in table.find_all("tr"):
            link = row.find("a")
            cell = row.find("td")
            if link is None or cell is None:
                continue
            stamp = cell.get_text(" ", strip=True).split()
            # Rows look like "Nov-14-26 08:00AM" or "Today 08:00AM" or just "08:00AM"
            if len(stamp) == 2:
                current_date = today.strftime("%b-%d-%y") if stamp[0] == "Today" else stamp[0]
                time_str = stamp[1]
            elif stamp:
                time_str = stamp[0]
            else:
                continue
            if current_date is None:
                continue
            try:
                local = pd.to_datetime(f"{current_date} {time_str}", format="%b-%d-%y %I:%M%p")
            except ValueError:
                continue
            published = local.tz_localize(config.MARKET_TZ)  # Finviz shows US/Eastern
            out.append(make_record("finviz", ticker, published, link.get_text(strip=True),
                                   "", link.get("href", "")))
        return out


class RedditSearch(Scraper):
    """Recent posts mentioning the ticker via Reddit's public JSON search."""
    name = "reddit"

    def __init__(self, session=None, subreddits=None, limit: int = 50):
        super().__init__(session)
        self.subreddits = subreddits or config.REDDIT_SUBREDDITS
        self.limit = limit

    def scrape(self, ticker: str) -> list[dict]:
        out = []
        for sub in self.subreddits:
            url = (f"https://www.reddit.com/r/{sub}/search.json?q={quote_plus(ticker)}"
                   f"&restrict_sr=1&sort=new&t=month&limit={self.limit}")
            r = fetch(self.session, url)
            time.sleep(config.REQUEST_DELAY_SECONDS)
            if r is None:
                continue
            for child in r.json().get("data", {}).get("children", []):
                d = child.get("data", {})
                out.append(make_record(
                    f"reddit/{sub}", ticker,
                    datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc),
                    d.get("title", ""), (d.get("selftext") or "")[:2000],
                    "https://www.reddit.com" + d.get("permalink", "")))
        return out


class StockTwitsStream(Scraper):
    """Latest StockTwits messages for a symbol (public stream endpoint)."""
    name = "stocktwits"

    def scrape(self, ticker: str) -> list[dict]:
        r = fetch(self.session, f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json")
        if r is None:
            return []
        out = []
        for m in r.json().get("messages", []):
            out.append(make_record("stocktwits", ticker, m.get("created_at"),
                                   m.get("body", ""), "",
                                   f"https://stocktwits.com/message/{m.get('id')}"))
        return out


ALL_SCRAPERS = {
    "yahoo_finance": YahooFinanceRSS,
    "google_news": GoogleNewsRSS,
    "finviz": FinvizNews,
    "reddit": RedditSearch,
    "stocktwits": StockTwitsStream,
}


def collect(tickers: list[str], sources: list[str] | None = None) -> pd.DataFrame:
    """Run every selected scraper for every ticker and return a de-duplicated frame."""
    session = make_session()
    scrapers = [ALL_SCRAPERS[s](session) for s in (sources or ALL_SCRAPERS)]
    rows = []
    for ticker in tickers:
        for scraper in scrapers:
            try:
                got = scraper.scrape(ticker)
            except Exception as exc:  # one broken site must not kill the run
                log.warning("%s failed for %s: %s", scraper.name, ticker, exc)
                got = []
            log.info("%-14s %-6s %4d items", scraper.name, ticker, len(got))
            rows.extend(got)
            time.sleep(config.REQUEST_DELAY_SECONDS)
    df = pd.DataFrame(rows, columns=["id", "source", "ticker", "published", "title", "text", "url"])
    if df.empty:
        return df
    # The same headline is often syndicated across sites: drop exact title repeats per ticker.
    df["_norm"] = df["title"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df = df.drop_duplicates("id").drop_duplicates(["ticker", "_norm"]).drop(columns="_norm")
    return df.reset_index(drop=True)
