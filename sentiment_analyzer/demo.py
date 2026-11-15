"""Synthetic demo data so the dashboard can be explored without network access.

Everything produced here is FAKE and is flagged is_demo=1 in the database; the
dashboard shows a banner whenever demo rows are on screen. The generator builds
a latent daily "news mood", writes headlines whose tone follows that mood, and
makes next-day returns partially depend on it - so the analysis has a known
signal to recover, which is also handy for testing the pipeline end to end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .nlp import load_training_data
from .scrapers import article_id

DEMO_SOURCES = ["yahoo_finance", "google_news", "finviz", "reddit/stocks", "stocktwits"]


def generate(tickers: list[str], days: int = 120, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pool = load_training_data()
    by_label = {lab: pool[pool["label"] == lab]["text"].tolist() for lab in config.LABELS}
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.offsets.BDay(1), periods=days)
    prices, articles = [], []
    for k, ticker in enumerate(tickers):
        mood = np.zeros(days)
        for t in range(1, days):  # AR(1) mood with occasional news shocks
            shock = rng.normal(0, 1.4) if rng.random() < 0.05 else 0.0
            mood[t] = 0.5 * mood[t - 1] + rng.normal(0, 0.45) + shock
        mood = np.tanh(mood)
        ret = 0.0004 + 0.014 * np.roll(mood, 1) + rng.normal(0, 0.015, days)  # mood leads by 1 day
        ret[0] = 0.0
        close = (80 + 60 * k) * np.exp(np.cumsum(ret))
        prices.append(pd.DataFrame({
            "ticker": ticker, "date": dates, "open": close * (1 + rng.normal(0, .003, days)),
            "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": rng.integers(2e7, 9e7, days).astype(float)}))
        name = config.COMPANY_NAMES.get(ticker, ticker)
        for t, d in enumerate(dates):
            n = rng.poisson(6 + 10 * abs(mood[t]))
            p_pos = 0.25 + 0.45 * max(mood[t], 0)
            p_neg = 0.25 + 0.45 * max(-mood[t], 0)
            labels = rng.choice(config.LABELS, size=n, p=[p_neg, 1 - p_pos - p_neg, p_pos])
            for i, lab in enumerate(labels):
                text = (rng.choice(by_label[lab]).replace("The company", name).replace("the company", name)
                        .replace("Company", name).replace("company", name))
                ts = (d + pd.Timedelta(hours=int(rng.integers(6, 15)), minutes=int(rng.integers(60))))
                src = DEMO_SOURCES[rng.integers(len(DEMO_SOURCES))]
                articles.append({"id": article_id("demo", f"{ticker}{d}{i}", text), "source": src,
                                 "ticker": ticker,
                                 "published": ts.tz_localize(config.MARKET_TZ).tz_convert("UTC"),
                                 "title": f"[DEMO] {text}", "text": "", "url": ""})
    return pd.DataFrame(articles), pd.concat(prices, ignore_index=True)
