import numpy as np
import pandas as pd

from sentiment_analyzer import analysis


def test_after_close_and_weekend_news_maps_to_next_session():
    ts = pd.Series(pd.to_datetime([
        "2026-11-13 15:00", "2026-11-13 16:30",  # Fri before / after close (ET)
        "2026-11-14 10:00",                      # Saturday
    ]).tz_localize("America/New_York"))
    days = analysis.assign_trading_day(ts).dt.strftime("%Y-%m-%d").tolist()
    assert days == ["2026-11-13", "2026-11-16", "2026-11-16"]


def _synthetic(lead: int):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2026-01-01", periods=200)
    sent = rng.normal(0, 0.5, len(dates))
    ret = np.roll(sent, lead) * 0.02 + rng.normal(0, 0.002, len(dates))
    close = 100 * np.cumprod(1 + ret)
    prices = pd.DataFrame({"ticker": "XYZ", "date": dates, "close": close, "volume": 1.0})
    daily = pd.DataFrame({"ticker": "XYZ", "date": dates, "n": 5, "mean_score": sent,
                          "pos_share": 0.3, "neg_share": 0.3, "net_sentiment": 0.0})
    return analysis.merge_with_prices(daily, prices)


def test_lag_correlation_recovers_known_lead():
    lc = analysis.lag_correlations(_synthetic(lead=1)).set_index("lag")
    assert lc["pearson_r"].idxmax() == 1
    assert lc.loc[1, "pearson_r"] > 0.9 and lc.loc[1, "pearson_p"] < 1e-6


def test_daily_sentiment_aggregates():
    a = pd.DataFrame({
        "ticker": ["A"] * 3,
        "published": pd.to_datetime(["2026-11-12 14:00"] * 3).tz_localize("UTC"),
        "label": ["positive", "positive", "negative"], "score": [0.8, 0.6, -0.5]})
    d = analysis.daily_sentiment(a).iloc[0]
    assert d["n"] == 3 and abs(d["net_sentiment"] - 1 / 3) < 1e-9
