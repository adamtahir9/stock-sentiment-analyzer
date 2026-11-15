"""Turn per-article scores into daily sentiment and relate it to price moves."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def assign_trading_day(published: pd.Series) -> pd.Series:
    """Map each timestamp to the trading session it can first affect.

    News published after the 4 p.m. ET close, or on a weekend, is attributed to
    the next business day - otherwise we'd "predict" moves that already happened.
    """
    local = pd.to_datetime(published, utc=True).dt.tz_convert(config.MARKET_TZ)
    day = local.dt.tz_localize(None).dt.normalize()
    after_close = local.dt.hour >= config.MARKET_CLOSE_HOUR
    day = day.where(~after_close, day + pd.Timedelta(days=1))
    bday = pd.offsets.BDay()
    return day.map(lambda d: d if bday.is_on_offset(d) else d + bday)


def daily_sentiment(articles: pd.DataFrame) -> pd.DataFrame:
    if articles.empty:
        return pd.DataFrame(columns=["ticker", "date", "n", "mean_score", "pos_share",
                                     "neg_share", "net_sentiment"])
    a = articles.assign(date=assign_trading_day(articles["published"]))
    g = a.groupby(["ticker", "date"])
    out = g.agg(n=("score", "size"), mean_score=("score", "mean"),
                pos_share=("label", lambda s: (s == "positive").mean()),
                neg_share=("label", lambda s: (s == "negative").mean())).reset_index()
    out["net_sentiment"] = out["pos_share"] - out["neg_share"]
    return out


def merge_with_prices(daily: pd.DataFrame, prices: pd.DataFrame, ema_span: int = 3,
                      z_window: int = 20) -> pd.DataFrame:
    """One row per trading day per ticker: price, returns, sentiment and its smoothing."""
    frames = []
    for ticker, p in prices.groupby("ticker"):
        p = p.sort_values("date")[["ticker", "date", "close", "volume"]].copy()
        p["return"] = p["close"].pct_change()
        p["next_return"] = p["return"].shift(-1)
        s = daily[daily["ticker"] == ticker].drop(columns="ticker")
        m = p.merge(s, on="date", how="left")
        m["n"] = m["n"].fillna(0).astype(int)
        m["sentiment_ema"] = m["mean_score"].ffill().ewm(span=ema_span, adjust=False).mean()
        roll = m["mean_score"].rolling(z_window, min_periods=5)
        m["sentiment_z"] = (m["mean_score"] - roll.mean().shift(1)) / roll.std().shift(1)
        m["shift_flag"] = m["sentiment_z"].abs() >= 2
        frames.append(m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def lag_correlations(merged: pd.DataFrame, max_lag: int = 5, min_obs: int = 10) -> pd.DataFrame:
    """Correlation of day-t sentiment with the return on day t+lag (lag 0 = same day).

    A positive coefficient at lag >= 1 means sentiment *led* the price.
    """
    rows = []
    for ticker, m in merged.groupby("ticker"):
        m = m.sort_values("date")
        for lag in range(0, max_lag + 1):
            x = m["mean_score"]
            y = m["return"].shift(-lag)
            ok = x.notna() & y.notna()
            n = int(ok.sum())
            if n < min_obs:
                rows.append({"ticker": ticker, "lag": lag, "n": n, "pearson_r": np.nan,
                             "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan})
                continue
            pr = stats.pearsonr(x[ok], y[ok])
            sr = stats.spearmanr(x[ok], y[ok])
            rows.append({"ticker": ticker, "lag": lag, "n": n,
                         "pearson_r": pr.statistic, "pearson_p": pr.pvalue,
                         "spearman_rho": sr.statistic, "spearman_p": sr.pvalue})
    return pd.DataFrame(rows)


def summary(merged: pd.DataFrame, articles: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Per-ticker headline numbers for the dashboard leaderboard."""
    rows = []
    for ticker, m in merged.groupby("ticker"):
        m = m.sort_values("date")
        a = articles[articles["ticker"] == ticker]
        end = m["date"].max()
        cur = m[m["date"] > end - pd.Timedelta(days=window_days)]
        prev = m[(m["date"] <= end - pd.Timedelta(days=window_days)) &
                 (m["date"] > end - pd.Timedelta(days=2 * window_days))]

        def wmean(f):  # article-weighted mean sentiment over a window
            f = f[f["n"] > 0]
            return float(np.average(f["mean_score"], weights=f["n"])) if len(f) else np.nan

        rows.append({
            "ticker": ticker, "articles": len(a),
            "sentiment_now": wmean(cur), "sentiment_prev": wmean(prev),
            "pos_share": (a["label"] == "positive").mean() if len(a) else np.nan,
            "neg_share": (a["label"] == "negative").mean() if len(a) else np.nan,
            "last_close": m["close"].iloc[-1],
            "return_window": m["close"].iloc[-1] / cur["close"].iloc[0] - 1 if len(cur) else np.nan,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sentiment_change"] = df["sentiment_now"] - df["sentiment_prev"]
    return df
