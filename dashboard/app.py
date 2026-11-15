"""Streamlit dashboard: investor sentiment vs. price.

Run:  streamlit run dashboard/app.py   (or: python -m sentiment_analyzer dashboard)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sentiment_analyzer import analysis
from sentiment_analyzer.storage import Store

# Diverging pair for polarity (blue = positive, red = negative, gray = neutral);
# validated colorblind-safe palette, never color-alone (labels/legend everywhere).
POS, NEG, NEU = "#2a78d6", "#e34948", "#8a8984"
LINE, MUTED = "#2a78d6", "#52514e"
LABEL_COLORS = {"positive": POS, "negative": NEG, "neutral": NEU}

st.set_page_config(page_title="Stock Sentiment Analyzer", page_icon="📈", layout="wide")


@st.cache_data(ttl=300)
def load():
    s = Store()
    return s.articles(), s.prices()


articles, prices = load()
st.title("AI-Powered Stock Sentiment Analyzer")

if articles.empty or prices.empty:
    st.info("No data yet. Run `python -m sentiment_analyzer collect` to scrape live data, "
            "or `python -m sentiment_analyzer demo` for offline synthetic data.")
    st.stop()

# ---------------- filters (one row, above the charts) ----------------
tickers = sorted(set(articles["ticker"]) & set(prices["ticker"]))
c1, c2, c3 = st.columns([1, 2, 2])
ticker = c1.selectbox("Ticker", tickers)
sources = sorted(articles["source"].unique())
picked = c2.multiselect("Sources", sources, default=sources)
min_d, max_d = prices["date"].min().date(), prices["date"].max().date()
rng = c3.date_input("Date range", (max(min_d, max_d - pd.Timedelta(days=90)), max_d),
                    min_value=min_d, max_value=max_d)
start, end = (pd.Timestamp(rng[0]), pd.Timestamp(rng[1])) if len(rng) == 2 else (pd.Timestamp(min_d), pd.Timestamp(max_d))

arts = articles[articles["source"].isin(picked)]
if arts["is_demo"].any() or prices["is_demo"].any():
    st.warning("Showing SYNTHETIC demo data (generated offline, not real market data). "
               "Run `collect` with network access for live data; `demo --clear` removes it.", icon="⚠️")

daily = analysis.daily_sentiment(arts)
merged_all = analysis.merge_with_prices(daily, prices)
merged = merged_all[(merged_all["date"] >= start) & (merged_all["date"] <= end)]
m = merged[merged["ticker"] == ticker].sort_values("date")
ta = arts[(arts["ticker"] == ticker)].assign(day=analysis.assign_trading_day(arts.loc[arts["ticker"] == ticker, "published"]))
ta = ta[(ta["day"] >= start) & (ta["day"] <= end)]

if m.empty:
    st.info("No price data in this range.")
    st.stop()

# ---------------- KPI row ----------------
summ = analysis.summary(merged, arts[arts["published"].dt.tz_localize(None) <= end + pd.Timedelta(days=1)])
row = summ[summ["ticker"] == ticker].iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Sentiment score (7d)", f"{row.sentiment_now:+.2f}",
          f"{row.sentiment_change:+.2f} vs prior 7d" if pd.notna(row.sentiment_change) else None,
          help="Article-weighted mean of P(positive) − P(negative), range −1 … +1")
k2.metric("Articles / posts", f"{len(ta):,}")
k3.metric("Positive · Negative share",
          f"{(ta['label'] == 'positive').mean():.0%} · {(ta['label'] == 'negative').mean():.0%}")
k4.metric("Last close", f"${m['close'].iloc[-1]:,.2f}",
          f"{m['close'].iloc[-1] / m['close'].iloc[0] - 1:+.1%} in range")

# ---------------- price + sentiment: two stacked panels, shared x, one y-axis each ----------------
st.subheader(f"{ticker}: price and daily sentiment")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    row_heights=[0.55, 0.45],
                    subplot_titles=("Close price (USD)", "Daily sentiment score (bars) · 3-day EMA (line)"))
fig.add_trace(go.Scatter(x=m["date"], y=m["close"], name="Close", line=dict(color=LINE, width=2),
                         hovertemplate="%{x|%b %d}<br>Close $%{y:,.2f}<extra></extra>"), 1, 1)
bar_colors = [POS if v > 0 else NEG for v in m["mean_score"].fillna(0)]
fig.add_trace(go.Bar(x=m["date"], y=m["mean_score"], name="Daily sentiment (blue +, red −)", marker_color=bar_colors,
                     customdata=m["n"], hovertemplate="%{x|%b %d}<br>Score %{y:+.2f}<br>%{customdata} items<extra></extra>"), 2, 1)
fig.add_trace(go.Scatter(x=m["date"], y=m["sentiment_ema"], name="3-day EMA",
                         line=dict(color=MUTED, width=2),
                         hovertemplate="EMA %{y:+.2f}<extra></extra>"), 2, 1)
shifts = m[m["shift_flag"]]
fig.add_trace(go.Scatter(x=shifts["date"], y=shifts["mean_score"], mode="markers", name="Sentiment shift (|z|≥2)",
                         marker=dict(symbol="diamond", size=10, color="#0b0b0b", line=dict(color="white", width=2)),
                         hovertemplate="%{x|%b %d}<br>Shift z-score: %{customdata:+.1f}<extra></extra>",
                         customdata=shifts["sentiment_z"]), 2, 1)
fig.update_layout(height=560, hovermode="x unified", bargap=0.25, margin=dict(t=40, b=10),
                  legend=dict(orientation="h", y=-0.08))
fig.update_yaxes(range=[-1, 1], row=2, col=1, zeroline=True, zerolinecolor="#c3c2b7")
st.plotly_chart(fig, width="stretch")

# ---------------- correlation + mix ----------------
left, right = st.columns(2)
with left:
    st.subheader("Does sentiment lead the price?")
    lc = analysis.lag_correlations(m, max_lag=5)
    if lc["pearson_r"].notna().any():
        sig = lc["pearson_p"] < 0.05
        fig2 = go.Figure(go.Bar(
            x=[f"t+{l}" if l else "same day" for l in lc["lag"]], y=lc["pearson_r"],
            marker_color=[POS if r > 0 else NEG for r in lc["pearson_r"].fillna(0)],
            text=[f"{r:+.2f}{' *' if s else ''}" for r, s in zip(lc["pearson_r"], sig)],
            textposition="outside", customdata=lc[["pearson_p", "n"]],
            hovertemplate="%{x}<br>r = %{y:+.3f}<br>p = %{customdata[0]:.3f}<br>n = %{customdata[1]}<extra></extra>"))
        fig2.update_layout(height=340, yaxis_title="Pearson r (sentiment vs return)",
                           xaxis_title="Return measured on day", margin=dict(t=10))
        st.plotly_chart(fig2, width="stretch")
        st.caption("\\* p < 0.05. Correlation is not causation; small samples are noisy.")
    else:
        st.info("Need at least 10 days with both sentiment and price data. Keep running `collect` daily.")
with right:
    st.subheader("Sentiment mix by source")
    mix = (ta.groupby(["source", "label"]).size().unstack(fill_value=0)
             .reindex(columns=["negative", "neutral", "positive"], fill_value=0))
    share = mix.div(mix.sum(axis=1), axis=0)
    fig3 = go.Figure()
    for lab in ["negative", "neutral", "positive"]:
        fig3.add_trace(go.Bar(y=share.index, x=share[lab], name=lab, orientation="h",
                              marker=dict(color=LABEL_COLORS[lab], line=dict(color="white", width=2)),
                              customdata=mix[lab],
                              hovertemplate=f"%{{y}}<br>{lab}: %{{x:.0%}} (%{{customdata}})<extra></extra>"))
    fig3.update_layout(barmode="stack", height=340, xaxis_tickformat=".0%", margin=dict(t=10),
                       legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig3, width="stretch")

# ---------------- cross-ticker leaderboard ----------------
st.subheader("All tickers: sentiment shifts")
board = summ[["ticker", "sentiment_now", "sentiment_prev", "sentiment_change", "articles",
              "pos_share", "neg_share", "return_window"]].sort_values("sentiment_change", ascending=False)
st.dataframe(board, hide_index=True, width="stretch", column_config={
    "sentiment_now": st.column_config.NumberColumn("Sentiment (7d)", format="%+.2f"),
    "sentiment_prev": st.column_config.NumberColumn("Prior 7d", format="%+.2f"),
    "sentiment_change": st.column_config.NumberColumn("Shift", format="%+.2f"),
    "pos_share": st.column_config.NumberColumn("Positive share", format="percent"),
    "neg_share": st.column_config.NumberColumn("Negative share", format="percent"),
    "return_window": st.column_config.NumberColumn("Price chg (7d)", format="percent"),
})

# ---------------- latest classified text ----------------
st.subheader("Latest classified headlines & posts")
latest = ta.sort_values("published", ascending=False).head(50)
st.dataframe(latest[["published", "source", "label", "score", "title", "url"]], hide_index=True,
             width="stretch", column_config={
                 "score": st.column_config.NumberColumn("Score", format="%+.2f"),
                 "url": st.column_config.LinkColumn("Link", display_text="open"),
             })
