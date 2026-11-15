# AI-Powered Stock Sentiment Analyzer

Scrapes financial news and social-media posts about publicly traded companies,
classifies each text as **positive / neutral / negative** with an NLP model, and
lines the daily sentiment up against price data to show whether shifts in
investor sentiment come before, with, or after market moves.

```
 ┌───────────── collect ─────────────┐    ┌──── NLP ────┐    ┌──── analyze ─────┐    ┌── dashboard ──┐
 Yahoo Finance RSS · Google News RSS     tokenize + negation   daily sentiment per     Streamlit + Plotly
 Finviz headlines · Reddit · StockTwits → TF-IDF (word+char) → trading day, EMA,     → KPIs, price vs
 + daily prices (yfinance / Stooq)        + finance lexicon     shift z-scores,         sentiment, lag
                    │                     → logistic regression lagged correlation       correlations
                    └────────────── SQLite (data/sentiment.db) ──────────────┘
```

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m sentiment_analyzer train          # train + evaluate the classifier
```

```bash
python -m sentiment_analyzer collect --tickers AAPL MSFT NVDA TSLA AMZN   # live scrape
```

```bash
python -m sentiment_analyzer analyze         # correlation report in the terminal
```

```bash
python -m sentiment_analyzer dashboard       # opens http://localhost:8501
```

No network access? `python -m sentiment_analyzer demo` loads **synthetic** data
(clearly labeled `[DEMO]`, with a warning banner in the dashboard) so you can try
the whole thing offline. Remove it with `python -m sentiment_analyzer demo --clear`.

Ad-hoc classification:

```bash
python -m sentiment_analyzer classify "Nvidia smashes estimates, raises guidance"
```

## How it works

| Stage | Module | Details |
|---|---|---|
| Data pipeline | `scrapers.py`, `prices.py` | 5 sources behind one `Scraper` interface; retries with backoff, polite delays, a failing site never stops a run; syndicated duplicates removed. Prices from yfinance with Stooq fallback. |
| Pre-processing | `nlp.py` | URLs/mentions stripped; cashtags, %, $ amounts and numbers → placeholder tokens; stop-words removed (negations kept); **negation scope marking** (`did not beat` → `not NEG_beat`) until the next punctuation. |
| Features | `nlp.py` | TF-IDF word uni/bi-grams + TF-IDF character 3–5-grams (robust to slang/typos in social posts) + finance lexicon counts with negation flipping. |
| Classifier | `nlp.py` | Multinomial logistic regression (class-balanced). Score per text = P(positive) − P(negative) ∈ [−1, 1]. |
| Storage | `storage.py` | SQLite; history accumulates across runs, so run `collect` daily (cron) to build a time series. |
| Analysis | `analysis.py` | News after the 4 p.m. ET close or on weekends is assigned to the **next** trading session (avoids look-ahead bias). Daily mean score, positive/negative share, 3-day EMA, 20-day z-score to flag sentiment shifts (abs z ≥ 2), Pearson + Spearman correlation of sentiment(t) with return(t+lag) for lags 0–5. |
| Dashboard | `dashboard/app.py` | KPI row, price and sentiment panels on a shared timeline, lag-correlation chart, sentiment mix by source, cross-ticker "who shifted most" table, latest classified headlines with links. |

## Model performance

Trained on the bundled 180 hand-labeled financial headlines/posts
(`data/training/financial_headlines.csv`):

| Metric | Value |
|---|---|
| 5-fold cross-validated accuracy | ~0.91 |
| Hold-out accuracy / macro-F1 | ~0.89 / 0.89 |

The seed set is small, so treat these numbers as optimistic. For a stronger model,
train on a larger public labeled dataset, for example Financial PhraseBank or
`zeroshot/twitter-financial-news-sentiment` on Hugging Face (its numeric labels
0/1/2 are mapped automatically):

```bash
python -m sentiment_analyzer train --data path/to/sent_train.csv
```

## Interpreting the results

- A positive correlation at **t+1 or later** means sentiment *led* price; at
  "same day" it moved *with* price (news often reacts to the move).
- Correlation is not causation, and with fewer than ~60 trading days of history
  the estimates are noisy. Let `collect` run daily for a few months.
- Demo data is generated with a built-in 1-day lead so you can see the analysis
  recover a known signal. Demo headlines come from the training set, so their
  scores are unrealistically confident.

## Troubleshooting

- **0 items collected**: the sites may be blocked by your network or rate-limiting
  you. Run with `-v` to see per-source counts. Reddit and StockTwits occasionally
  return 403/429 to anonymous clients; the other sources still work.
- **Finviz layout changed**: the parser targets `table#news-table`; adjust `FinvizNews.parse`.
- Respect each site's terms of service and robots.txt; this is for personal research.

## Tests

```bash
python -m pytest -q
```

Covers the tokenizer and negation handling, model training and score direction,
trading-day assignment, recovery of a known lead/lag relationship, and the Finviz parser.

## Project layout

```
sentiment_analyzer/   config, scrapers, prices, nlp, storage, analysis, demo, cli
dashboard/app.py      Streamlit dashboard
data/training/        labeled training data
tests/                pytest suite
```
