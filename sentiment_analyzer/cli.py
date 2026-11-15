"""Command-line entry point.

    python -m sentiment_analyzer train   [--data my_labeled.csv]
    python -m sentiment_analyzer collect [--tickers AAPL TSLA] [--sources reddit finviz]
    python -m sentiment_analyzer demo    [--tickers ...]
    python -m sentiment_analyzer analyze [--ticker AAPL]
    python -m sentiment_analyzer classify "Apple smashes earnings expectations"
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys

import pandas as pd

from . import analysis, config, demo, prices, scrapers
from .nlp import SentimentModel, load_or_train, load_training_data
from .storage import Store

log = logging.getLogger("sentiment_analyzer")


def score_articles(df: pd.DataFrame, model: SentimentModel) -> pd.DataFrame:
    if df.empty:
        return df
    text = (df["title"].fillna("") + ". " + df["text"].fillna("").str[:500]).str.strip(". ")
    return pd.concat([df.reset_index(drop=True), model.predict(text.tolist())], axis=1)


def cmd_train(args):
    df = load_training_data(args.data)
    model = SentimentModel()
    m = model.train(df)
    model.save()
    print(f"Trained on {m['n_train']} labeled texts  ->  {config.MODEL_PATH}")
    print(f"  hold-out accuracy {m['holdout_accuracy']:.3f}   macro-F1 {m['holdout_macro_f1']:.3f}"
          f"   5-fold CV accuracy {m['cv5_accuracy']:.3f}")
    for cls, words in model.top_features(10).items():
        print(f"  top {cls:<8}: {', '.join(words)}")


def cmd_collect(args):
    store, model = Store(), load_or_train()
    raw = scrapers.collect(args.tickers, args.sources)
    print(f"Scraped {len(raw)} unique items for {', '.join(args.tickers)}")
    n = store.upsert_articles(score_articles(raw, model))
    print(f"Stored {n} classified items")
    for t in args.tickers:
        p = prices.get_prices(t, args.period)
        store.upsert_prices(p)
        print(f"  {t}: {len(p)} daily price rows")
    if n == 0:
        print("No text was collected - check network access (see README > Troubleshooting).")


def cmd_demo(args):
    store, model = Store(), load_or_train()
    store.clear_demo()
    arts, px = demo.generate(args.tickers, days=args.days)
    store.upsert_articles(score_articles(arts, model), is_demo=True)
    store.upsert_prices(px, is_demo=True)
    print(f"Loaded SYNTHETIC demo data: {len(arts)} articles, {len(px)} price rows "
          f"for {', '.join(args.tickers)}. Remove with: python -m sentiment_analyzer demo --clear")


def cmd_clear_demo():
    Store().clear_demo()
    print("Demo rows removed.")


def cmd_analyze(args):
    store = Store()
    arts, px = store.articles(args.ticker), store.prices(args.ticker)
    if arts.empty or px.empty:
        sys.exit("Database is empty - run `collect` (or `demo`) first.")
    merged = analysis.merge_with_prices(analysis.daily_sentiment(arts), px)
    pd.set_option("display.width", 140)
    print("\n== Sentiment summary (last 7 days vs prior 7) ==")
    print(analysis.summary(merged, arts).round(3).to_string(index=False))
    print("\n== Sentiment(t) vs return(t+lag) ==")
    lc = analysis.lag_correlations(merged)
    print(lc.round(3).to_string(index=False))
    shifts = merged[merged["shift_flag"]][["ticker", "date", "mean_score", "sentiment_z", "n"]]
    print(f"\n== Sentiment shifts (|z| >= 2): {len(shifts)} ==")
    print(shifts.tail(15).round({"mean_score": 2, "sentiment_z": 2}).to_string(index=False))


def cmd_classify(args):
    model = load_or_train()
    out = model.predict(args.text)
    for t, (_, r) in zip(args.text, out.iterrows()):
        print(f"{r['label']:<8} score={r['score']:+.2f}  | {t}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sentiment_analyzer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("train", help="train the sentiment classifier")
    s.add_argument("--data", default=config.TRAINING_CSV, help="CSV with text,label columns")

    s = sub.add_parser("collect", help="scrape news/social posts, classify, fetch prices")
    s.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS, type=str.upper)
    s.add_argument("--sources", nargs="+", choices=list(scrapers.ALL_SCRAPERS))
    s.add_argument("--period", default="6mo", help="price history period for yfinance")

    s = sub.add_parser("demo", help="load clearly-labeled synthetic data (offline)")
    s.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS, type=str.upper)
    s.add_argument("--days", type=int, default=120)
    s.add_argument("--clear", action="store_true", help="remove demo rows instead")

    s = sub.add_parser("analyze", help="print sentiment/price correlation report")
    s.add_argument("--ticker", type=str.upper)

    s = sub.add_parser("classify", help="classify ad-hoc text")
    s.add_argument("text", nargs="+")

    sub.add_parser("dashboard", help="launch the Streamlit dashboard")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "demo":
        cmd_clear_demo() if args.clear else cmd_demo(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "classify":
        cmd_classify(args)
    elif args.cmd == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        str(config.ROOT / "dashboard" / "app.py")], check=False)
