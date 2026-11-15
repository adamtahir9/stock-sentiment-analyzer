"""NLP layer: text cleaning, tokenization, TF-IDF features and sentiment classification.

Model: TF-IDF (word uni+bigrams, plus character n-grams for robustness to
misspellings / slang in social posts) feeding a multinomial logistic regression
that sorts text into negative / neutral / positive.

Per-document sentiment score = P(positive) - P(negative), in [-1, 1].
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from . import config

log = logging.getLogger(__name__)

# Small built-in stop-word list (no NLTK download needed). Negations and
# direction words ("up", "down", "above", "below") are deliberately kept:
# they carry sentiment in finance text.
STOPWORDS = frozenset("""
a an the and or of to in on for at by with from as is are was were be been being this that
these those it its it's he she they them their his her we our you your i me my will would
shall should can could may might must has have had do does did so than then there here
about into over after before while during per via also just only very s amp
""".split())

NEGATIONS = frozenset({"not", "no", "never", "without", "nor", "cannot", "neither",
                       "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't",
                       "didn't", "won't", "can't", "fails", "failed", "fail"})

_URL = re.compile(r"https?://\S+|www\.\S+")
_CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
_MENTION = re.compile(r"@\w+")
_PCT = re.compile(r"[+-]?\d+(?:\.\d+)?\s?%")
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:[bmk]|bn|billion|million|thousand)?\b", re.I)
_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_TOKEN = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]|[.!?;,]")
_CLAUSE_END = frozenset(".!?;,")


def normalize(text: str) -> str:
    """Lower-case and replace URLs, tickers, mentions, numbers with placeholder tokens."""
    text = text or ""
    text = _URL.sub(" ", text)
    text = _CASHTAG.sub(" tickertoken ", text)
    text = _MENTION.sub(" ", text)
    text = _PCT.sub(" pcttoken ", text)
    text = _MONEY.sub(" moneytoken ", text)
    text = _NUM.sub(" numtoken ", text)
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Tokenize, drop stop-words, and mark the scope of negations.

    "Revenue did not beat estimates." -> ['revenue', 'not', 'NEG_beat', 'NEG_estimates']
    """
    tokens, negating = [], False
    for tok in _TOKEN.findall(normalize(text)):
        if tok in _CLAUSE_END:
            negating = False
            continue
        if tok in NEGATIONS:
            negating = True
            tokens.append("not")
            continue
        if tok in STOPWORDS:
            continue
        tokens.append(f"NEG_{tok}" if negating else tok)
    return tokens


# Compact finance sentiment lexicon (in the spirit of Loughran-McDonald). Entries
# are stems: a token matches if it starts with the stem ("surg" -> surge, surged).
POSITIVE_STEMS = ("beat", "surg", "soar", "jump", "rall", "gain", "rose", "rise", "climb",
                  "record", "strong", "upgrad", "outperform", "bullish", "exceed", "top",
                  "boost", "rais", "growth", "grow", "expand", "improv", "profit", "rebound",
                  "approv", "win", "won", "upbeat", "optimis", "impress", "smash", "crush",
                  "high", "upside", "buy", "moon", "accelerat")
NEGATIVE_STEMS = ("miss", "plung", "plummet", "tumbl", "sink", "sank", "slump", "fell", "fall",
                  "drop", "declin", "loss", "lose", "weak", "downgrad", "underperform",
                  "bearish", "cut", "slash", "lower", "layoff", "recall", "lawsuit", "fraud",
                  "investigat", "bankrupt", "warn", "disappoint", "concern", "crater", "slid",
                  "slide", "shrink", "contract", "deteriorat", "gloom", "selloff", "dump",
                  "low", "risk", "delay", "write", "suspend", "short", "puts", "garbage")


def _matches(tok: str, stems: tuple) -> bool:
    return tok.startswith(stems)


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """Dense lexicon counts with negation flipping: [pos, neg, net, has_negation]."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            pos = neg = 0
            toks = tokenize(text)
            for t in toks:
                negated = t.startswith("NEG_")
                w = t[4:] if negated else t
                p, n = _matches(w, POSITIVE_STEMS), _matches(w, NEGATIVE_STEMS)
                if negated:
                    p, n = n, p
                pos += p
                neg += n
            rows.append([pos, neg, (pos - neg) / (1 + pos + neg), float("not" in toks)])
        return sparse.csr_matrix(np.asarray(rows, dtype=float))

    def get_feature_names_out(self, input_features=None):
        return np.array(["lex_pos", "lex_neg", "lex_net", "lex_negation"])


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                  ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("chars", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                                  ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                  max_features=30000)),
        ("lexicon", LexiconFeatures()),
    ])
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    return Pipeline([("tfidf", features), ("clf", clf)])


def load_training_data(path: Path | str = config.TRAINING_CSV) -> pd.DataFrame:
    """CSV with columns text,label (label in negative/neutral/positive, or 0/1/2)."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "sentence" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"sentence": "text"})
    if pd.api.types.is_numeric_dtype(df["label"]):
        # zeroshot/twitter-financial-news-sentiment uses 0=bearish 1=bullish 2=neutral
        df["label"] = df["label"].map({0: "negative", 1: "positive", 2: "neutral"})
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(config.LABELS)].dropna(subset=["text"])
    return df[["text", "label"]].drop_duplicates("text").reset_index(drop=True)


class SentimentModel:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline or build_pipeline()
        self.metrics: dict = {}

    # ---------- training ----------
    def train(self, df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> dict:
        X, y = df["text"].tolist(), df["label"].tolist()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                                  stratify=y, random_state=seed)
        self.pipeline.fit(X_tr, y_tr)
        report = classification_report(y_te, self.pipeline.predict(X_te),
                                       output_dict=True, zero_division=0)
        cv_pred = cross_val_predict(build_pipeline(), X, y,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        cv_acc = float(np.mean(np.array(cv_pred) == np.array(y)))
        self.pipeline.fit(X, y)  # final model uses all labeled data
        self.metrics = {
            "n_train": len(X), "holdout_accuracy": report["accuracy"],
            "holdout_macro_f1": report["macro avg"]["f1-score"], "cv5_accuracy": cv_acc,
            "report": report,
        }
        return self.metrics

    # ---------- inference ----------
    def predict(self, texts: list[str]) -> pd.DataFrame:
        if len(texts) == 0:
            return pd.DataFrame(columns=["label", "score", "p_negative", "p_neutral", "p_positive"])
        proba = self.pipeline.predict_proba(list(texts))
        classes = list(self.pipeline.classes_)
        p = {c: proba[:, classes.index(c)] for c in config.LABELS}
        return pd.DataFrame({
            "label": np.array(classes)[proba.argmax(axis=1)],
            "score": p["positive"] - p["negative"],
            "p_negative": p["negative"], "p_neutral": p["neutral"], "p_positive": p["positive"],
        })

    def top_features(self, n: int = 15) -> dict[str, list[str]]:
        """Most influential word features per class (for explainability)."""
        names = self.pipeline.named_steps["tfidf"].get_feature_names_out()
        coef = self.pipeline.named_steps["clf"].coef_
        out = {}
        for i, cls in enumerate(self.pipeline.classes_):
            order = np.argsort(coef[i])[::-1]
            words = [names[j].split("__", 1)[1] for j in order if names[j].startswith("words__")]
            out[cls] = words[:n]
        return out

    # ---------- persistence ----------
    def save(self, path: Path | str = config.MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metrics": self.metrics}, path)

    @classmethod
    def load(cls, path: Path | str = config.MODEL_PATH) -> "SentimentModel":
        blob = joblib.load(path)
        m = cls(blob["pipeline"])
        m.metrics = blob.get("metrics", {})
        return m


def load_or_train(path: Path | str = config.MODEL_PATH) -> SentimentModel:
    if Path(path).exists():
        return SentimentModel.load(path)
    log.info("No trained model found - training on %s", config.TRAINING_CSV)
    model = SentimentModel()
    model.train(load_training_data())
    model.save(path)
    return model
