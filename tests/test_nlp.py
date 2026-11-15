from sentiment_analyzer.nlp import SentimentModel, load_training_data, normalize, tokenize


def test_normalize_placeholders():
    t = normalize("$AAPL up 5.2% to $3.1 billion, see https://x.com")
    assert "tickertoken" in t and "pcttoken" in t and "moneytoken" in t and "http" not in t


def test_negation_scope_ends_at_punctuation():
    assert tokenize("Revenue did not beat estimates. Shares rose") == \
        ["revenue", "not", "NEG_beat", "NEG_estimates", "shares", "rose"]


def test_model_trains_and_scores_direction():
    model = SentimentModel()
    metrics = model.train(load_training_data())
    assert metrics["cv5_accuracy"] > 0.7
    out = model.predict(["Shares soar after record profit beats estimates",
                         "Stock plunges as company misses forecasts and cuts guidance"])
    assert list(out["label"]) == ["positive", "negative"]
    assert out["score"].iloc[0] > 0 > out["score"].iloc[1]
    assert out["score"].between(-1, 1).all()
