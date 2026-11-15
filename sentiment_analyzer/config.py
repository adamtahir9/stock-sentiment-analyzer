"""Central configuration: paths, default tickers, HTTP settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "sentiment.db"
MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
TRAINING_CSV = DATA_DIR / "training" / "financial_headlines.csv"

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

# Company names help the news / social searches find relevant posts.
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "TSLA": "Tesla",
    "AMZN": "Amazon", "GOOGL": "Alphabet", "META": "Meta", "NFLX": "Netflix",
    "AMD": "AMD", "JPM": "JPMorgan",
}

REDDIT_SUBREDDITS = ["stocks", "investing", "wallstreetbets"]

HTTP_TIMEOUT = 15
HTTP_RETRIES = 3
REQUEST_DELAY_SECONDS = 1.0  # polite delay between requests to one host
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) stock-sentiment-analyzer/1.0"

LABELS = ["negative", "neutral", "positive"]
MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16
