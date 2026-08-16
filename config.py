import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent

DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MOVIES_CSV  = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = DATA_DIR / "tmdb_5000_credits.csv"
CLEAN_CSV   = OUTPUT_DIR / "tmdb_cleaned.csv"
MODEL_PATH  = MODELS_DIR / "movie_revenue_model.joblib"

TMDB_API_KEY  = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

TOP_N_CAST   = 3
MIN_BUDGET   = 1_000
MIN_REVENUE  = 1_000
RANDOM_STATE = 42

KEEP_GENRES = [
    "Drama", "Comedy", "Thriller", "Action",
    "Romance", "Adventure", "Crime", "Horror",
    "Animation", "Science Fiction",
]