# Movie Revenue Predictor - Film Revenue Forecasting

A Streamlit app that helps film investors estimate box-office revenue from movie details.

## Problem It Solves

Film investors spend hours building spreadsheets to estimate revenue. Movie Revenue Predictor
lets you input movie details and instantly get a data-backed revenue forecast.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Setup

1. Download the [TMDB 5000 Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle
2. Place both CSVs in `data/`
3. (Optional) Add your TMDB API key to `.env` for cast popularity enrichment
4. Launch: `streamlit run app.py`

## Features

- **Home**: Overview with dataset stats and model performance
- **Predictor**: Enter movie details to get a revenue forecast with confidence range
- **Data Explorer**: Visualize the dataset with budget/revenue scatter, genre breakdown, and distribution
- **Model Details**: Regression formula, performance metrics, feature importance, and limitations

## Data & Model

- **Dataset**: TMDB 5000 Movies + Credits (Kaggle)
- **Model**: OLS Multiple Regression (R2 ~0.65)
- **Features**: Budget, cast popularity, audience rating, genre

## Project Structure

```
movie-revenue-predictor/
  app.py              # Main Streamlit app
  data.py             # Load & clean data
  model.py            # Load model, predict revenue, find comparable films
  utils.py            # Formatting helpers
  config.py           # Paths and constants
  requirements.txt
  .env.example
  README.md
  data/
    tmdb_5000_movies.csv
    tmdb_5000_credits.csv
  models/
    movie_revenue_model.joblib
    tmdb_cleaned.csv
  tests/
    test_preprocessing.py
```

## Limitations

- Data is from pre-streaming era (2000-2016)
- Predicts US theatrical revenue only
- Linear model may underfit complex dynamics
- **Not investment advice** - use as a starting point only
