import ast
import time
import logging

import numpy as np
import pandas as pd
import requests

from config import (
    MOVIES_CSV, CREDITS_CSV, CLEAN_CSV,
    TMDB_API_KEY, TMDB_BASE_URL, TOP_N_CAST,
    MIN_BUDGET, MIN_REVENUE, KEEP_GENRES,
)

logger = logging.getLogger(__name__)


def load_raw() -> pd.DataFrame:
    if not MOVIES_CSV.exists():
        raise FileNotFoundError(
            f"Movies file not found: {MOVIES_CSV}\n"
            "Download from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata "
            "and place it in the data/ folder."
        )
    if not CREDITS_CSV.exists():
        raise FileNotFoundError(
            f"Credits file not found: {CREDITS_CSV}\n"
            "Download from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata "
            "and place it in the data/ folder."
        )
    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id", how="left", suffixes=("", "_credits"))
    return df


def _safe_parse(value) -> list:
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def extract_genre_names(genres_str) -> list[str]:
    parsed = _safe_parse(genres_str)
    return [g["name"] for g in parsed if "name" in g]


def search_person(name: str, api_key: str | None = None) -> list[dict]:
    """Search TMDB for people by name."""
    key = api_key or TMDB_API_KEY
    if not key:
        return []
    url = f"{TMDB_BASE_URL}/search/person"
    try:
        resp = requests.get(url, params={"api_key": key, "query": name}, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            return [
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "popularity": r.get("popularity", 0.0),
                    "known_for": r.get("known_for_department", ""),
                }
                for r in results[:10]
                if r.get("name")
            ]
    except requests.RequestException:
        pass
    return []


def extract_cast_names(cast_str, top_n: int = TOP_N_CAST) -> list[str]:
    parsed = _safe_parse(cast_str)
    sorted_cast = sorted(parsed, key=lambda x: x.get("order", 999))
    return [c["name"] for c in sorted_cast[:top_n] if "name" in c]


def _get_person_popularity(person_id: int, api_key: str) -> float | None:
    url = f"{TMDB_BASE_URL}/person/{person_id}"
    try:
        resp = requests.get(url, params={"api_key": api_key}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("popularity")
        if resp.status_code == 429:
            time.sleep(10)
    except requests.RequestException:
        pass
    return None


def fetch_cast_popularity(
    df: pd.DataFrame,
    api_key: str | None = None,
    top_n: int = TOP_N_CAST,
    delay: float = 0.25,
) -> pd.Series:
    key = api_key or TMDB_API_KEY
    if not key:
        raise ValueError(
            "No TMDB API key found. Set TMDB_API_KEY in your .env file."
        )
    scores = []
    for cast_str in df["cast"]:
        person_ids = [
            c["id"] for c in sorted(
                _safe_parse(cast_str), key=lambda x: x.get("order", 999)
            )[:top_n] if "id" in c
        ]
        pops = []
        for pid in person_ids:
            pop = _get_person_popularity(pid, key)
            if pop is not None:
                pops.append(pop)
            time.sleep(delay)
        scores.append(float(np.mean(pops)) if pops else np.nan)
    return pd.Series(scores, index=df.index, name="cast_popularity_score")


def drop_zero_financials(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    return df


def drop_low_financials(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        (df["budget"] >= MIN_BUDGET) & (df["revenue"] >= MIN_REVENUE)
    ].copy()
    return df


def drop_missing_cast_popularity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["cast_popularity_score"]).copy()
    return df


def impute_runtime(df: pd.DataFrame) -> pd.DataFrame:
    n_missing = df["runtime"].isna().sum()
    if n_missing > 0:
        median_rt = df["runtime"].median()
        df["runtime"] = df["runtime"].fillna(median_rt)
    return df


def drop_missing_release_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["release_date"]).copy()
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if "release_year" not in df.columns:
        df["release_date"] = pd.to_datetime(
            df["release_date"], errors="coerce"
        )
        df["_year_tmp"] = df["release_date"].dt.year
        key = ["title", "_year_tmp"]
    else:
        key = ["title", "release_year"]
    df = df.drop_duplicates(subset=key).copy()
    df = df.drop(columns=["_year_tmp"], errors="ignore")
    return df


def add_cast_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    if "cast_popularity_score" not in df.columns:
        df = df.copy()
        df["cast_popularity_score"] = fetch_cast_popularity(df)
    return df


def add_roi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["roi"] = (df["revenue"] - df["budget"]) / df["budget"]
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_revenue"]  = np.log1p(df["revenue"])
    df["log_budget"]   = np.log1p(df["budget"])
    df["log_cast_pop"] = np.log1p(df["cast_popularity_score"])
    return df


def add_primary_genre(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["genres_list"]  = df["genres"].apply(extract_genre_names)
    df["primary_genre"] = df["genres_list"].apply(
        lambda gl: gl[0] if gl else "Other"
    )
    df["primary_genre"] = df["primary_genre"].where(
        df["primary_genre"].isin(KEEP_GENRES), other="Other"
    )
    return df


def run_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = (
        df_raw.pipe(drop_zero_financials)
        .pipe(drop_missing_release_date)
        .pipe(impute_runtime)
        .pipe(drop_duplicates)
        .pipe(drop_low_financials)
        .pipe(add_cast_popularity_score)
        .pipe(add_roi)
        .pipe(add_log_transforms)
        .pipe(add_primary_genre)
        .pipe(drop_missing_cast_popularity)
    )
    return df.reset_index(drop=True)


def load_clean_data() -> pd.DataFrame | None:
    if CLEAN_CSV.exists():
        df = pd.read_csv(CLEAN_CSV)
        if "primary_genre" in df.columns:
            df["primary_genre"] = pd.Categorical(df["primary_genre"])
        return df
    return None
