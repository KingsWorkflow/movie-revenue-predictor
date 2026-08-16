import numpy as np
import pandas as pd
import joblib
import streamlit as st

from config import MODEL_PATH


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_revenue(model: dict, budget: float, cast_pop: float,
                    vote_avg: float, genre: str) -> dict:
    payload    = model
    model_full = payload["model_full"]
    df_train   = payload["df"]

    dm = df_train.copy()
    if not hasattr(dm["primary_genre"].dtype, "categories"):
        dm["primary_genre"] = pd.Categorical(dm["primary_genre"])

    inp = pd.DataFrame({
        "log_budget":    [np.log1p(budget)],
        "log_cast_pop":  [np.log1p(cast_pop)],
        "vote_average":  [vote_avg],
        "primary_genre": pd.Categorical(
            [genre],
            categories=dm["primary_genre"].cat.categories,
        ),
    })

    try:
        log_pred = model_full.predict(inp)[0]
        rev      = float(np.expm1(log_pred))
        ci       = model_full.get_prediction(inp).conf_int(alpha=0.05)
        ci_lower = float(np.expm1(ci[0][0]))
        ci_upper = float(np.expm1(ci[0][1]))
        mlabel   = "full"
    except Exception:
        model_simple = payload["model_simple"]
        log_pred     = model_simple.predict(inp)[0]
        rev          = float(np.expm1(log_pred))
        ci_lower     = rev * 0.6
        ci_upper     = rev * 1.4
        mlabel       = "simple"

    roi       = (rev - budget) / budget
    roi_lower = (ci_lower - budget) / budget
    roi_upper = (ci_upper - budget) / budget

    return {
        "revenue":   rev,
        "ci_lower":  ci_lower,
        "ci_upper":  ci_upper,
        "roi":       roi,
        "roi_lower": roi_lower,
        "roi_upper": roi_upper,
        "model":     mlabel,
    }


def find_comparable_films(df: pd.DataFrame, genre: str,
                          budget: float, cast_pop: float,
                          vote_avg: float, n: int = 10) -> pd.DataFrame:
    mask = df["primary_genre"] == genre
    lo, hi = budget * 0.5, budget * 1.5
    comps = df[mask & (df["budget"] >= lo) & (df["budget"] <= hi)].copy()
    if comps.empty:
        comps = df[df["primary_genre"] == genre].copy()
    comps["similarity"] = (
        np.abs(np.log1p(comps["cast_popularity_score"]) - np.log1p(cast_pop))
        + np.abs(comps["vote_average"] - vote_avg)
    )
    top = comps.nsmallest(n, "similarity")[
        ["title", "budget", "revenue", "primary_genre", "vote_average"]
    ]
    return top.reset_index(drop=True)