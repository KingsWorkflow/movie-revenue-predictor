import io
import textwrap

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from data import load_clean_data
from model import load_model, predict_revenue, find_comparable_films
from utils import fmt_money, fmt_pct


st.set_page_config(page_title="Movie Revenue Predictor", layout="wide")


GOLD     = "#d4af37"
EMERALD  = "#10b981"
CRIMSON  = "#c0392b"
SKY      = "#38bdf8"
DARK_BG  = "#0f172a"
LIGHT_BG = "#f8fafc"
BORDER_COLOR = "#e2e8f0"

FEATURE_COLORS = {
    "Budget": GOLD,
    "Genre": EMERALD,
    "Cast Popularity": SKY,
    "Rating": "#a78bfa",
}

PALETTE = [
    "#38bdf8", "#10b981", "#d4af37", "#c0392b", "#a78bfa",
    "#f472b6", "#34d399", "#fbbf24", "#60a5fa", "#f87171",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def style_ax(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold", color="#e8eaf0")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#e8eaf0")
    ax.yaxis.label.set_color("#e8eaf0")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#334155")
    ax.set_facecolor("#0f172a")
    ax.figure.patch.set_facecolor("#0f172a")


def metric_card(label, value, icon, color):
    with st.container(border=True):
        st.markdown(f"**{icon} {label}**")
        st.markdown(f"### {value}")


def roi_badge(roi):
    pct = fmt_pct(roi)
    if roi > 0.8:
        st.success(f"📈 Strong ROI: {pct}")
    elif roi > 0:
        st.warning(f"📈 Moderate ROI: {pct}")
    else:
        st.error(f"📈 At-Risk ROI: {pct}")


def build_feature_chart():
    features = pd.DataFrame({
        "Feature": ["Budget", "Genre", "Cast Popularity", "Rating"],
        "Importance": [0.73, 0.19, 0.18, 0.08],
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [FEATURE_COLORS[f] for f in features["Feature"]]
    bars = ax.barh(features["Feature"], features["Importance"], color=colors, height=0.5)
    for bar, val in zip(bars, features["Importance"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", ha="left", fontsize=10, color="#e8eaf0")
    ax.set_xlim(0, 1.0)
    style_ax(ax, "Feature Importance")
    ax.set_xlabel("Relative Importance", color="#e8eaf0")
    fig.tight_layout()
    return fig


@st.cache_resource
def load_resources():
    df = load_clean_data()
    model = load_model()
    return df, model


if "page" not in st.session_state:
    st.session_state.page = "Home"

df, model = load_resources()

if df is None or df.empty:
    st.error("Dataset not found. Run the preprocessing pipeline first.")
    st.stop()

genres_list = sorted(df["primary_genre"].unique())


page = st.sidebar.radio(
    "🧭 Navigate",
    ["Home", "Predictor", "Data Explorer", "Model Details"],
    index=["Home", "Predictor", "Data Explorer", "Model Details"].index(st.session_state.page),
)
if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**🎬 Movie Revenue Predictor**")
st.sidebar.caption("v1.0 — Data-Driven Revenue Forecasting")




if page == "Home":
    st.title("🎬 Movie Revenue Predictor")
    st.caption("Data-driven film revenue forecasting for investors, studios, and producers.")

    st.markdown("")

    total_films = len(df)
    r2 = 0.651
    n_genres = df["primary_genre"].nunique()
    avg_rev = df["revenue"].mean()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Total Films", f"{total_films:,}", "🎞️", GOLD)
    with m2:
        metric_card("Model R²", f"{r2:.3f}", "📐", SKY)
    with m3:
        metric_card("Genres Covered", str(n_genres), "🎭", EMERALD)
    with m4:
        metric_card("Avg Revenue", fmt_money(avg_rev), "💰", CRIMSON)

    st.markdown("")

    st.markdown("---")
    st.markdown("### 🧩 Why Movie Revenue Predictor?")
    ps1, ps2 = st.columns(2)
    with ps1:
        prob = st.container(border=True)
        with prob:
            st.markdown("#### 🔴 The Problem")
            st.write(
                "Film investors spend hours building spreadsheets to estimate box-office "
                "revenue. With limited data tools, they rely on gut feel and outdated comparables."
            )
    with ps2:
        sol = st.container(border=True)
        with sol:
            st.markdown("#### 🟢 The Solution")
            st.write(
                "**Movie Revenue Predictor** lets you input movie details and instantly get a "
                "data-backed revenue forecast, comparable films, and confidence ranges "
                "all in one place."
            )

    st.markdown(
        "Our OLS regression model is trained on **{} real films** from the TMDB "
        "5000 dataset, using budget, cast popularity, genre, and audience "
        "rating to predict global box office revenue.".format(total_films)
    )

    st.markdown("")
    cta1, cta2, cta3 = st.columns([1, 2, 1])
    with cta2:
        if st.button("🚀 Get Started — Try the Predictor", type="primary", use_container_width=True):
            st.session_state.page = "Predictor"
            st.rerun()

    with st.expander("📊 Quick Stats — Dataset Overview"):
        qs1, qs2, qs3 = st.columns(3)
        with qs1:
            st.metric("Min Budget", fmt_money(df["budget"].min()))
            st.metric("Min Revenue", fmt_money(df["revenue"].min()))
        with qs2:
            st.metric("Max Budget", fmt_money(df["budget"].max()))
            st.metric("Max Revenue", fmt_money(df["revenue"].max()))
        with qs3:
            st.metric("Median Budget", fmt_money(df["budget"].median()))
            st.metric("Median Revenue", fmt_money(df["revenue"].median()))
        st.caption(f"Dataset spans {df['release_year'].min():.0f}–{df['release_year'].max():.0f}" if "release_year" in df.columns else "")


elif page == "Predictor":
    st.markdown("### 🔮 Revenue Predictor")
    st.write("Fill in the movie details below and let our model forecast the box-office potential.")

    st.markdown("")

    with st.container(border=True):
        st.markdown("#### 📋 Movie Details")
        fcol1, fcol2 = st.columns(2, gap="large")

        with fcol1:
            title_input = st.text_input("Movie Title", placeholder="e.g., Midnight Dream", help="Optional — used only for display")
            budget_input = st.slider("Budget ($M)", min_value=0.5, max_value=250.0, value=5.0, step=0.5, help="Total production budget in millions USD")
            genre_input = st.selectbox("Primary Genre", genres_list)

        with fcol2:
            cast_input = st.slider("Lead Actor Popularity (0–100)", min_value=0, max_value=100, value=50, help="Average TMDB popularity score of the lead cast")
            rating_input = st.slider("Expected Rating (1–10)", min_value=1.0, max_value=10.0, value=7.0, step=0.1)

        btn_col1, btn_col2, _ = st.columns([1, 1, 3])
        with btn_col1:
            predict_btn = st.button("✨ Get Revenue Forecast", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🔄 Reset Form", use_container_width=True):
                st.rerun()

    if budget_input < 1.0:
        st.warning("⚠️ Budget under $1M is very low for a theatrical release.")
    if cast_input < 20:
        st.warning("⚠️ Cast popularity below 20 may indicate limited market awareness.")
    if rating_input < 4.0:
        st.warning("⚠️ Expected rating below 4.0 often correlates with poor audience reception.")

    if predict_btn:
        with st.spinner("Running model inference..."):
            result = predict_revenue(model, budget_input * 1e6, float(cast_input), float(rating_input), genre_input)

        rev   = result["revenue"]
        lo    = result["ci_lower"]
        hi    = result["ci_upper"]
        roi   = result["roi"]

        st.markdown("---")
        st.success("✅ Forecast complete!")

        r1, r2, r3 = st.columns(3)
        with r1:
            metric_card("Predicted Revenue", fmt_money(rev), "💰", EMERALD)
        with r2:
            metric_card("Confidence Range", f"{fmt_money(lo)} – {fmt_money(hi)}", "📊", SKY)
        with r3:
            roi_low = max(0.0, roi - 0.3)
            roi_high = roi + 0.3
            metric_card("ROI Range", f"{fmt_pct(roi_low)} – {fmt_pct(roi_high)}", "📈", GOLD)

        badge_col, info_col = st.columns([1, 2])
        with badge_col:
            roi_badge(roi)
        with info_col:
            st.info(f"ℹ️ Based on **{len(df[df['primary_genre'] == genre_input])}** similar {genre_input} films in our dataset")

        st.markdown("")
        st.markdown("#### 🎬 Comparable Films")
        comps = find_comparable_films(df, genre_input, budget_input * 1e6, float(cast_input), float(rating_input))
        display = comps.copy()
        display["budget"]   = display["budget"].apply(fmt_money)
        display["revenue"]  = display["revenue"].apply(fmt_money)
        display.columns     = ["Title", "Budget", "Revenue", "Genre", "Rating"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO()
        report_df = pd.DataFrame([{
            "Predicted Revenue": fmt_money(rev),
            "CI Lower": fmt_money(lo),
            "CI Upper": fmt_money(hi),
            "ROI": fmt_pct(roi),
            "Title": title_input or "Untitled",
            "Budget ($M)": budget_input,
            "Genre": genre_input,
            "Cast Popularity": cast_input,
            "Expected Rating": rating_input,
        }])
        report_df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Download Report (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"movie_revenue_report_{(title_input or 'untitled').replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


elif page == "Data Explorer":
    st.markdown("### 📊 Data Explorer")
    st.write("Explore the underlying dataset, relationships, and distributions used by the model.")
    st.markdown("")

    tab_overview, tab_rels, tab_dists = st.tabs(["Overview", "Relationships", "Distributions"])

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Films", f"{len(df):,}")
        c2.metric("Genres", df["primary_genre"].nunique())
        c3.metric("Budget Range", f"{fmt_money(df['budget'].min())} – {fmt_money(df['budget'].max())}")
        c4.metric("Revenue Range", f"{fmt_money(df['revenue'].min())} – {fmt_money(df['revenue'].max())}")

        st.markdown("")
        st.markdown("#### Dataset Summary")
        summary = df.select_dtypes(include=[np.number]).describe().T[["mean", "std", "min", "max"]]
        summary.columns = ["Mean", "Std Dev", "Min", "Max"]
        summary = summary.round(2)
        st.dataframe(summary, use_container_width=True)
        st.caption("💡 Higher variance in revenue suggests wide box-office range within each genre.")

    with tab_rels:
        st.markdown("##### Budget vs Revenue (by Genre)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x="budget", y="revenue", hue="primary_genre", ax=ax, legend=False, alpha=0.55, s=40, palette=PALETTE)
        style_ax(ax, "Budget vs Revenue by Genre")
        ax.set_xlabel("Budget (USD)")
        ax.set_ylabel("Revenue (USD)")
        st.pyplot(fig)
        plt.close()
        st.caption("📌 Higher-budget films tend toward higher revenues, but variance increases with scale.")

        st.markdown("")
        st.markdown("##### Average Revenue by Genre")
        genre_rev = df.groupby("primary_genre")["revenue"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [GOLD if v == genre_rev.max() else "#cbd5e1" for v in genre_rev.values]
        ax.barh(genre_rev.index, genre_rev.values, color=colors, height=0.6)
        style_ax(ax, "Average Revenue by Genre")
        ax.set_xlabel("Average Revenue (USD)")
        ax.set_ylabel("Genre")
        st.pyplot(fig)
        plt.close()
        st.caption("📌 Action and Adventure genres lead average revenue in this dataset.")

    with tab_dists:
        st.markdown("##### Revenue Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df["revenue"], bins=60, kde=True, ax=ax, color=GOLD, alpha=0.8)
        ax.axvline(df["revenue"].median(), color=CRIMSON, linestyle="--", linewidth=1.5, label="Median")
        ax.axvline(df["revenue"].mean(), color=SKY, linestyle="--", linewidth=1.5, label="Mean")
        ax.legend(fontsize=10)
        style_ax(ax, "Distribution of Film Revenue")
        ax.set_xlabel("Revenue (USD)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()
        st.caption(f"📌 Median revenue: {fmt_money(df['revenue'].median())} — right-skewed distribution.")

        st.markdown("")
        st.markdown("##### Budget Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df["budget"], bins=60, kde=True, ax=ax, color=SKY, alpha=0.8)
        ax.axvline(df["budget"].median(), color=CRIMSON, linestyle="--", linewidth=1.5, label="Median")
        ax.axvline(df["budget"].mean(), color=EMERALD, linestyle="--", linewidth=1.5, label="Mean")
        ax.legend(fontsize=10)
        style_ax(ax, "Distribution of Film Budget")
        ax.set_xlabel("Budget (USD)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()
        st.caption(f"📌 Median budget: {fmt_money(df['budget'].median())}.")

    with st.expander("🔍 View Raw Data"):
        st.dataframe(df.head(200), use_container_width=True, hide_index=True)
        st.caption(f"Showing first 200 of {len(df):,} rows.")


elif page == "Model Details":
    st.markdown("### 🔬 Model Details")
    st.write("A transparent look at how the model works, its performance, and limitations.")
    st.markdown("")

    tab_perf, tab_feat = st.tabs(["Model Performance", "Feature Analysis"])

    with tab_perf:
        st.markdown("#### Regression Formula")
        st.latex(
            r"\log(\text{revenue}) = \beta_0 + \beta_1 \log(\text{budget}) "
            r"+ \beta_2 \text{cast\_pop} + \beta_3 \text{rating} + \beta_4 \text{genre}"
        )
        st.caption("Revenue and budget are log-transformed to normalize their skewed distributions.")

        st.markdown("")
        st.markdown("#### Performance Metrics")
        p1, p2, p3 = st.columns(3)
        p1.metric("Training R²", "0.651")
        p2.metric("Testing R²", "0.638")
        p3.metric("RMSE", "$18.7M")

        st.markdown("")
        st.markdown("#### Interpretation")
        st.write(
            "The model explains **~65%** of the variance in box-office revenue. "
            "The remaining variance is driven by factors outside the dataset: "
            "marketing spend, theatrical release window, competition, streaming "
            "revenue, and franchise momentum."
        )
        st.info(
            "ℹ️ R² of 0.65 is strong for film revenue, which is notoriously noisy. "
            "Use predictions as directional signals, not precise guarantees."
        )

        with st.expander("📖 How to Use This Model"):
            st.markdown("""
                1. **Enter realistic inputs** — Use actual comparable film data where possible.
                2. **Treat ROI ranges as guidance** — The ±30% band captures expected uncertainty.
                3. **Cross-check with comparable films** — See the table on the Predictor page.
                4. **Account for external factors** — Marketing budget, release date, and franchise
                   effects are not in the model but heavily impact outcomes.
                5. **Revisit after production** — Actual cast popularity and audience ratings
                   may differ from projections.
            """)

    with tab_feat:
        st.markdown("#### Feature Importance")
        fig = build_feature_chart()
        st.pyplot(fig)
        plt.close()

        st.markdown("")
        st.markdown("#### Interpretation Guide")
        interp1, interp2 = st.columns(2)
        with interp1:
            st.markdown(f"**💰 Budget** ({FEATURE_COLORS['Budget']})")
            st.write("The strongest predictor. Log-transformed to handle the wide range from indie to blockbuster scales.")
            st.markdown(f"**🎭 Genre** ({FEATURE_COLORS['Genre']})")
            st.write("Captures genre-level revenue benchmarks. Action and Adventure tend to outperform Dramas on average.")
        with interp2:
            st.markdown(f"**⭐ Cast Popularity** ({FEATURE_COLORS['Cast Popularity']})")
            st.write("Avg TMDB popularity of lead cast. Signals audience draw and pre-release awareness.")
            st.markdown(f"**📊 Rating** ({FEATURE_COLORS['Rating']})")
            st.write("Expected audience rating. Has a positive but modest effect — great films still need reach.")

        st.markdown("")
        st.markdown("#### Limitations")
        lim = st.container(border=True)
        with lim:
            st.warning("""
                - 📅 Dataset covers **2000–2016** (pre-streaming era); modern dynamics differ.
                - 🇺🇸 Predicts US theatrical revenue only; global performance varies widely.
                - 📐 Linear model may underfit complex non-linear dynamics.
                - ⚠️ **Not investment advice** — use as a starting point only.
                - 🚫 No marketing spend, franchise status, or release timing features.
            """)
