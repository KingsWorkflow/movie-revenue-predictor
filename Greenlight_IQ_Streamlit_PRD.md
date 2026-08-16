# Movie Revenue Predictor — Streamlit PRD v3.0
## Film Revenue Forecasting App (Simple & Clean)

---

## **PROJECT OVERVIEW**

**What:** A Streamlit app that helps film investors estimate box-office revenue from movie details.

**Why:** Streamlit makes it super easy to build data apps without web framework complexity.

**Timeline:** 6–8 weeks

**Tech Stack:**
- **Python** (pandas, numpy, scikit-learn)
- **Streamlit** (for UI/interaction)
- **Matplotlib/Seaborn** (for visualizations)
- **Pre-trained regression model** (from your research project)
- **Static CSV dataset** (Kaggle TMDB 5000)

**No:** Flask, Django, databases, authentication, HTML/CSS

---

## **THE PROBLEM**

A film investor needs to quickly answer: **"What's a realistic box-office forecast for this $5M indie drama?"**

**Today:** They build spreadsheets manually (hours of work)  
**With Movie Revenue Predictor:** Input movie details → Get estimate + comparable films → Done (5 minutes)

---

## **WHAT YOU'RE BUILDING** *(MVP Scope)*

### Core Features (Simple)

| Feature | Description |
|---------|-------------|
| **Input Form** | Fields: Budget, Genre (dropdown), Lead Actor, Expected Rating |
| **Revenue Forecast** | Run model → Show predicted revenue + confidence range |
| **Comparable Movies** | Show 5–10 real movies from dataset that match the profile |
| **Visualizations** | Charts showing model logic (budget vs revenue, genre breakdown, etc.) |
| **Summary Report** | Display all info in a readable format |

---

## **APP STRUCTURE** *(What Streamlit Displays)*

### Page 1: Home
```
Movie Revenue Predictor
A Data-Driven Film Revenue Forecasting Tool

📊 Quick Overview:
   - Trained on 2,397 films from Kaggle dataset
   - Model accuracy: R² = 0.651
   - Forecasts US theatrical revenue

[Get Started] Button → Goes to Predictor
[View Data] Button → Goes to Data Explorer
```

### Page 2: Predictor
```
┌─────────────────────────────────────────────┐
│  MOVIE REVENUE FORECAST                     │
├─────────────────────────────────────────────┤
│                                             │
│  Movie Title: [text input]                  │
│  Budget ($M): [slider 0.5 - 250]            │
│  Genre: [dropdown]                          │
│  Lead Actor: [text input]                   │
│  Expected Rating: [slider 1 - 10]           │
│                                             │
│  [FORECAST] Button                          │
│                                             │
├─────────────────────────────────────────────┤
│  RESULTS                                    │
├─────────────────────────────────────────────┤
│                                             │
│  Estimated Revenue: $12.4M                  │
│  Confidence Range: $9.2M – $16.8M           │
│  ROI: 148% – 236%                           │
│                                             │
│  Model Confidence: Based on 127 similar     │
│  dramas in our dataset                      │
│                                             │
├─────────────────────────────────────────────┤
│  TOP 10 COMPARABLE FILMS                    │
├─────────────────────────────────────────────┤
│                                             │
│  Title            Budget   Revenue  Genre  │
│  ──────────────────────────────────────    │
│  Moonlight        $1.5M    $65.3M  Drama   │
│  Spotlight        $6.0M    $90.2M  Drama   │
│  Whiplash         $3.2M    $49.3M  Drama   │
│  ...                                       │
│                                             │
└─────────────────────────────────────────────┘
```

### Page 3: Data Explorer
```
┌─────────────────────────────────────────────┐
│  DATA OVERVIEW                              │
├─────────────────────────────────────────────┤
│  Total films: 2,397                         │
│  Genres: 18                                 │
│  Budget range: $500K – $250M                │
│  Revenue range: $1M – $2.8B                 │
│                                             │
│  [Visual: Budget vs Revenue scatter plot]   │
│  [Visual: Revenue by genre bar chart]       │
│  [Visual: Dataset distribution]             │
└─────────────────────────────────────────────┘
```

### Page 4: Model Details
```
┌─────────────────────────────────────────────┐
│  HOW THE MODEL WORKS                        │
├─────────────────────────────────────────────┤
│                                             │
│  Formula:                                   │
│  log(revenue) = β₀ + β₁·log(budget)         │
│                 + β₂·cast_pop                │
│                 + β₃·rating                 │
│                 + β₄·genre                  │
│                                             │
│  Performance:                               │
│  • Training R²: 0.651                       │
│  • Testing R²: 0.638                        │
│  • RMSE: $18.7M                             │
│                                             │
│  Feature Importance:                        │
│  • Budget: 0.73 (strongest)                 │
│  • Genre: 0.19                              │
│  • Cast: 0.18                               │
│  • Rating: 0.08                             │
│                                             │
│  Limitations:                               │
│  ⚠ Data from 2000–2016 (pre-streaming)     │
│  ⚠ US theatrical only                       │
│  ⚠ Linear model (may underfit nonlinear)    │
│                                             │
└─────────────────────────────────────────────┘
```

---

## **CODE STRUCTURE** *(What You Write)*

```
movie-revenue-predictor/
│
├── app.py                    # Main Streamlit app
├── data.py                   # Load & clean data
├── model.py                  # Load trained model, make predictions
├── utils.py                  # Helper functions (find comps, format output)
├── requirements.txt          # pandas, streamlit, scikit-learn, matplotlib
│
├── data/
│   └── tmdb_5000_movies.csv  # Dataset
│
├── models/
│   └── regression_model.pkl  # Pre-trained OLS model
│
└── README.md                 # How to run
```

---

## **DETAILED CODE BREAKDOWN**

### `data.py` — Load & Clean Data

```python
import pandas as pd
import numpy as np

def load_and_clean_data():
    """Load TMDB dataset and clean it"""
    
    # 1. Load
    df = pd.read_csv('data/tmdb_5000_movies.csv')
    print(f"Loaded {len(df)} films")
    
    # 2. Clean: Remove zero/missing budget or revenue
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    print(f"After cleaning: {len(df)} films")
    
    # 3. Feature engineering
    df['log_budget'] = np.log10(df['budget'])
    df['log_revenue'] = np.log10(df['revenue'])
    df['primary_genre'] = df['genres'].apply(extract_primary_genre)
    df['cast_popularity'] = df['cast'].apply(extract_cast_popularity)
    
    # 4. Return clean dataset
    return df

def extract_primary_genre(genre_str):
    """Extract first genre from JSON string"""
    # Logic here
    pass

def extract_cast_popularity(cast_str):
    """Extract first actor's popularity score"""
    # Logic here
    pass
```

### `model.py` — Load Model & Predict

```python
import pickle
import numpy as np
import pandas as pd

def load_model():
    """Load pre-trained OLS regression model"""
    with open('models/regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict_revenue(model, budget_m, genre, cast_pop, rating):
    """
    Predict box-office revenue for a movie
    
    Args:
        budget_m: Budget in millions
        genre: Genre name (str)
        cast_pop: Cast popularity score (0-100)
        rating: Expected audience rating (1-10)
    
    Returns:
        (predicted_revenue, confidence_range)
    """
    # Convert to features
    log_budget = np.log10(budget_m * 1_000_000)
    
    # Create feature vector
    # (This depends on your actual model columns)
    features = {
        'log_budget': log_budget,
        'cast_popularity': cast_pop,
        'audience_rating': rating,
        'genre_Drama': 1 if genre == 'Drama' else 0,
        'genre_Action': 1 if genre == 'Action' else 0,
        # ... (other genres)
    }
    
    X = pd.DataFrame([features])
    
    # Predict log revenue
    log_pred = model.predict(X)[0]
    revenue_pred = 10 ** log_pred  # Convert back from log scale
    
    # Add confidence interval (e.g., ±50%)
    lower_bound = revenue_pred * 0.75
    upper_bound = revenue_pred * 1.33
    
    return revenue_pred, (lower_bound, upper_bound)

def find_comparable_films(df, genre, budget_m, cast_pop, rating, n=10):
    """Find similar films from dataset to show as comps"""
    
    # Filter: Same genre, similar budget (±50%)
    comps = df[
        (df['primary_genre'] == genre) &
        (df['budget'] > budget_m * 1_000_000 * 0.5) &
        (df['budget'] < budget_m * 1_000_000 * 1.5)
    ]
    
    # Sort by similarity
    comps['similarity'] = (
        np.abs(np.log10(comps['cast_popularity']) - cast_pop) +
        np.abs(comps['vote_average'] - rating)
    )
    
    # Return top 10
    return comps.nsmallest(n, 'similarity')[
        ['title', 'budget', 'revenue', 'primary_genre', 'vote_average']
    ]
```

### `app.py` — Main Streamlit App

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data import load_and_clean_data
from model import load_model, predict_revenue, find_comparable_films

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Movie Revenue Predictor", layout="wide")

# ============ LOAD DATA & MODEL ============
@st.cache_resource
def load_resources():
    df = load_and_clean_data()
    model = load_model()
    return df, model

df, model = load_resources()

# ============ SIDEBAR NAVIGATION ============
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🎬 Predictor", "📊 Data Explorer", "🤖 Model Details"]
)

# ============ PAGE 1: HOME ============
if page == "🏠 Home":
    st.title("Movie Revenue Predictor")
    st.subheader("A Data-Driven Film Revenue Forecasting Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Films in Dataset", f"{len(df):,}")
        st.metric("Model R² Score", "0.651")
    
    with col2:
        st.metric("Genres Covered", 18)
        st.metric("Forecasts Made", "Real-time")
    
    st.write("""
    ### The Problem
    Film investors spend **hours** building spreadsheets to estimate revenue. 
    With limited data tools, they rely on gut feel and outdated comps.
    
    ### The Solution
    Input movie details → Get an **instant, data-backed forecast** → 
    See comparable films → **Make faster decisions**.
    
    ### How It Works
    1. Enter your movie's budget, genre, cast, and expected rating
    2. Our ML model predicts US theatrical revenue based on 2,397 real films
    3. See comparable films that validate the estimate
    4. Share your forecast with your investment committee
    """)

# ============ PAGE 2: PREDICTOR ============
elif page == "🎬 Predictor":
    st.title("Movie Revenue Forecaster")
    
    # Left column: Input form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Movie Details")
        
        title = st.text_input("Movie Title", placeholder="e.g., Midnight Dream")
        budget_m = st.slider("Budget ($M)", 0.5, 250.0, 5.0)
        genre = st.selectbox(
            "Primary Genre",
            ["Drama", "Action", "Adventure", "Comedy", "Horror", "Thriller"]
        )
        cast_pop = st.slider("Lead Cast Popularity (0-100)", 0, 100, 50)
        rating = st.slider("Expected Audience Rating (1-10)", 1.0, 10.0, 7.0)
    
    # Button to forecast
    if st.button("Get Revenue Forecast", key="forecast_btn"):
        
        # Make prediction
        pred_revenue, (lower, upper) = predict_revenue(
            model, budget_m, genre, cast_pop, rating
        )
        
        # Display results
        col2.subheader("📈 Forecast Results")
        
        col2.metric(
            "Estimated Revenue",
            f"${pred_revenue / 1_000_000:.1f}M",
            f"Range: ${lower/1_000_000:.1f}M – ${upper/1_000_000:.1f}M"
        )
        
        roi_low = (lower / (budget_m * 1_000_000)) * 100 - 100
        roi_high = (upper / (budget_m * 1_000_000)) * 100 - 100
        col2.metric("ROI Range", f"{roi_low:.0f}% – {roi_high:.0f}%")
        
        # Count comparable films
        comps = find_comparable_films(df, genre, budget_m, cast_pop, rating)
        col2.info(
            f"✓ Based on {len(comps)} similar {genre} films in our dataset"
        )
        
        # Show comparable films
        st.subheader("🎥 Comparable Films")
        
        comps_display = comps[[
            'title', 'budget', 'revenue', 'vote_average'
        ]].head(10).copy()
        
        comps_display.columns = [
            'Title', 'Budget ($M)', 'Revenue ($M)', 'Rating'
        ]
        comps_display['Budget ($M)'] = (comps_display['Budget ($M)'] / 1_000_000).round(1)
        comps_display['Revenue ($M)'] = (comps_display['Revenue ($M)'] / 1_000_000).round(1)
        
        st.dataframe(comps_display, use_container_width=True)
        
        # Summary
        st.success(f"""
        ### Forecast Summary for "{title}"
        
        **Input:** {genre} film with ${budget_m:.1f}M budget, cast popularity {cast_pop}, expected rating {rating:.1f}
        
        **Prediction:** Revenue of ${pred_revenue/1_000_000:.1f}M (range: ${lower/1_000_000:.1f}M–${upper/1_000_000:.1f}M)
        
        **Reasoning:** This forecast is based on {len(comps)} similar {genre} films in our training dataset. 
        The model explains 65% of revenue variance using budget, cast popularity, rating, and genre.
        
        **Note:** This is a statistical estimate for US theatrical revenue only. Actual results depend on marketing, timing, and market conditions.
        """)

# ============ PAGE 3: DATA EXPLORER ============
elif page == "📊 Data Explorer":
    st.title("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Films", f"{len(df):,}")
    col2.metric("Budget Range", f"${df['budget'].min()/1e6:.1f}M – ${df['budget'].max()/1e6:.0f}M")
    col3.metric("Revenue Range", f"${df['revenue'].min()/1e6:.1f}M – ${df['revenue'].max()/1e9:.1f}B")
    
    # Visualizations
    st.subheader("Budget vs Revenue (Log Scale)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['log_budget'], df['log_revenue'], alpha=0.5)
    ax.set_xlabel('Log Budget')
    ax.set_ylabel('Log Revenue')
    ax.set_title('Strong positive correlation: R² = 0.65')
    st.pyplot(fig)
    
    st.subheader("Revenue by Genre")
    genre_revenue = df.groupby('primary_genre')['revenue'].median().sort_values(ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(10, 5))
    genre_revenue.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Median Revenue ($)')
    ax.set_title('Top Genres by Revenue')
    st.pyplot(fig)

# ============ PAGE 4: MODEL DETAILS ============
elif page == "🤖 Model Details":
    st.title("How the Model Works")
    
    st.subheader("The Regression Formula")
    st.latex(r'''
    \log(revenue) = \beta_0 + \beta_1 \log(budget) + \beta_2 \text{cast\_pop} 
    + \beta_3 \text{rating} + \beta_4 \text{genre}
    ''')
    
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training R²", "0.651")
    col2.metric("Testing R²", "0.638")
    col3.metric("RMSE", "$18.7M")
    
    st.write("""
    **Interpretation:**
    - The model explains 65.1% of revenue variance in training data
    - It generalizes well to unseen data (testing R² = 0.638)
    - Average prediction error is ±$18.7M
    """)
    
    st.subheader("Feature Importance")
    features = pd.DataFrame({
        'Feature': ['Budget', 'Genre', 'Cast Popularity', 'Rating'],
        'Importance': [0.73, 0.19, 0.18, 0.08]
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features['Feature'], features['Importance'])
    ax.set_xlabel('Coefficient Value')
    st.pyplot(fig)
    
    st.subheader("Limitations & Assumptions")
    st.warning("""
    ⚠️ **Important Limitations:**
    - Dataset spans 2000–2016 (pre-streaming era; may not capture modern dynamics)
    - Predicts US theatrical revenue only (no international or ancillary)
    - Linear model may underfit nonlinear relationships (e.g., franchise effects)
    - Trained on historical data; assumes normal market conditions
    - **Not investment advice** — use as a starting point, not sole decision factor
    """)
```

---

## **WHAT YOU NEED TO SUBMIT**

1. **`app.py`** — Main Streamlit app (all pages)
2. **`data.py`** — Data loading & cleaning
3. **`model.py`** — Model loading & predictions
4. **`utils.py`** — Helper functions
5. **`requirements.txt`** — Dependencies
6. **`README.md`** — How to run
7. **Pre-trained model** saved as `models/regression_model.pkl`
8. **Data file** `data/tmdb_5000_movies.csv`

---

## **HOW TO RUN**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run app
streamlit run app.py

# 3. Open browser
# App launches at http://localhost:8501
```

---

## **REQUIREMENTS.txt**

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.1
```

---

## **README.md TEMPLATE**

```markdown
# Movie Revenue Predictor — Film Revenue Forecasting

A Streamlit app that helps independent film investors estimate box-office 
revenue based on budget, cast, genre, and expected rating.

## Problem It Solves

Film investors currently:
- Build spreadsheets manually (hours of work)
- Have no fast way to validate revenue estimates
- Can't easily explain forecasts to their committees

**Movie Revenue Predictor** solves this: Input → Forecast → Comps → Done (5 minutes)

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Features

- **🎬 Predictor**: Enter movie details, get instant revenue forecast
- **📊 Data Explorer**: Visualize the dataset and relationships
- **🤖 Model Details**: Understand how the model works
- **🏠 Home**: Overview and problem statement

## Data & Model

- **Dataset**: 2,397 films from Kaggle TMDB 5000 (2000–2016)
- **Model**: Multiple linear regression (OLS)
- **Performance**: R² = 0.65 (explains 65% of revenue variance)
- **Features**: Budget, cast popularity, rating, genre

## Limitations

- Data is from pre-streaming era (2000–2016)
- Predicts US theatrical revenue only
- Linear model may underfit complex dynamics
- **Not investment advice** — use as a starting point only

## Project Structure

```
├── app.py              # Main Streamlit app
├── data.py             # Load & clean data
├── model.py            # Model & predictions
├── utils.py            # Helper functions
├── requirements.txt
├── data/
│   └── tmdb_5000_movies.csv
└── models/
    └── regression_model.pkl
```
```

---

## **SIMPLIFIED TIMELINE**

| Week | Task |
|------|------|
| 1–2 | Data loading, cleaning, feature engineering |
| 2–3 | Train & save regression model |
| 3–4 | Build Streamlit pages (Home, Predictor) |
| 4–5 | Add Data Explorer and Model Details pages |
| 5–6 | Visualizations, polish UI |
| 6–7 | Testing, documentation, README |
| 7–8 | Final demo & submission |

---

## **SUCCESS CRITERIA**

✓ App loads without errors  
✓ All 4 pages work and display correctly  
✓ Forecast form takes inputs and returns predictions  
✓ Comparable films display correctly  
✓ Visualizations are clear and informative  
✓ Code is modular (separate files for data, model, etc.)  
✓ README explains how to run  
✓ Demo works live in 5 minutes  

---

## **WHY STREAMLIT IS PERFECT FOR THIS**

- ✅ No HTML/CSS/JavaScript needed
- ✅ Built-in form widgets (sliders, dropdowns, buttons)
- ✅ Easy data visualization (matplotlib, plotly)
- ✅ Can save/share apps easily
- ✅ Great for data science projects
- ✅ You already built your prototype in Streamlit!

---

**END OF DOCUMENT**
