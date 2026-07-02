"""
app.py
------
Streamlit web app for predicting used-car selling prices.

Phase 1 additions:
    - City selector dropdown (6 Indian cities)
    - Brand popularity automatically looked up from data/brand_popularity.json
    - Prediction confidence range (±15% band shown as context)
    - Feature importance chart
    - "Model last trained on..." metadata line

Run from the project root:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor  # required so joblib can deserialize the saved XGBoost model

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
MODEL_PATH      = "models/car_price_model.pkl"
DATA_PATH       = "data/processed_car_data.csv"
POPULARITY_PATH = "data/brand_popularity.json"

CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"]

OWNER_LABELS = {
    0: "First Owner",
    1: "Second Owner",
    2: "Third Owner",
    3: "Fourth & Above Owner",
}
TRANSMISSION_LABELS = {0: "Manual", 1: "Automatic"}


# --------------------------------------------------------------------------- #
# LOAD RESOURCES  (cached — only runs once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model(filepath: str):
    return joblib.load(filepath)


@st.cache_data
def load_reference_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


@st.cache_data
def load_popularity_map(filepath: str) -> dict:
    """Load the brand → popularity score mapping saved during preprocessing."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_training_columns(model) -> list:
    return list(model.feature_names_in_)


def extract_categories(columns, prefix: str) -> list:
    return sorted(col[len(prefix):] for col in columns if col.startswith(prefix))


def get_realistic_ranges(df: pd.DataFrame) -> dict:
    return {
        "km_driven_max": int(df["km_driven"].max()) if "km_driven" in df.columns else 300_000,
        "car_age_max":   int(df["car_age"].max())   if "car_age"   in df.columns else 30,
    }


# --------------------------------------------------------------------------- #
# BUILD INPUT DATAFRAME
# --------------------------------------------------------------------------- #
def build_input_dataframe(
    feature_columns: list,
    km_driven: int,
    car_age: int,
    owner: int,
    transmission: int,
    fuel: str,
    brand: str,
    city: str,
    brand_popularity: float,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the model's training columns exactly.
    All columns start at 0; we set the relevant ones to their input values.
    """
    input_df = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)

    # Compute interaction features (must mirror data_preprocessing.py exactly)
    km_per_year        = km_driven / (car_age + 1)
    age_km_interaction = car_age   * km_driven
    popularity_x_age   = brand_popularity * car_age

    # Direct numeric / ordinal features
    for col, val in {
        "km_driven":           km_driven,
        "car_age":             car_age,
        "owner":               owner,
        "transmission":        transmission,
        "brand_popularity":    brand_popularity,
        "km_per_year":         km_per_year,          # Phase 2: new
        "age_km_interaction":  age_km_interaction,   # Phase 2: new
        "popularity_x_age":    popularity_x_age,     # Phase 2: new
    }.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # One-hot: fuel
    fuel_col = f"fuel_{fuel}"
    if fuel_col in input_df.columns:
        input_df.at[0, fuel_col] = 1

    # One-hot: brand
    brand_col = f"brand_{brand}"
    if brand_col in input_df.columns:
        input_df.at[0, brand_col] = 1

    # One-hot: city (Phase 1: new)
    city_col = f"city_{city}"
    if city_col in input_df.columns:
        input_df.at[0, city_col] = 1

    return input_df


# --------------------------------------------------------------------------- #
# VALIDATION
# --------------------------------------------------------------------------- #
def validate_inputs(km_driven: int, car_age: int, ranges: dict) -> list:
    warnings = []
    if km_driven > ranges["km_driven_max"]:
        warnings.append(
            f"km_driven ({km_driven:,}) exceeds the highest value seen during "
            f"training (~{ranges['km_driven_max']:,} km). Prediction may be less reliable."
        )
    if car_age > ranges["car_age_max"]:
        warnings.append(
            f"car_age ({car_age} yrs) exceeds the highest value seen during "
            f"training (~{ranges['car_age_max']} yrs). Prediction may be less reliable."
        )
    return warnings


# --------------------------------------------------------------------------- #
# PREDICT + FORMAT
# --------------------------------------------------------------------------- #
def predict_price(model, input_df: pd.DataFrame) -> float:
    return float(model.predict(input_df)[0])


def format_inr(amount: float) -> str:
    return f"₹{amount:,.0f}"


# --------------------------------------------------------------------------- #
# CUSTOM CSS
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; max-width: 880px; }
        .app-subtitle { text-align:center; color:#6b7280; font-size:1.05rem; margin-top:-0.5rem; margin-bottom:1.5rem; }
        .section-header { font-size:1.1rem; font-weight:600; margin-bottom:0.5rem; }
        div.stButton > button { width:100%; height:3rem; font-size:1.05rem; font-weight:600; border-radius:10px; }
        .result-card { background:linear-gradient(135deg,#16a34a,#15803d); color:white; padding:1.75rem; border-radius:16px; text-align:center; margin-top:1rem; box-shadow:0 4px 14px rgba(22,163,74,0.25); }
        .result-card .label { font-size:0.95rem; opacity:0.85; margin-bottom:0.2rem; }
        .result-card .price { font-size:2.4rem; font-weight:700; margin:0; }
        .result-card .range { font-size:0.9rem; opacity:0.8; margin-top:0.4rem; }
        .meta-line { text-align:center; color:#9ca3af; font-size:0.82rem; margin-top:0.6rem; }
        .disclaimer { text-align:center; color:#9ca3af; font-size:0.80rem; margin-top:0.5rem; }
        .popularity-badge { background:#f0fdf4; border:1px solid #86efac; color:#15803d; border-radius:8px; padding:0.3rem 0.7rem; font-size:0.85rem; font-weight:600; display:inline-block; }
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# MAIN UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
    inject_css()

    # Title
    st.markdown("<h1 style='text-align:center;'>🚗 Used Car Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Real-time price estimates powered by a Random Forest model 🌳</div>", unsafe_allow_html=True)

    # Load resources
    try:
        model          = load_model(MODEL_PATH)
        feature_cols   = get_training_columns(model)
    except FileNotFoundError:
        st.error(f"❌ Model not found at '{MODEL_PATH}'. Run `python src/train_model.py` first.")
        st.stop()

    try:
        ref_df         = load_reference_data(DATA_PATH)
        popularity_map = load_popularity_map(POPULARITY_PATH)
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.stop()

    fuel_options  = extract_categories(ref_df.columns, "fuel_")
    brand_options = extract_categories(ref_df.columns, "brand_")
    ranges        = get_realistic_ranges(ref_df)

    # Model metadata line
    model_rows = len(ref_df)
    st.markdown(
        f"<div class='meta-line'>Model trained on {model_rows:,} listings · "
        f"{len(feature_cols)} features · "
        f"Cities: {', '.join(CITIES)}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ------------------------------------------------------------------ #
    # INPUT FORM
    # ------------------------------------------------------------------ #
    with st.container(border=True):
        st.markdown("<div class='section-header'>📋 Car Details</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            km_driven = st.number_input(
                "🛣️ Kilometers Driven", min_value=0, max_value=2_000_000,
                value=50_000, step=1_000,
            )
            car_age = st.slider(
                "📅 Car Age (years)",
                min_value=0, max_value=max(ranges["car_age_max"], 30),
                value=5,
            )
            transmission_label = st.selectbox(
                "⚙️ Transmission", list(TRANSMISSION_LABELS.values())
            )
            city = st.selectbox("📍 City", CITIES)          # Phase 1: new

        with col2:
            owner_label = st.selectbox("👤 Owner", list(OWNER_LABELS.values()))
            fuel        = st.selectbox("⛽ Fuel Type", fuel_options)
            brand       = st.selectbox("🏷️ Brand", brand_options)

            # Show brand popularity score as a live badge
            pop_score = popularity_map.get(brand, 50)
            st.markdown(
                f"<div style='margin-top:0.4rem;'>📊 Brand Popularity&nbsp;&nbsp;"
                f"<span class='popularity-badge'>{pop_score} / 100</span></div>",
                unsafe_allow_html=True,
            )

    # Decode labels back to numeric codes
    transmission = [k for k, v in TRANSMISSION_LABELS.items() if v == transmission_label][0]
    owner        = [k for k, v in OWNER_LABELS.items()        if v == owner_label][0]

    st.write("")

    # ------------------------------------------------------------------ #
    # PREDICT
    # ------------------------------------------------------------------ #
    if st.button("💰 Predict Price", type="primary"):

        for w in validate_inputs(km_driven, car_age, ranges):
            st.warning(f"⚠️ {w}")

        try:
            input_df = build_input_dataframe(
                feature_columns  = feature_cols,
                km_driven        = km_driven,
                car_age          = car_age,
                owner            = owner,
                transmission     = transmission,
                fuel             = fuel,
                brand            = brand,
                city             = city,
                brand_popularity = pop_score,
            )

            price     = predict_price(model, input_df)
            low, high = price * 0.85, price * 1.15   # ±15% confidence band

            st.markdown(f"""
            <div class="result-card">
                <div class="label">Estimated Selling Price</div>
                <p class="price">{format_inr(price)}</p>
                <div class="range">Likely range: {format_inr(low)} – {format_inr(high)}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                "<div class='disclaimer'>⚠️ Estimate based on historical CarDekho data. "
                "Actual market price may vary.</div>",
                unsafe_allow_html=True,
            )

            # ---- Feature importance chart (Phase 2 preview) ----
            with st.expander("📈 See what drives this prediction"):
                importance = pd.DataFrame({
                    "Feature":    model.feature_names_in_,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=False).head(12)

                st.bar_chart(importance.set_index("Feature")["Importance"])

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


if __name__ == "__main__":
    main()