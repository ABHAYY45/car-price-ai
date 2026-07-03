"""
app.py
------
Streamlit web app for predicting used-car selling prices.

Phase 3 changes:
    REMOVED: city selector (was fake random data)
    REMOVED: brand_popularity badge (was a hardcoded manual table)
    ADDED:   brand_avg_price lookup from data/brand_avg_price.json
    ADDED:   log_km_driven computed in build_input_dataframe
    KEPT:    km_per_year, age_km_interaction, confidence range,
             feature importance expander

Run from the project root:
    streamlit run app.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor  # needed so joblib can deserialize XGBoost model

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
MODEL_PATH     = "models/car_price_model.pkl"
DATA_PATH      = "data/processed_car_data.csv"
BRAND_AVG_PATH = "data/brand_avg_price.json"

OWNER_LABELS = {
    0: "First Owner",
    1: "Second Owner",
    2: "Third Owner",
    3: "Fourth & Above Owner",
}
TRANSMISSION_LABELS = {0: "Manual", 1: "Automatic"}


# --------------------------------------------------------------------------- #
# LOAD RESOURCES
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model(filepath: str):
    return joblib.load(filepath)


@st.cache_data
def load_reference_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


@st.cache_data
def load_brand_avg_map(filepath: str) -> dict:
    """
    Load the brand -> avg_price mapping computed from training data.
    This was saved by train_model.py so the app never recomputes it
    (which would risk using the full dataset and causing leakage).
    """
    with open(filepath, "r") as f:
        return json.load(f)


def get_training_columns(model) -> list:
    return list(model.feature_names_in_)


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
    brand_avg_price: float,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the model's training columns exactly.

    Every feature computed here must mirror what data_preprocessing.py and
    train_model.py produce during training. If the two ever diverge, the
    model will silently predict wrong values (no crash, just bad output).

    Checklist of mirrored computations:
        log_km_driven   = np.log1p(km_driven)      [mirrors add_log_km_driven()]
        brand_freq      = looked up from ref CSV    [mirrors add_brand_freq()]
        km_per_year     = km_driven / (car_age + 1) [mirrors add_interaction_features()]
        age_km          = car_age * km_driven       [mirrors add_interaction_features()]
        brand_avg_price = from brand_avg_price.json [mirrors add_brand_avg_price()]
        brand_{X}       = 1 if selected brand       [mirrors encode_brand()]
        fuel_{X}        = 1 if selected fuel        [mirrors encode_categorical_features()]
    """
    # Start with all zeros for every column the model expects
    input_df = pd.DataFrame(
        [np.zeros(len(feature_columns))], columns=feature_columns
    )

    # --- Numeric / ordinal features ---
    direct = {
        "km_driven":       km_driven,
        "transmission":    transmission,
        "car_age":         car_age,
        "owner":           owner,
        "log_km_driven":   np.log1p(km_driven),
        "km_per_year":     km_driven / (car_age + 1),
        "age_km_interaction": car_age * km_driven,
        "brand_avg_price": brand_avg_price,
    }
    for col, val in direct.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # brand_freq: look up from reference data (count of this brand in training set)
    # This is set separately in main() and passed via brand_freq_val -- see caller.

    # --- One-hot: fuel ---
    fuel_col = f"fuel_{fuel}"
    if fuel_col in input_df.columns:
        input_df.at[0, fuel_col] = 1

    # --- One-hot: brand ---
    brand_col = f"brand_{brand}"
    if brand_col in input_df.columns:
        input_df.at[0, brand_col] = 1
    # If brand not found (baseline category dropped by drop_first=True),
    # all brand_ columns stay 0 -- this correctly represents the baseline.

    return input_df


# --------------------------------------------------------------------------- #
# VALIDATION
# --------------------------------------------------------------------------- #
def validate_inputs(km_driven: int, car_age: int, ranges: dict) -> list:
    warnings = []
    if km_driven > ranges["km_driven_max"]:
        warnings.append(
            f"km_driven ({km_driven:,}) exceeds training data max "
            f"(~{ranges['km_driven_max']:,} km). Prediction may be less reliable."
        )
    if car_age > ranges["car_age_max"]:
        warnings.append(
            f"car_age ({car_age} yrs) exceeds training data max "
            f"(~{ranges['car_age_max']} yrs). Prediction may be less reliable."
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
        .app-subtitle { text-align:center; color:#6b7280; font-size:1.05rem;
                        margin-top:-0.5rem; margin-bottom:1.5rem; }
        .section-header { font-size:1.1rem; font-weight:600; margin-bottom:0.5rem; }
        div.stButton > button { width:100%; height:3rem; font-size:1.05rem;
                                font-weight:600; border-radius:10px; }
        .result-card { background:linear-gradient(135deg,#16a34a,#15803d);
                       color:white; padding:1.75rem; border-radius:16px;
                       text-align:center; margin-top:1rem;
                       box-shadow:0 4px 14px rgba(22,163,74,0.25); }
        .result-card .label { font-size:0.95rem; opacity:0.85; margin-bottom:0.2rem; }
        .result-card .price { font-size:2.4rem; font-weight:700; margin:0; }
        .result-card .range { font-size:0.9rem; opacity:0.8; margin-top:0.4rem; }
        .meta-line  { text-align:center; color:#9ca3af; font-size:0.82rem; margin-top:0.6rem; }
        .disclaimer { text-align:center; color:#9ca3af; font-size:0.80rem; margin-top:0.5rem; }
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# MAIN UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
    inject_css()

    st.markdown(
        "<h1 style='text-align:center;'>🚗 Used Car Price Predictor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='app-subtitle'>Data-driven price estimates powered by XGBoost 🌳</div>",
        unsafe_allow_html=True,
    )

    # --- Load resources ---
    try:
        model        = load_model(MODEL_PATH)
        feature_cols = get_training_columns(model)
    except FileNotFoundError:
        st.error(f"❌ Model not found at '{MODEL_PATH}'. Run `python src/train_model.py` first.")
        st.stop()

    try:
        ref_df        = load_reference_data(DATA_PATH)
        brand_avg_map = load_brand_avg_map(BRAND_AVG_PATH)
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.stop()

    # Derive UI options from the actual data
    brand_options = sorted(ref_df["brand"].dropna().unique().tolist())
    fuel_options  = sorted([
        col.replace("fuel_", "")
        for col in ref_df.columns if col.startswith("fuel_")
    ])
    ranges        = get_realistic_ranges(ref_df)
    global_mean   = brand_avg_map.get("__global_mean__", ref_df["selling_price"].mean())

    # Brand freq lookup (safe: uses X counts only, no target)
    brand_freq_map = ref_df["brand"].value_counts().to_dict()

    st.markdown(
        f"<div class='meta-line'>Model trained on {len(ref_df):,} listings · "
        f"{len(feature_cols)} features · {len(brand_options)} brands</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # --- Input form ---
    with st.container(border=True):
        st.markdown(
            "<div class='section-header'>📋 Car Details</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2, gap="large")

        with col1:
            km_driven = st.number_input(
                "🛣️ Kilometers Driven",
                min_value=0, max_value=2_000_000, value=50_000, step=1_000,
            )
            car_age = st.slider(
                "📅 Car Age (years)",
                min_value=0, max_value=max(ranges["car_age_max"], 30), value=5,
            )
            transmission_label = st.selectbox(
                "⚙️ Transmission", list(TRANSMISSION_LABELS.values())
            )

        with col2:
            owner_label = st.selectbox("👤 Owner", list(OWNER_LABELS.values()))
            fuel        = st.selectbox("⛽ Fuel Type", fuel_options)
            brand       = st.selectbox("🏷️ Brand", brand_options)

    transmission = [k for k, v in TRANSMISSION_LABELS.items() if v == transmission_label][0]
    owner        = [k for k, v in OWNER_LABELS.items()        if v == owner_label][0]

    # Look up brand_avg_price from the training-derived JSON map
    brand_avg_price = brand_avg_map.get(brand, global_mean)
    brand_freq_val  = brand_freq_map.get(brand, 1)

    st.write("")

    # --- Predict ---
    if st.button("💰 Predict Price", type="primary"):

        for w in validate_inputs(km_driven, car_age, ranges):
            st.warning(f"⚠️ {w}")

        try:
            input_df = build_input_dataframe(
                feature_columns = feature_cols,
                km_driven       = km_driven,
                car_age         = car_age,
                owner           = owner,
                transmission    = transmission,
                fuel            = fuel,
                brand           = brand,
                brand_avg_price = brand_avg_price,
            )

            # Set brand_freq separately (needs the lookup map)
            if "brand_freq" in input_df.columns:
                input_df.at[0, "brand_freq"] = brand_freq_val

            price     = predict_price(model, input_df)
            low, high = price * 0.85, price * 1.15

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

            with st.expander("📈 What's driving this prediction?"):
                importance = pd.DataFrame({
                    "Feature":    model.feature_names_in_,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=False).head(12)
                st.bar_chart(importance.set_index("Feature")["Importance"])

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


if __name__ == "__main__":
    main()