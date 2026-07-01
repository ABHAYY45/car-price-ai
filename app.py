"""
app.py
------
Streamlit web app for predicting used-car selling prices using a trained
RandomForestRegressor.
 
Run with:
    streamlit run src/app.py
(run from the project root, so the relative paths below resolve correctly)
 
Expected project structure:
    data/processed_car_data.csv -> used to populate dropdown options & ranges
    models/car_price_model.pkl  -> the trained model
    src/app.py                  -> this file
"""
 
import joblib
import numpy as np
import pandas as pd
import streamlit as st
 
# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
MODEL_PATH = "models/car_price_model.pkl"
DATA_PATH = "data/processed_car_data.csv"
 
# Label maps for the dropdowns (must match how data_preprocessing.py
# encoded these columns -- adjust here if your encoding scheme differs).
OWNER_LABELS = {
    0: "0 - First Owner",
    1: "1 - Second Owner",
    2: "2 - Third Owner",
    3: "3 - Fourth & Above Owner",
}
TRANSMISSION_LABELS = {0: "Manual", 1: "Automatic"}
 
 
# --------------------------------------------------------------------------- #
# STEP 1: LOAD MODEL (cached so it's not reloaded on every interaction)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model(filepath: str):
    """Load the trained RandomForestRegressor model saved with joblib."""
    return joblib.load(filepath)
 
 
# --------------------------------------------------------------------------- #
# STEP 2: LOAD PROCESSED DATA (used ONLY to build the UI, not for alignment)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_reference_data(filepath: str) -> pd.DataFrame:
    """
    Load the processed training dataset.
 
    We use this purely to:
        - discover which fuel types / brands exist as dropdown options
          (by reading the one-hot encoded column names), and
        - compute realistic min/max ranges for km_driven and car_age
          (so we can warn the user about unrealistic inputs).
 
    NOTE: We do NOT use this file to decide the final column order fed
    into the model. That comes from the model itself in Step 3 below --
    relying on a CSV file for that is fragile, since the file on disk can
    drift out of sync with what the model was actually trained on.
    """
    return pd.read_csv(filepath)
 
 
def get_training_columns(model) -> list:
    """
    Get the exact feature column names -- and their order -- the model
    expects, straight from the model object itself. scikit-learn stores
    this automatically (`model.feature_names_in_`) when trained on a
    DataFrame, so this is always guaranteed to be correct and in sync.
    """
    return list(model.feature_names_in_)
 
 
def extract_categories(columns, prefix: str) -> list:
    """
    Given a list of column names and a prefix (e.g. "fuel_" or "brand_"),
    return the category names with the prefix stripped off.
    Example: "fuel_Petrol" -> "Petrol"
    """
    categories = [col[len(prefix):] for col in columns if col.startswith(prefix)]
    return sorted(categories)
 
 
def get_realistic_ranges(df: pd.DataFrame) -> dict:
    """
    Compute sensible min/max ranges from the training data, used to:
        - set slider bounds for car_age
        - flag unrealistic km_driven / car_age values entered by the user
    """
    ranges = {
        "km_driven_min": int(df["km_driven"].min()) if "km_driven" in df.columns else 0,
        "km_driven_max": int(df["km_driven"].max()) if "km_driven" in df.columns else 300000,
        "car_age_min": int(df["car_age"].min()) if "car_age" in df.columns else 0,
        "car_age_max": int(df["car_age"].max()) if "car_age" in df.columns else 30,
    }
    return ranges
 
 
# --------------------------------------------------------------------------- #
# STEP 3: BUILD A MODEL-READY INPUT ROW FROM USER SELECTIONS
# --------------------------------------------------------------------------- #
def build_input_dataframe(
    feature_columns: list,
    km_driven: int,
    car_age: int,
    owner: int,
    transmission: int,
    fuel: str,
    brand: str,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame that exactly matches the model's expected
    feature columns:
        1. Start with ALL feature columns set to 0.
        2. Fill in the plain numeric/ordinal features directly.
        3. Set the ONE matching fuel_<fuel> column to 1 (one-hot encoding).
        4. Set the ONE matching brand_<brand> column to 1 (one-hot encoding).
 
    Any column the model expects but that we don't explicitly set (e.g. a
    seller_type_* column, since this app doesn't collect that input) is
    safely left at 0 -- meaning "not this category" / baseline.
    """
    # 1. Start with a single row of all zeros, with the correct column names
    input_df = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)
 
    # 2. Fill in the directly-numeric columns, but ONLY if the model
    #    actually has that column (keeps this safe against schema changes)
    direct_values = {
        "km_driven": km_driven,
        "car_age": car_age,
        "owner": owner,
        "transmission": transmission,
    }
    for col_name, value in direct_values.items():
        if col_name in input_df.columns:
            input_df.at[0, col_name] = value
        else:
            st.warning(f"⚠️ Column '{col_name}' not found in the trained model -- skipped.")
 
    # 3. One-hot encode the selected fuel type
    fuel_col = f"fuel_{fuel}"
    if fuel_col in input_df.columns:
        input_df.at[0, fuel_col] = 1
    else:
        # This happens if `fuel` is the "baseline" category that was dropped
        # during one-hot encoding (drop_first=True) -- leaving all fuel_*
        # columns at 0 correctly represents that baseline category.
        st.info(f"ℹ️ '{fuel}' has no dedicated column -- treated as the baseline fuel category.")
 
    # 4. One-hot encode the selected brand
    brand_col = f"brand_{brand}"
    if brand_col in input_df.columns:
        input_df.at[0, brand_col] = 1
    else:
        st.info(f"ℹ️ '{brand}' has no dedicated column -- treated as the baseline brand category.")
 
    return input_df
 
 
# --------------------------------------------------------------------------- #
# STEP 4: VALIDATE INPUTS (warn, but don't block, on unrealistic values)
# --------------------------------------------------------------------------- #
def validate_inputs(km_driven: int, car_age: int, ranges: dict) -> list:
    """
    Check user inputs against the realistic range seen in the training
    data. Returns a list of warning messages (empty list = all good).
    We WARN rather than block, since the model can still produce a
    (less reliable) prediction outside this range.
    """
    warnings = []
 
    if km_driven < 0:
        warnings.append("km_driven cannot be negative.")
    elif km_driven > ranges["km_driven_max"]:
        warnings.append(
            f"km_driven ({km_driven:,}) is higher than anything seen in training "
            f"data (max ~{ranges['km_driven_max']:,} km). Prediction may be unreliable."
        )
 
    if car_age < 0:
        warnings.append("car_age cannot be negative.")
    elif car_age > ranges["car_age_max"]:
        warnings.append(
            f"car_age ({car_age} yrs) is higher than anything seen in training "
            f"data (max ~{ranges['car_age_max']} yrs). Prediction may be unreliable."
        )
 
    return warnings
 
 
# --------------------------------------------------------------------------- #
# STEP 5: PREDICT + FORMAT
# --------------------------------------------------------------------------- #
def predict_price(model, input_df: pd.DataFrame) -> float:
    """Run the model on the prepared input row and return the predicted price."""
    prediction = model.predict(input_df)
    return float(prediction[0])
 
 
def format_currency(amount: float) -> str:
    """Format a number as Indian Rupees with comma separators, e.g. ₹4,50,000.00."""
    return f"₹{amount:,.2f}"
 
 
# --------------------------------------------------------------------------- #
# STEP 6: CUSTOM CSS (presentation only -- no logic here)
# --------------------------------------------------------------------------- #
def inject_custom_css():
    """Inject custom CSS for a cleaner, more polished look and feel."""
    st.markdown(
        """
        <style>
            /* Center and constrain the main content for a polished look */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 850px;
            }
 
            /* Subtitle under the main title */
            .app-subtitle {
                text-align: center;
                color: #6b7280;
                font-size: 1.05rem;
                margin-top: -0.5rem;
                margin-bottom: 1.5rem;
            }
 
            /* Section headers with a little extra breathing room */
            .section-header {
                font-size: 1.15rem;
                font-weight: 600;
                margin-top: 0.5rem;
                margin-bottom: 0.75rem;
            }
 
            /* Make the predict button full-width and prominent */
            div.stButton > button {
                width: 100%;
                height: 3rem;
                font-size: 1.05rem;
                font-weight: 600;
                border-radius: 10px;
            }
 
            /* Highlighted prediction result card */
            .result-card {
                background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
                color: white;
                padding: 1.75rem;
                border-radius: 16px;
                text-align: center;
                margin-top: 1rem;
                box-shadow: 0 4px 14px rgba(22, 163, 74, 0.25);
            }
            .result-card .label {
                font-size: 0.95rem;
                opacity: 0.9;
                margin-bottom: 0.25rem;
            }
            .result-card .price {
                font-size: 2.2rem;
                font-weight: 700;
                margin: 0;
            }
 
            /* Disclaimer text */
            .disclaimer {
                text-align: center;
                color: #9ca3af;
                font-size: 0.82rem;
                margin-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
 
 
# --------------------------------------------------------------------------- #
# STEP 7: STREAMLIT UI
# --------------------------------------------------------------------------- #
def main():
    # --- Page setup ---
    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
    inject_custom_css()
 
    # --- Centered title + subtitle ---
    st.markdown("<h1 style='text-align:center;'>🚗 Used Car Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>Get an instant price estimate powered by a trained "
        "Random Forest model 🌳</div>",
        unsafe_allow_html=True,
    )
 
    # --- Load model + reference data (wrapped so the app never crashes silently) ---
    try:
        model = load_model(MODEL_PATH)
        feature_columns = get_training_columns(model)
    except FileNotFoundError:
        st.error(f"❌ Could not find the trained model at '{MODEL_PATH}'. "
                 "Make sure you've run train_model.py first.")
        st.stop()
 
    try:
        reference_df = load_reference_data(DATA_PATH)
        fuel_options = extract_categories(reference_df.columns, "fuel_")
        brand_options = extract_categories(reference_df.columns, "brand_")
        ranges = get_realistic_ranges(reference_df)
    except FileNotFoundError:
        st.error(f"❌ Could not find the processed dataset at '{DATA_PATH}'.")
        st.stop()
 
    if not fuel_options or not brand_options:
        st.warning("⚠️ No fuel/brand one-hot columns detected in the dataset. "
                   "Dropdowns may be incomplete.")
 
    # --- Input section, inside a bordered container for visual grouping ---
    with st.container(border=True):
        st.markdown("<div class='section-header'>📋 Car Details</div>", unsafe_allow_html=True)
 
        col1, col2 = st.columns(2, gap="large")
 
        with col1:
            km_driven = st.number_input(
                "🛣️ Kilometers Driven", min_value=0, max_value=2_000_000,
                value=50000, step=1000,
            )
            car_age = st.slider(
                "📅 Car Age (years)",
                min_value=0, max_value=max(ranges["car_age_max"], 30),
                value=min(5, ranges["car_age_max"]),
            )
            transmission_label = st.selectbox(
                "⚙️ Transmission", options=list(TRANSMISSION_LABELS.values())
            )
 
        with col2:
            owner_label = st.selectbox("👤 Owner", options=list(OWNER_LABELS.values()))
            fuel = st.selectbox("⛽ Fuel Type", options=fuel_options)
            brand = st.selectbox("🏷️ Brand", options=brand_options)
 
    # Convert the human-readable dropdown labels back to their numeric codes
    transmission = [k for k, v in TRANSMISSION_LABELS.items() if v == transmission_label][0]
    owner = [k for k, v in OWNER_LABELS.items() if v == owner_label][0]
 
    st.write("")  # small spacer
 
    # --- Prediction trigger ---
    predict_clicked = st.button("💰 Predict Price", type="primary")
 
    if predict_clicked:
        # Show warnings for unrealistic inputs, but still proceed with prediction
        warnings = validate_inputs(km_driven, car_age, ranges)
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
 
        try:
            input_df = build_input_dataframe(
                feature_columns=feature_columns,
                km_driven=km_driven,
                car_age=car_age,
                owner=owner,
                transmission=transmission,
                fuel=fuel,
                brand=brand,
            )
            predicted_price = predict_price(model, input_df)
 
            # --- Highlighted result card ---
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Estimated Selling Price</div>
                    <p class="price">{format_currency(predicted_price)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
            # --- Disclaimer ---
            st.markdown(
                "<div class='disclaimer'>⚠️ This is an estimate based on historical data "
                "and may not reflect the actual market price.</div>",
                unsafe_allow_html=True,
            )
 
        except Exception as e:
            # Catch-all so the app degrades gracefully instead of crashing
            st.error(f"❌ Something went wrong while predicting: {e}")
 
 
if __name__ == "__main__":
    main()
 