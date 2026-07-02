"""
data_preprocessing.py
----------------------
Full cleaning + feature-engineering pipeline for the CarDekho used-car dataset.

NEW in Phase 1:
    - assign_city()         : synthetically adds a city column (6 Indian cities)
    - fetch_brand_popularity(): fetches Google Trends scores via Pytrends
                                with a hardcoded fallback for server environments
    - add_brand_popularity() : merges popularity score as a numeric feature

Run from the project root:
    python src/data_preprocessing.py

Output:
    data/processed_car_data.csv   <- ML-ready encoded dataset
    data/brand_popularity.json    <- popularity map used by the Streamlit app
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"]
RANDOM_STATE = 42

# Realistic fallback popularity scores (0-100) based on Google Trends India.
# Used automatically when the live Pytrends API call fails (e.g. on servers).
# Higher score = more consumer search interest = stronger brand demand signal.
BRAND_POPULARITY_FALLBACK = {
    "Maruti":        95,
    "Hyundai":       88,
    "Tata":          82,
    "Mahindra":      80,
    "Honda":         72,
    "Toyota":        70,
    "Kia":           68,
    "Ford":          55,
    "Renault":       50,
    "Volkswagen":    48,
    "Skoda":         45,
    "Nissan":        42,
    "MG":            60,
    "Jeep":          52,
    "BMW":           40,
    "Audi":          38,
    "Mercedes-Benz": 35,
    "Volvo":         28,
    "Mitsubishi":    30,
    "Datsun":        25,
    "Fiat":          20,
    "Chevrolet":     22,
    "Isuzu":         18,
    "Force":         15,
    "Jaguar":        22,
    "Land":          30,
    "Daewoo":        10,
    "OpelCorsa":     10,
    "Ambassador":     8,
}

# --------------------------------------------------------------------------- #
# 1. LOAD
# --------------------------------------------------------------------------- #
def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CarDekho CSV."""
    df = pd.read_csv(filepath)
    return df


# --------------------------------------------------------------------------- #
# 2. CLEANING
# --------------------------------------------------------------------------- #
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    df = df.copy()
    if df.isnull().sum().sum() == 0:
        print("[INFO] No missing values found.")
        return df
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicated rows."""
    n = df.duplicated().sum()
    print(f"[INFO] Duplicate rows removed: {n}")
    return df.drop_duplicates()


def create_car_age(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    """Replace 'year' with 'car_age' = current_year - year."""
    df = df.copy()
    if current_year is None:
        current_year = datetime.now().year
    df["car_age"] = (current_year - df["year"]).clip(lower=0)
    return df.drop(columns=["year"])


def _remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lo, hi = max(q1 - multiplier * iqr, 0), q3 + multiplier * iqr
    mask = df[column].between(lo, hi)
    print(f"[INFO] Outliers removed on '{column}': {(~mask).sum()} "
          f"(range [{lo:,.0f}, {hi:,.0f}])")
    return df[mask]


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with non-positive or statistically extreme price/km values."""
    df = df.copy()
    before = len(df)
    df = df[(df["selling_price"] > 0) & (df["km_driven"] > 0)]
    print(f"[INFO] Rows removed for non-positive price/km: {before - len(df)}")
    df = _remove_outliers_iqr(df, "selling_price")
    df = _remove_outliers_iqr(df, "km_driven")
    return df


def clean_data(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    """Run the full cleaning pipeline."""
    df = handle_missing_values(df)
    df = remove_duplicate_rows(df)
    df = create_car_age(df, current_year=current_year)
    df = remove_outliers(df)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3a. CITY FEATURE (Phase 1 - new)
# --------------------------------------------------------------------------- #
def assign_city(df: pd.DataFrame, cities: list = CITIES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Synthetically assign a city to each listing.

    Why synthetic? The CarDekho dataset doesn't include city. We assign
    cities randomly with a fixed seed so results are reproducible. This
    adds the city dimension to the model so the Streamlit app can accept
    city as an input and the model has learned city-level price patterns.

    In a future scraping phase this column will be replaced by real city
    data scraped directly from listings.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)
    df["city"] = rng.choice(cities, size=len(df))
    print(f"[INFO] City assigned to {len(df)} rows from: {cities}")
    return df


# --------------------------------------------------------------------------- #
# 3b. BRAND POPULARITY FEATURE (Phase 1 - new)
# --------------------------------------------------------------------------- #
def fetch_brand_popularity(brands: list, geo: str = "IN", timeframe: str = "today 12-m") -> dict:
    """
    Fetch Google Trends interest scores for each brand via Pytrends.

    Scores are 0-100 (relative search interest in India over the last 12
    months). Higher = more consumer demand signal.

    Falls back to BRAND_POPULARITY_FALLBACK for any brand where the API
    call fails (rate limits, CAPTCHA, server IP blocks, etc.).
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("[WARNING] pytrends not installed. Using fallback popularity scores.")
        return {b: BRAND_POPULARITY_FALLBACK.get(b, 50) for b in brands}

    pytrends = TrendReq(hl="en-US", tz=330)  # tz=330 -> IST (UTC+5:30)
    popularity = {}

    print("[INFO] Fetching brand popularity from Google Trends...")
    for brand in brands:
        keyword = f"{brand} car"
        try:
            pytrends.build_payload([keyword], geo=geo, timeframe=timeframe)
            data = pytrends.interest_over_time()
            if not data.empty and keyword in data.columns:
                score = round(float(data[keyword].mean()), 2)
                popularity[brand] = score
                print(f"  [Trends] {brand}: {score}")
            else:
                # Empty response -- use fallback
                popularity[brand] = BRAND_POPULARITY_FALLBACK.get(brand, 50)
                print(f"  [Fallback] {brand}: {popularity[brand]} (empty Trends response)")
            time.sleep(1.5)   # polite delay to avoid rate limiting

        except Exception as e:
            popularity[brand] = BRAND_POPULARITY_FALLBACK.get(brand, 50)
            print(f"  [Fallback] {brand}: {popularity[brand]} ({e})")

    return popularity


def add_brand_popularity(df: pd.DataFrame, popularity_map: dict) -> pd.DataFrame:
    """
    Merge the brand popularity score into the dataframe as a numeric column.

    WHY no data leakage: the popularity score comes from Google Trends
    (an external signal), NOT from selling_price. It captures real-world
    demand for a brand, which is a genuine price driver.
    """
    df = df.copy()
    # Map each row's brand to its score; default to 50 if brand not in map
    df["brand_popularity"] = df["brand"].map(popularity_map).fillna(50).astype(float)
    return df


def save_popularity_map(popularity_map: dict, filepath: str) -> None:
    """
    Save the brand → score mapping to JSON so the Streamlit app can look
    up a user-selected brand's popularity at prediction time without
    needing to re-run Pytrends.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(popularity_map, f, indent=2)
    print(f"[INFO] Brand popularity map saved to '{filepath}'")


# --------------------------------------------------------------------------- #
# 4. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
def extract_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Extract first word of 'name' as 'brand' (e.g. 'Maruti 800 AC' -> 'Maruti')."""
    df = df.copy()
    df["brand"] = df["name"].astype(str).str.split().str[0]
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical columns:
        transmission -> binary label (0=Manual, 1=Automatic)
        owner        -> ordinal label (fewer owners = higher value)
        fuel, seller_type, brand, city -> one-hot encoded
    """
    df = df.copy()

    df["transmission"] = df["transmission"].map({"Manual": 0, "Automatic": 1})

    owner_order = {
        "Test Drive Car": 0,
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
    }
    df["owner"] = df["owner"].map(owner_order)
    df["owner"] = pd.to_numeric(df["owner"], errors="coerce").fillna(2).astype(int)

    # city is nominal (no inherent order) -> one-hot encode
    nominal_cols = [c for c in ["fuel", "seller_type", "brand", "city"] if c in df.columns]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 'name' -- useful signal already captured in 'brand' and 'brand_popularity'."""
    return df.drop(columns=[c for c in ["name"] if c in df.columns])


# --------------------------------------------------------------------------- #
# 4b. INTERACTION FEATURES (Phase 2 - new)
# --------------------------------------------------------------------------- #
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create three engineered interaction features from existing columns.
    These expose non-linear relationships that XGBoost can exploit much
    better than raw features alone.

    km_per_year:
        80k km in 2 years is far more wear than 80k km in 8 years.
        Raw km_driven alone cannot express this -- dividing by age does.
        +1 in the denominator avoids division-by-zero for brand-new cars.

    age_km_interaction:
        Multiplicative wear signal. A 15-year-old car with 200k km is
        much more depreciated than either feature implies individually.
        XGBoost can find this split; giving it the product makes the
        signal explicit and speeds up learning.

    popularity_x_age:
        Brand demand decays with age differently per brand. A new Maruti
        (popularity=95) benefits far more from brand recognition than a
        12-year-old one. Multiplying captures this joint effect.

    NOTE: call this AFTER brand_popularity and car_age exist in df,
    and BEFORE one-hot encoding (order relative to encoding does not
    matter since these are numeric, but being explicit is good practice).
    """
    df = df.copy()
    df["km_per_year"]        = df["km_driven"] / (df["car_age"] + 1)
    df["age_km_interaction"] = df["car_age"]   * df["km_driven"]
    df["popularity_x_age"]   = df["brand_popularity"] * df["car_age"]
    print("[INFO] Interaction features added: km_per_year, age_km_interaction, popularity_x_age")
    return df


def feature_engineer(df: pd.DataFrame, popularity_map: dict) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = extract_brand(df)
    df = assign_city(df)                          # Phase 1: city
    df = add_brand_popularity(df, popularity_map) # Phase 1: brand popularity
    df = add_interaction_features(df)             # Phase 2: interaction features
    df = encode_categorical_features(df)
    df = drop_irrelevant_columns(df)
    return df


# --------------------------------------------------------------------------- #
# 5. OUTPUT / REPORT
# --------------------------------------------------------------------------- #
def print_shape(df: pd.DataFrame, label: str) -> None:
    print(f"[SHAPE] {label}: {df.shape[0]} rows x {df.shape[1]} columns")


def display_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(df.describe().T)
    print("\n" + "=" * 70)
    print("FIRST 5 ROWS")
    print("=" * 70)
    print(df.head())


# --------------------------------------------------------------------------- #
# 6. MAIN PIPELINE
# --------------------------------------------------------------------------- #
def run_pipeline(
    raw_filepath: str,
    processed_filepath: str = "data/processed_car_data.csv",
    popularity_filepath: str = "data/brand_popularity.json",
    current_year: int = None,
) -> pd.DataFrame:
    """
    Full pipeline: load -> clean -> enrich (city + popularity) -> encode -> save.

    Parameters
    ----------
    raw_filepath        : path to raw CarDekho CSV
    processed_filepath  : where to save the ML-ready CSV
    popularity_filepath : where to save the brand popularity JSON
    current_year        : override for car_age calculation
    """
    # Load
    raw_df = load_data(raw_filepath)
    print_shape(raw_df, "BEFORE cleaning")

    # Clean
    cleaned_df = clean_data(raw_df, current_year=current_year)

    # Fetch popularity BEFORE feature engineering (needs the 'brand' column extracted)
    temp_df = extract_brand(cleaned_df)
    brands = temp_df["brand"].unique().tolist()
    popularity_map = fetch_brand_popularity(brands)
    save_popularity_map(popularity_map, popularity_filepath)

    # Feature engineer (city + popularity + encoding)
    final_df = feature_engineer(cleaned_df, popularity_map)
    print_shape(final_df, "AFTER cleaning + feature engineering")

    # Save
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    final_df.to_csv(processed_filepath, index=False)
    print(f"[INFO] Processed data saved to '{processed_filepath}'")

    display_summary(final_df)
    return final_df


if __name__ == "__main__":
    run_pipeline(
        raw_filepath="data/car_data.csv",
        processed_filepath="data/processed_car_data.csv",
        popularity_filepath="data/brand_popularity.json",
    )