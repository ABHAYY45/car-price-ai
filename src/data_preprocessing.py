"""
data_preprocessing.py
----------------------
Cleaning + feature engineering pipeline for the CarDekho used-car dataset.

Phase 3 changes:
    REMOVED: city (was randomly generated noise)
    REMOVED: brand_popularity (was a hardcoded manual lookup, not data-driven)
    REMOVED: popularity_x_age (depended on brand_popularity)
    ADDED:   log_km_driven  (reduces right-skew in km_driven distribution)
    ADDED:   brand_freq     (how many times a brand appears -- demand proxy)
    KEPT:    km_per_year, age_km_interaction

IMPORTANT -- brand encoding is intentionally NOT done here:
    brand_avg_price (a target-encoded feature) must be computed AFTER the
    train/test split to avoid data leakage.  So we output 'brand' as a raw
    string column and let train_model.py handle one-hot encoding + avg-price
    mapping on training data only.

Run from the project root:
    python src/data_preprocessing.py

Outputs:
    data/processed_car_data.csv   <- ML-ready CSV (brand still raw string)
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# 1. LOAD
# --------------------------------------------------------------------------- #
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded '{filepath}' -> shape: {df.shape}")
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
    n = df.duplicated().sum()
    print(f"[INFO] Duplicate rows removed: {n}")
    return df.drop_duplicates()


def create_car_age(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    """Replace 'year' with car_age = current_year - year."""
    df = df.copy()
    if current_year is None:
        current_year = datetime.now().year
    df["car_age"] = (current_year - df["year"]).clip(lower=0)
    return df.drop(columns=["year"])


def _remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lo = max(q1 - multiplier * iqr, 0)
    hi = q3 + multiplier * iqr
    mask = df[column].between(lo, hi)
    print(f"[INFO] Outliers removed on '{column}': {(~mask).sum()} "
          f"(kept range [{lo:,.0f}, {hi:,.0f}])")
    return df[mask]


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df[(df["selling_price"] > 0) & (df["km_driven"] > 0)]
    print(f"[INFO] Rows removed (non-positive price/km): {before - len(df)}")
    df = _remove_outliers_iqr(df, "selling_price")
    df = _remove_outliers_iqr(df, "km_driven")
    return df


def clean_data(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    df = handle_missing_values(df)
    df = remove_duplicate_rows(df)
    df = create_car_age(df, current_year=current_year)
    df = remove_outliers(df)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
def extract_brand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pull the first word of 'name' as 'brand'.
    e.g. 'Maruti 800 AC' -> 'Maruti'

    NOTE: 'brand' is kept as a raw string here.  One-hot encoding and
    brand_avg_price computation happen in train_model.py AFTER the
    train/test split (to prevent data leakage from the target variable).
    """
    df = df.copy()
    df["brand"] = df["name"].astype(str).str.split().str[0]
    return df


def add_log_km_driven(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform km_driven to reduce its strong right skew.

    km_driven has a long tail -- a few cars with 300k+ km pull the
    distribution far right.  log1p compresses that tail so the model
    sees a roughly bell-shaped signal rather than an extreme outlier-
    dominated one.  log1p (= log(1 + x)) is used instead of log so
    that a value of 0 maps to 0 rather than -inf.

    WHY no leakage: this is a deterministic transform of an existing
    feature column (km_driven). It does not use selling_price at all.
    """
    df = df.copy()
    df["log_km_driven"] = np.log1p(df["km_driven"])
    return df


def add_brand_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many times each brand appears in the dataset.

    A brand that appears 800 times (Maruti) vs 12 times (Daewoo) carries
    very different market information -- high frequency implies wider
    availability, more competitive pricing, and faster resale.  This is
    a lightweight demand proxy derived purely from X, not from y.

    LEAKAGE NOTE: brand_freq is computed on the full dataset (train +
    test rows combined).  This is standard practice for count/frequency
    features because we are only using the distribution of X (brand
    names), NOT the target variable (selling_price).  There is no
    target leakage here.
    """
    df = df.copy()
    freq_map = df["brand"].value_counts().to_dict()
    df["brand_freq"] = df["brand"].map(freq_map)
    print(f"[INFO] brand_freq added. Unique brands: {df['brand'].nunique()}")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two proven interaction features from Phase 2 (kept because they
    contributed meaningfully to the feature importance rankings).

    km_per_year:
        80k km over 2 years signals much heavier wear than the same
        distance over 8 years.  Dividing normalises km_driven by age.
        +1 avoids division-by-zero for brand-new (age=0) cars.

    age_km_interaction:
        Multiplicative wear signal. Captures the combined depreciation
        effect that neither feature expresses alone.

    NOTE: popularity_x_age is intentionally removed here because it
    depended on brand_popularity, which we are removing in Phase 3.
    """
    df = df.copy()
    df["km_per_year"]        = df["km_driven"] / (df["car_age"] + 1)
    df["age_km_interaction"] = df["car_age"]   * df["km_driven"]
    print("[INFO] Interaction features added: km_per_year, age_km_interaction")
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns that are safe to encode on the full dataset:

        transmission -> binary (0 = Manual, 1 = Automatic)
        owner        -> ordinal (fewer owners = higher value)
        fuel         -> one-hot (nominal, no order)
        seller_type  -> one-hot (nominal, no order)

    'brand' is intentionally NOT encoded here.
    It stays as a raw string so train_model.py can:
        (a) compute brand_avg_price from training rows only, and
        (b) one-hot encode brand consistently across train and test sets.
    """
    df = df.copy()

    df["transmission"] = df["transmission"].map({"Manual": 0, "Automatic": 1})

    owner_order = {
        "Test Drive Car":       0,
        "First Owner":          1,
        "Second Owner":         2,
        "Third Owner":          3,
        "Fourth & Above Owner": 4,
    }
    df["owner"] = df["owner"].map(owner_order)
    df["owner"] = pd.to_numeric(df["owner"], errors="coerce").fillna(2).astype(int)

    # Only fuel and seller_type are one-hot encoded here.
    # brand is kept raw (see docstring above).
    df = pd.get_dummies(df, columns=["fuel", "seller_type"], drop_first=True)

    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 'name' -- its signal is already captured in 'brand'."""
    return df.drop(columns=[c for c in ["name"] if c in df.columns])


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = extract_brand(df)           # raw brand string (kept for train_model.py)
    df = add_log_km_driven(df)       # Phase 3: log transform
    df = add_brand_freq(df)          # Phase 3: frequency/demand proxy
    df = add_interaction_features(df)# Phase 2: km_per_year, age_km_interaction
    df = encode_categorical_features(df)
    df = drop_irrelevant_columns(df)
    return df


# --------------------------------------------------------------------------- #
# 4. REPORT
# --------------------------------------------------------------------------- #
def print_shape(df: pd.DataFrame, label: str) -> None:
    print(f"[SHAPE] {label}: {df.shape[0]} rows x {df.shape[1]} columns")


def display_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("NUMERIC COLUMNS IN FINAL DATASET")
    print("=" * 70)
    print(df.dtypes.to_string())
    print("\n" + "=" * 70)
    print("FIRST 3 ROWS")
    print("=" * 70)
    print(df.head(3).to_string())


# --------------------------------------------------------------------------- #
# 5. MAIN PIPELINE
# --------------------------------------------------------------------------- #
def run_pipeline(
    raw_filepath: str,
    processed_filepath: str = "data/processed_car_data.csv",
    current_year: int = None,
) -> pd.DataFrame:
    raw_df = load_data(raw_filepath)
    print_shape(raw_df, "BEFORE cleaning")

    cleaned_df = clean_data(raw_df, current_year=current_year)
    final_df   = feature_engineer(cleaned_df)

    print_shape(final_df, "AFTER cleaning + feature engineering")

    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    final_df.to_csv(processed_filepath, index=False)
    print(f"[INFO] Saved to '{processed_filepath}'")

    display_summary(final_df)
    return final_df


if __name__ == "__main__":
    run_pipeline(
        raw_filepath="data/car_data.csv",
        processed_filepath="data/processed_car_data.csv",
    )