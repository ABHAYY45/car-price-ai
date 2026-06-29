"""
car_data_cleaning.py
---------------------
Data Cleaning + Feature Engineering pipeline for the CarDekho used-car dataset.
 
This script ONLY prepares the data for machine learning. It does NOT train
any model. It is meant to be imported (call `run_pipeline(path)`) or run
directly as a script.
 
Pipeline stages:
    1. Load data
    2. Clean data (missing values, duplicates, car_age, outliers)
    3. Feature-engineer categorical columns (encoding)
    4. Report shape / summary stats / preview
 
Author: (generated for Abhay's Car Price Prediction + Recommendation project)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
 
 
# --------------------------------------------------------------------------- #
# 1. DATA LOADING
# --------------------------------------------------------------------------- #
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CarDekho dataset from a CSV file into a pandas DataFrame.
 
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
 
    Returns
    -------
    pd.DataFrame
        Raw, unprocessed dataframe.
    """
    df = pd.read_csv(filepath)
    return df
 
 
# --------------------------------------------------------------------------- #
# 2. DATA CLEANING
# --------------------------------------------------------------------------- #
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in a generic, column-type-aware way:
        - Numeric columns  -> filled with the column MEDIAN (robust to outliers)
        - Categorical (object/string) columns -> filled with the column MODE
 
    This dataset currently has no missing values, but the logic is kept
    generic so the pipeline stays robust if a different/updated version
    of the dataset (or new scraped data) contains NaNs.
 
    Parameters
    ----------
    df : pd.DataFrame
 
    Returns
    -------
    pd.DataFrame
        Dataframe with missing values imputed.
    """
    df = df.copy()
 
    missing_before = df.isnull().sum()
    if missing_before.sum() == 0:
        print("[INFO] No missing values found.")
        return df
 
    print("[INFO] Missing values found before imputation:")
    print(missing_before[missing_before > 0])
 
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
 
        if pd.api.types.is_numeric_dtype(df[col]):
            # Median is preferred over mean since it's robust to skew/outliers
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            # For categorical/text columns, fill with the most frequent value
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
 
    return df
 
 
def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully duplicated rows from the dataframe.
 
    Parameters
    ----------
    df : pd.DataFrame
 
    Returns
    -------
    pd.DataFrame
        Dataframe without duplicate rows.
    """
    df = df.copy()
    n_duplicates = df.duplicated().sum()
    print(f"[INFO] Duplicate rows found: {n_duplicates}")
    df = df.drop_duplicates()
    return df
 
 
def create_car_age(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    """
    Convert the 'year' column into 'car_age' = current_year - year.
    Drops the original 'year' column afterwards.
 
    Parameters
    ----------
    df : pd.DataFrame
    current_year : int, optional
        Reference year used to compute age. Defaults to today's calendar year.
 
    Returns
    -------
    pd.DataFrame
        Dataframe with 'car_age' instead of 'year'.
    """
    df = df.copy()
 
    if current_year is None:
        current_year = datetime.now().year
 
    df["car_age"] = current_year - df["year"]
 
    # Defensive check: car_age should never be negative (year can't be in the future)
    df["car_age"] = df["car_age"].clip(lower=0)
 
    df = df.drop(columns=["year"])
    return df
 
 
def _remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Generic IQR-based outlier removal helper.
    Removes rows where `column` falls outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
 
    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column to filter on.
    multiplier : float
        IQR multiplier (1.5 = standard "mild" outlier threshold).
 
    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
 
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
 
    # A price/km value can never be negative -> floor the lower bound at 0
    lower_bound = max(lower_bound, 0)
 
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    removed = (~mask).sum()
    print(f"[INFO] Outliers removed on '{column}': {removed} "
          f"(valid range ~ [{lower_bound:,.0f}, {upper_bound:,.0f}])")
 
    return df[mask]
 
 
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unrealistic / extreme rows based on domain knowledge + statistics:
        - selling_price: must be positive, and within a statistically
          reasonable IQR-based range (removes extreme luxury / junk-data prices).
        - km_driven: must be positive, and within a statistically reasonable
          IQR-based range (removes odometer entry errors, e.g. 0 km or
          800,000+ km on a used car).
 
    Parameters
    ----------
    df : pd.DataFrame
 
    Returns
    -------
    pd.DataFrame
        Dataframe with outlier rows removed.
    """
    df = df.copy()
 
    # --- Hard domain-knowledge filters (sanity bounds) ---
    before = len(df)
    df = df[df["selling_price"] > 0]
    df = df[df["km_driven"] > 0]
    print(f"[INFO] Rows removed for non-positive price/km: {before - len(df)}")
 
    # --- Statistical IQR-based filtering ---
    df = _remove_outliers_iqr(df, "selling_price", multiplier=1.5)
    df = _remove_outliers_iqr(df, "km_driven", multiplier=1.5)
 
    return df
 
 
def clean_data(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    """
    Run the full cleaning pipeline:
        missing values -> duplicates -> car_age -> outliers
 
    Parameters
    ----------
    df : pd.DataFrame
    current_year : int, optional
 
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = handle_missing_values(df)
    df = remove_duplicate_rows(df)
    df = create_car_age(df, current_year=current_year)
    df = remove_outliers(df)
 
    # Reset index after all the row filtering above
    df = df.reset_index(drop=True)
    return df
 
 
# --------------------------------------------------------------------------- #
# 3. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
def extract_brand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the car 'brand' (first word of 'name'), e.g. "Maruti 800 AC" -> "Maruti".
    This is useful for the recommendation-system side of the project
    (recommending cars of similar brand/segment) and is far more usable
    as a model feature than the full free-text 'name' column.
 
    Parameters
    ----------
    df : pd.DataFrame
 
    Returns
    -------
    pd.DataFrame
        Dataframe with a new 'brand' column.
    """
    df = df.copy()
    df["brand"] = df["name"].astype(str).str.split().str[0]
    return df
 
 
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using the most appropriate strategy for each:
 
        - transmission (binary: Manual/Automatic)
              -> Label encoding (0/1), since it's already binary.
        - owner (ordinal: First < Second < Third < Fourth & Above, + Test Drive Car)
              -> Label/ordinal encoding, since there is a natural ranking
                 (fewer previous owners == generally higher value).
        - fuel, seller_type (nominal, no inherent order)
              -> One-Hot encoding, since label encoding would wrongly imply
                 an order between categories like Petrol/Diesel/CNG.
        - brand (nominal, high-cardinality)
              -> One-Hot encoding (kept, as it can be informative for both
                 price prediction and recommendation).
 
    Parameters
    ----------
    df : pd.DataFrame
 
    Returns
    -------
    pd.DataFrame
        Dataframe with all categorical columns numerically encoded.
    """
    df = df.copy()
 
    # --- Binary label encoding: transmission ---
    df["transmission"] = df["transmission"].map({"Manual": 0, "Automatic": 1})
 
    # --- Ordinal label encoding: owner ---
    owner_order = {
        "Test Drive Car": 0,
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
    }
    df["owner"] = df["owner"].map(owner_order)
    # Any unseen/unexpected category falls back to the median rank
    df["owner"] = df["owner"].fillna(df["owner"].median())
 
    # --- One-Hot encoding: nominal categorical columns ---
    nominal_cols = [c for c in ["fuel", "seller_type", "brand"] if c in df.columns]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
 
    return df
 
 
def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not useful as direct ML features.
    'name' is dropped because it's free text / extremely high-cardinality;
    its useful signal (brand) has already been extracted separately.
 
    Parameters
    ----------
    df : pd.DataFrame
 
    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    cols_to_drop = [c for c in ["name"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df
 
 
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature-engineering pipeline:
        extract brand -> encode categoricals -> drop irrelevant columns
 
    Parameters
    ----------
    df : pd.DataFrame
 
    Returns
    -------
    pd.DataFrame
        ML-ready dataframe.
    """
    df = extract_brand(df)
    df = encode_categorical_features(df)
    df = drop_irrelevant_columns(df)
    return df
 
 
# --------------------------------------------------------------------------- #
# 4. REPORTING / OUTPUT
# --------------------------------------------------------------------------- #
def print_shape(df: pd.DataFrame, label: str) -> None:
    """Print the shape of the dataframe with a descriptive label."""
    print(f"[SHAPE] {label}: {df.shape[0]} rows x {df.shape[1]} columns")
 
 
def display_summary(df: pd.DataFrame) -> None:
    """Print summary statistics and the first 5 rows of the dataframe."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (numeric columns)")
    print("=" * 70)
    # include='all' would mix numeric + bool dummy columns awkwardly,
    # so we keep this focused on numeric describe() for readability.
    print(df.describe().T)
 
    print("\n" + "=" * 70)
    print("FIRST 5 ROWS")
    print("=" * 70)
    print(df.head())
 
 
# --------------------------------------------------------------------------- #
# 5. MAIN PIPELINE
# --------------------------------------------------------------------------- #
def run_pipeline(filepath: str, current_year: int = None) -> pd.DataFrame:
    # Stage 1: Load
    raw_df = load_data(filepath)
    print_shape(raw_df, "BEFORE cleaning")

    # Stage 2: Clean
    cleaned_df = clean_data(raw_df, current_year=current_year)

    # Stage 3: Feature engineer
    final_df = feature_engineer(cleaned_df)
    print_shape(final_df, "AFTER cleaning + feature engineering")

    # Stage 4: Report
    display_summary(final_df)

    #Save file
    output_path = "data/processed_car_data.csv"
    os.makedirs("data", exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"[INFO] Processed data saved at: {output_path}")

    return final_df
 
 
if __name__ == "__main__":
    # Change this path if your CSV lives elsewhere
    DATA_PATH = "data/car_data.csv"
 
    final_dataframe = run_pipeline(DATA_PATH)
