"""
data_preprocessing.py
----------------------
Full cleaning + feature-engineering pipeline for the new CarDekho dataset.

New 13-column schema:
    car_name, brand, model, vehicle_age, km_driven, seller_type,
    fuel_type, transmission_type, mileage, engine, max_power,
    seats, selling_price

Run from the project root:
    python src/data_preprocessing.py

Output:
    data/processed_car_data.csv   <- fully numeric, ML-ready dataset
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
INPUT_PATH  = "data/new_car_data.csv"
OUTPUT_PATH = "data/processed_car_data.csv"

UNIT_COLUMNS = {
    "mileage":   "kmpl",
    "engine":    "cc",
    "max_power": "bhp",
}

CATEGORICAL_COLUMNS = ["fuel_type", "seller_type", "transmission_type"]
DROP_COLUMNS        = ["car_name", "brand", "model"]


# --------------------------------------------------------------------------- #
# STEP 1 — LOAD
# --------------------------------------------------------------------------- #
def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV and print a quick sanity check."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: '{filepath}'")
        print("        Place your dataset at data/new_car_data.csv and retry.")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded '{filepath}' — {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# --------------------------------------------------------------------------- #
# STEP 2 — STANDARDISE COLUMN NAMES
# --------------------------------------------------------------------------- #
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase all column names and replace spaces/dashes with underscores.
    e.g. 'Fuel Type' -> 'fuel_type', 'Max Power' -> 'max_power'

    Done first so every downstream function references stable names
    regardless of how the source CSV was exported.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    print(f"[INFO] Columns standardised: {df.columns.tolist()}")
    return df


# --------------------------------------------------------------------------- #
# STEP 3 — HANDLE MISSING VALUES
# --------------------------------------------------------------------------- #
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Targeted missing-value strategy per column group:

        selling_price          -> DROP row: can't train without the target.
        engine, max_power,
        mileage                -> DROP row: high-signal features; imputing
                                  engine/power from median is misleading
                                  (a 1000cc and 2000cc car cannot share
                                  the same imputed value meaningfully).
        seats                  -> FILL with mode (5 covers ~80% of cars).
        vehicle_age, km_driven -> FILL with median (safe for continuous cols).
    """
    df   = df.copy()
    before = len(df)

    # Target variable — must exist
    df = df.dropna(subset=["selling_price"])
    print(f"[INFO] Dropped {before - len(df)} rows with null selling_price")

    # Critical numeric features — must exist (unit strings parsed in step 4,
    # but missing entirely means we drop now rather than later)
    critical = [c for c in ["engine", "max_power", "mileage"] if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=critical)
    print(f"[INFO] Dropped {n_before - len(df)} rows with null engine/max_power/mileage")

    # seats — fill with mode
    if "seats" in df.columns and df["seats"].isnull().any():
        mode_val = df["seats"].mode(dropna=True).iloc[0]
        n_filled = df["seats"].isnull().sum()
        df["seats"] = df["seats"].fillna(mode_val)
        print(f"[INFO] Filled {n_filled} null 'seats' with mode ({mode_val})")

    # vehicle_age, km_driven — fill with median
    for col in ["vehicle_age", "km_driven"]:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            n_filled   = df[col].isnull().sum()
            df[col]    = df[col].fillna(median_val)
            print(f"[INFO] Filled {n_filled} null '{col}' with median ({median_val:.0f})")

    df = df.reset_index(drop=True)
    print(f"[INFO] Rows after missing-value handling: {len(df):,}  "
          f"(removed {before - len(df):,} total)")
    return df


# --------------------------------------------------------------------------- #
# STEP 4 — CLEAN NUMERIC COLUMNS
# --------------------------------------------------------------------------- #
def _strip_unit(series: pd.Series, unit: str) -> pd.Series:
    """
    Strip a unit suffix and convert to numeric.

    "19.7 kmpl" -> 19.7  |  "1248 CC" -> 1248.0  |  "88.5 bhp" -> 88.5

    Unparseable values are coerced to NaN (errors="coerce") so a single
    bad row never aborts the whole pipeline.  They are dropped after
    all columns are processed.
    """
    return (
        series
        .astype(str)
        .str.lower()
        .str.replace(unit, "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
        .pipe(pd.to_numeric, errors="coerce")
    )


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse all unit-suffixed string columns into plain numeric values,
    then drop any rows where conversion failed.
    """
    df = df.copy()

    for col, unit in UNIT_COLUMNS.items():
        if col not in df.columns:
            print(f"[WARNING] Expected column '{col}' not found — skipping.")
            continue

        df[col] = _strip_unit(df[col], unit)
        lo, hi  = df[col].min(), df[col].max()
        nulls   = df[col].isnull().sum()

        if nulls:
            print(f"[WARNING] '{col}': {nulls} values could not be parsed "
                  "(will be dropped).")
        print(f"[INFO] '{col}' cleaned — range [{lo:.1f}, {hi:.1f}]")

    # engine values are always whole numbers (cubic centimetres)
    if "engine" in df.columns:
        df["engine"] = df["engine"].round(0).astype("Int64")  # nullable int

    # Drop rows that still have NaN after conversion
    numeric_cols = [c for c in UNIT_COLUMNS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[INFO] Dropped {removed} rows with unparseable numeric values.")

    return df


# --------------------------------------------------------------------------- #
# STEP 5 — FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive three features from existing columns.

    km_per_year
        Normalises usage by age. 80k km in 2 years signals far more wear
        than 80k km in 8 years. +1 avoids division-by-zero for new cars.

    power_to_engine
        bhp-per-CC is a compact proxy for engine modernity and tuning.
        A modern 1.0L turbo (120 bhp / 1000 CC = 0.12) vs an older 1.6L
        naturally-aspirated (90 bhp / 1600 CC = 0.056) — a single number
        that captures what engine size and power alone cannot.

    log_km_driven
        km_driven has a strong right skew. Log1p compresses the long tail
        so the model sees a roughly symmetric distribution and extreme
        high-km outliers don't dominate.
    """
    df = df.copy()

    df["km_per_year"]     = df["km_driven"] / (df["vehicle_age"] + 1)
    df["power_to_engine"] = df["max_power"]  / df["engine"].astype(float)
    df["log_km_driven"]   = np.log1p(df["km_driven"])

    print("[INFO] Features added: km_per_year, power_to_engine, log_km_driven")
    return df


# --------------------------------------------------------------------------- #
# STEP 6 — ENCODE CATEGORICALS
# --------------------------------------------------------------------------- #
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all nominal categorical columns.

    drop_first=True removes one dummy per group to avoid the dummy variable
    trap (perfect multicollinearity):
        fuel_type         -> baseline = first alphabetical (e.g. CNG or Diesel)
        seller_type       -> baseline = Dealer
        transmission_type -> baseline = Automatic

    Column names are lowercased and underscored after encoding for
    consistency with the rest of the pipeline.
    """
    df = df.copy()
    cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    missing = set(CATEGORICAL_COLUMNS) - set(cols)
    if missing:
        print(f"[WARNING] Categorical columns not found, skipped: {missing}")

    df = pd.get_dummies(df, columns=cols, drop_first=True)

    # Normalise any capitalised dummy column names pandas might produce
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    new_dummies = [c for c in df.columns
                   if any(c.startswith(base) for base in cols)]
    print(f"[INFO] One-hot encoded {cols} -> new columns: {new_dummies}")
    return df


# --------------------------------------------------------------------------- #
# STEP 7 — DROP UNNECESSARY COLUMNS
# --------------------------------------------------------------------------- #
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove high-cardinality text columns that would add noise, not signal:

        car_name  -> near-unique free text; can't be encoded sensibly
        brand     -> signal already embedded in engine/power/mileage patterns
        model     -> same reasoning; too many sparse categories
    """
    df = df.copy()
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"[INFO] Dropped columns: {to_drop}")
    return df


# --------------------------------------------------------------------------- #
# STEP 8 — FINAL VALIDATION
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard gate before saving. Two requirements must both pass:
        1. Zero NaN values anywhere in the dataframe.
        2. Every column is numeric (int64 or float64).

    If NaNs remain they are dropped with a warning.
    If non-numeric columns remain the pipeline exits so you know exactly
    what broke rather than saving a silently broken dataset.
    """
    df = df.copy()

    # --- NaN check ---
    nan_counts = df.isnull().sum()
    bad_cols   = nan_counts[nan_counts > 0]
    if not bad_cols.empty:
        print("[WARNING] NaN values remain — dropping affected rows:")
        for col, n in bad_cols.items():
            print(f"    {col}: {n} NaNs")
        df = df.dropna().reset_index(drop=True)
        print(f"[INFO] Rows after final NaN drop: {len(df):,}")

    # --- Dtype check ---
    non_numeric = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        print(f"[ERROR] Non-numeric columns remain: {non_numeric}")
        print("        These must be encoded or dropped before training.")
        sys.exit(1)

    # Ensure boolean dummies and nullable ints are cast to plain int64
    # (scikit-learn and XGBoost both require float64 / int64, not bool/Int64)
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("int64")
        elif str(df[col].dtype) == "Int64":
            df[col] = df[col].astype("int64")

    print(f"[INFO] Validation passed — {df.shape[0]:,} rows, "
          f"{df.shape[1]} columns, zero NaNs, all numeric.")
    return df


# --------------------------------------------------------------------------- #
# STEP 9 — SAVE
# --------------------------------------------------------------------------- #
def save_data(df: pd.DataFrame, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[INFO] Saved -> '{filepath}'")


# --------------------------------------------------------------------------- #
# REPORT
# --------------------------------------------------------------------------- #
def print_report(df: pd.DataFrame) -> None:
    key_cols = [c for c in [
        "selling_price", "km_driven", "vehicle_age",
        "mileage", "engine", "max_power",
        "km_per_year", "power_to_engine",
    ] if c in df.columns]

    print("\n" + "=" * 65)
    print("FINAL DATASET REPORT")
    print("=" * 65)
    print(f"Shape   : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"\nAll columns:")
    for col in df.columns:
        print(f"    {col:<40} {str(df[col].dtype):<10}")
    print("\nKey statistics:")
    print(df[key_cols].describe().T[["mean", "min", "max"]].round(2).to_string())
    print("=" * 65)


# --------------------------------------------------------------------------- #
# MAIN PIPELINE
# --------------------------------------------------------------------------- #
def run_pipeline(
    input_path:  str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Run all stages in order. Importable by train_model.py."""
    print("\n" + "=" * 65)
    print(f"DATA PREPROCESSING  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 65 + "\n")

    df = load_data(input_path)           # 1. Load
    df = standardise_columns(df)        # 2. Standardise column names
    df = handle_missing_values(df)      # 3. Drop / fill missing values
    df = clean_numeric_columns(df)      # 4. Strip unit suffixes -> numeric
    df = add_features(df)               # 5. Feature engineering
    df = encode_categoricals(df)        # 6. One-hot encode categoricals
    df = drop_columns(df)               # 7. Drop noisy text columns
    df = validate(df)                   # 8. NaN + dtype final check
    save_data(df, output_path)          # 9. Save

    print_report(df)
    return df


def main():
    run_pipeline(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()