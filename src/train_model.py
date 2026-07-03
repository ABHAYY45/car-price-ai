"""
train_model.py
--------------
Trains and evaluates two regression models on the processed CarDekho dataset,
then saves the best-performing one to disk.

Models compared:
    1. LinearRegression    (fast baseline — sets the floor)
    2. RandomForestRegressor (ensemble — usually beats linear on tabular data)

Run from the project root:
    python src/train_model.py

Input  : data/processed_car_data.csv
Output : models/car_price_model.pkl
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DATA_PATH    = "data/processed_car_data.csv"
MODEL_PATH   = "models/car_price_model.pkl"
TARGET       = "selling_price"
TEST_SIZE    = 0.2
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# STEP 1 — LOAD DATA
# --------------------------------------------------------------------------- #
def load_data(filepath: str) -> pd.DataFrame:
    """Load the processed CSV and drop the index column if present."""
    df = pd.read_csv(filepath)

    # Drop pandas index column that appears when a CSV is saved with index=True
    unnamed = [c for c in df.columns if "unnamed" in c.lower()]
    if unnamed:
        df = df.drop(columns=unnamed)
        print(f"[INFO] Dropped index column(s): {unnamed}")

    print(f"[INFO] Loaded '{filepath}' — {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# --------------------------------------------------------------------------- #
# STEP 2 — SPLIT FEATURES AND TARGET
# --------------------------------------------------------------------------- #
def get_X_y(df: pd.DataFrame, target: str):
    """Separate features (X) from the target variable (y)."""
    X = df.drop(columns=[target])
    y = df[target]
    print(f"[INFO] Features: {X.shape[1]} columns | Target: '{target}'")
    return X, y


# --------------------------------------------------------------------------- #
# STEP 3 — TRAIN / TEST SPLIT
# --------------------------------------------------------------------------- #
def split(X, y, test_size: float, random_state: int):
    """80 / 20 train-test split with a fixed seed for reproducibility."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# STEP 4 — TRAIN
# --------------------------------------------------------------------------- #
def train_linear(X_train, y_train) -> LinearRegression:
    """
    Fit a simple LinearRegression.

    Acts as a baseline. If Random Forest barely beats it, your features
    are already capturing the relationship well. If it beats Linear by
    a large margin, there are significant non-linearities in the data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[INFO] LinearRegression trained.")
    return model


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    """
    Fit a RandomForestRegressor with 100 trees.

    n_jobs=-1  -> uses all CPU cores for faster training
    random_state -> fixed seed for reproducibility
    """
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[INFO] RandomForestRegressor trained.")
    return model


# --------------------------------------------------------------------------- #
# STEP 5 — EVALUATE
# --------------------------------------------------------------------------- #
def evaluate(model, X_test, y_test) -> dict:
    """
    Compute three standard regression metrics on the held-out test set.

        R²   -> proportion of variance explained (1.0 = perfect).
        MAE  -> average absolute error in the same units as selling_price.
        RMSE -> like MAE but penalises large errors more heavily.
    """
    y_pred = model.predict(X_test)
    return {
        "R2":   r2_score(y_test, y_pred),
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def print_results(name: str, metrics: dict) -> None:
    """Print metrics in the format requested."""
    print(f"\n{name}:")
    print(f"  R²   : {metrics['R2']:.4f}")
    print(f"  MAE  : ₹{metrics['MAE']:>12,.2f}")
    print(f"  RMSE : ₹{metrics['RMSE']:>12,.2f}")


# --------------------------------------------------------------------------- #
# STEP 6 — SELECT BEST MODEL
# --------------------------------------------------------------------------- #
def best_model(models: dict) -> tuple:
    """
    Pick the model with the highest R² score.
    Returns (name, model_object).
    """
    winner = max(models, key=lambda name: models[name]["metrics"]["R2"])
    return winner, models[winner]["model"]


# --------------------------------------------------------------------------- #
# STEP 7 — SAVE
# --------------------------------------------------------------------------- #
def save_model(model, filepath: str) -> None:
    """Persist the trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\n[INFO] Best model saved -> '{filepath}'")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    print("\n" + "=" * 50)
    print("MODEL TRAINING PIPELINE")
    print("=" * 50)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Features / target
    X, y = get_X_y(df, TARGET)

    # 3. Split
    X_train, X_test, y_train, y_test = split(X, y, TEST_SIZE, RANDOM_STATE)

    # 4. Train both models
    lr = train_linear(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    # 5. Evaluate both
    lr_metrics = evaluate(lr, X_test, y_test)
    rf_metrics = evaluate(rf, X_test, y_test)

    # 6. Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print_results("Linear Regression", lr_metrics)
    print_results("Random Forest",     rf_metrics)

    # 7. Pick winner
    models = {
        "Linear Regression": {"model": lr, "metrics": lr_metrics},
        "Random Forest":     {"model": rf, "metrics": rf_metrics},
    }
    winner_name, winner_model = best_model(models)

    print("\n" + "=" * 50)
    print(f"BEST MODEL: {winner_name}  (R² = {models[winner_name]['metrics']['R2']:.4f})")
    print("=" * 50)

    # 8. Save best model
    save_model(winner_model, MODEL_PATH)


if __name__ == "__main__":
    main()