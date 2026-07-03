"""
train_model.py
--------------
Trains an XGBoostRegressor on the processed CarDekho dataset.

Phase 3 changes:
    - Computes brand_avg_price AFTER train/test split (no data leakage)
    - Saves the brand -> avg_price mapping to data/brand_avg_price.json
      so app.py can look up any selected brand at prediction time
    - One-hot encodes 'brand' column here (not in preprocessing) so that
      brand_avg_price can be computed from training rows only

LEAKAGE PREVENTION (the key section):
    brand_avg_price is a target-encoded feature -- it uses selling_price
    (the target variable) to compute the mean price per brand.  If we
    computed it on the full dataset before splitting, test-set prices
    would bleed into a training feature, artificially inflating R².

    Fix:
        1. Split into train / test FIRST.
        2. Compute mean selling_price per brand using ONLY training rows.
        3. Map those means onto both train and test sets.
        4. Any brand in test that was never seen in train gets the global
           training mean as a safe fallback.

Run from the project root:
    python src/train_model.py
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DATA_PATH         = "data/processed_car_data.csv"
MODEL_PATH        = "models/car_price_model.pkl"
BRAND_AVG_PATH    = "data/brand_avg_price.json"
TARGET_COLUMN     = "selling_price"
TEST_SIZE         = 0.2
RANDOM_STATE      = 42

PARAM_DISTRIBUTIONS = {
    "n_estimators":     randint(200, 600),
    "max_depth":        randint(3, 9),
    "learning_rate":    uniform(0.03, 0.17),
    "subsample":        uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "min_child_weight": randint(1, 6),
    "gamma":            uniform(0, 0.3),
}
N_ITER  = 30
CV_FOLD = 5


# --------------------------------------------------------------------------- #
# STEP 1: LOAD
# --------------------------------------------------------------------------- #
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded '{filepath}' -> shape: {df.shape}")
    return df


# --------------------------------------------------------------------------- #
# STEP 2: FEATURES / TARGET
# --------------------------------------------------------------------------- #
def split_features_target(df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    print(f"[INFO] Features: {X.shape[1]} columns | Rows: {X.shape[0]}")
    return X, y


# --------------------------------------------------------------------------- #
# STEP 3: TRAIN-TEST SPLIT
# --------------------------------------------------------------------------- #
def split_train_test(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# STEP 4: BRAND_AVG_PRICE  <-- leakage prevention happens here
# --------------------------------------------------------------------------- #
def add_brand_avg_price(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    save_path: str = BRAND_AVG_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean selling_price per brand using ONLY training rows,
    then map those means onto both train and test sets.

    WHY THIS PREVENTS LEAKAGE:
        - y_train contains selling prices for training rows only.
        - We never touch y_test here.
        - When we map onto X_test, we are applying a lookup table that was
          built from training data -- the same way a model would see a new
          car at inference time.  The test set prices play no role.

    FALLBACK:
        If a brand appears in the test set but not in training (rare with
        this dataset but possible with new scraped data), we substitute
        the global training mean.  This is the standard safe fallback for
        target encoding.

    SAVED TO JSON:
        The mapping is saved to data/brand_avg_price.json so the Streamlit
        app can look up the avg price for any brand the user selects --
        without re-reading the training set at runtime.
    """
    # Join X_train's brand column with y_train to compute per-brand means
    train_df = X_train[["brand"]].copy()
    train_df["selling_price"] = y_train.values

    # Mean price per brand -- computed from TRAINING ROWS ONLY
    brand_avg_map = train_df.groupby("brand")["selling_price"].mean().to_dict()

    # Global training mean: safe fallback for unseen brands
    global_mean = float(y_train.mean())

    # Apply to train set
    X_train = X_train.copy()
    X_train["brand_avg_price"] = (
        X_train["brand"].map(brand_avg_map).fillna(global_mean)
    )

    # Apply to test set using the SAME training-derived map (no test prices used)
    X_test = X_test.copy()
    X_test["brand_avg_price"] = (
        X_test["brand"].map(brand_avg_map).fillna(global_mean)
    )

    # Save mapping so app.py can use it at prediction time
    mapping_to_save = {**brand_avg_map, "__global_mean__": global_mean}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(mapping_to_save, f, indent=2)

    print(f"[INFO] brand_avg_price computed from {len(brand_avg_map)} training brands.")
    print(f"[INFO] Global fallback mean: ₹{global_mean:,.0f}")
    print(f"[INFO] Mapping saved to '{save_path}'")

    return X_train, X_test


# --------------------------------------------------------------------------- #
# STEP 5: ONE-HOT ENCODE BRAND
# --------------------------------------------------------------------------- #
def encode_brand(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode the raw 'brand' column and align train/test columns.

    This is done AFTER brand_avg_price is computed (which needs the raw
    brand string).  We use pd.get_dummies on X_train to define the column
    set, then reindex X_test to match -- ensuring test never gets extra
    columns that didn't exist in training.
    """
    X_train = pd.get_dummies(X_train, columns=["brand"], drop_first=True)

    X_test  = pd.get_dummies(X_test,  columns=["brand"], drop_first=True)
    # Reindex test to exactly match training columns (fill missing with 0)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"[INFO] Brand one-hot encoded. Final feature count: {X_train.shape[1]}")
    return X_train, X_test


# --------------------------------------------------------------------------- #
# STEP 6: TUNE + TRAIN
# --------------------------------------------------------------------------- #
def tune_and_train(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Hyperparameter search then fit best XGBoost model on full training set."""
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="r2",
        cv=CV_FOLD,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    print(f"\n[INFO] Hyperparameter search "
          f"({N_ITER} iterations x {CV_FOLD}-fold = {N_ITER * CV_FOLD} fits)...")
    search.fit(X_train, y_train)

    print(f"\n[INFO] Best CV R²: {search.best_score_:.4f}")
    print("[INFO] Best params:")
    for k, v in search.best_params_.items():
        print(f"         {k:<22}: {v}")

    return search.best_estimator_


# --------------------------------------------------------------------------- #
# STEP 7: EVALUATE
# --------------------------------------------------------------------------- #
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "R2 Score": r2_score(y_test, y_pred),
        "MAE":      mean_absolute_error(y_test, y_pred),
        "RMSE":     np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def print_metrics(metrics: dict, previous_r2: float = 0.6937) -> None:
    delta = metrics["R2 Score"] - previous_r2
    arrow = "▲" if delta >= 0 else "▼"
    print("\n" + "=" * 52)
    print("MODEL EVALUATION RESULTS")
    print("=" * 52)
    print(f"R2 Score : {metrics['R2 Score']:.4f}   "
          f"{arrow} {abs(delta):.4f} vs Phase 2 baseline (0.6937)")
    print(f"MAE      : ₹{metrics['MAE']:>12,.0f}")
    print(f"RMSE     : ₹{metrics['RMSE']:>12,.0f}")
    print("=" * 52)


def print_feature_importance(model, feature_names: list, top_n: int = 15) -> None:
    importance_df = (
        pd.DataFrame({"feature": feature_names,
                      "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n{'=' * 52}")
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print(f"{'=' * 52}")
    for _, row in importance_df.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 300)
        print(f"{row['feature']:<35} {row['importance']:.4f}  {bar}")
    print("=" * 52)


# --------------------------------------------------------------------------- #
# STEP 8: SAVE
# --------------------------------------------------------------------------- #
def save_model(model, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\n[INFO] Model saved -> '{filepath}'")


# --------------------------------------------------------------------------- #
# MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    print(f"\n[START] Phase 3 training — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load preprocessed data (brand is still a raw string here)
    df = load_data(DATA_PATH)

    # 2. Separate features and target
    X, y = split_features_target(df, TARGET_COLUMN)

    # 3. Train/test split -- must happen BEFORE any target-based feature computation
    X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE, RANDOM_STATE)

    # ------------------------------------------------------------------ #
    # 4. LEAKAGE-SAFE brand_avg_price
    #    Computed from y_train only. y_test is never touched here.
    # ------------------------------------------------------------------ #
    X_train, X_test = add_brand_avg_price(X_train, X_test, y_train)

    # 5. One-hot encode brand (AFTER avg price is computed, so raw string
    #    is still available in step 4)
    X_train, X_test = encode_brand(X_train, X_test)

    # 6. Tune + train XGBoost
    model = tune_and_train(X_train, y_train)

    # 7. Evaluate on held-out test set
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)
    print_feature_importance(model, list(X_train.columns), top_n=15)

    # 8. Save model
    save_model(model, MODEL_PATH)

    print(f"\n[DONE] R² = {metrics['R2 Score']:.4f}")


if __name__ == "__main__":
    main()