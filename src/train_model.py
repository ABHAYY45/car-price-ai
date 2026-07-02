"""
train_model.py
--------------
Trains an XGBoostRegressor on the processed CarDekho dataset.

Phase 2 changes vs Phase 1:
    - Switched from RandomForestRegressor to XGBRegressor
    - Added RandomizedSearchCV for hyperparameter tuning
    - Prints best params found so you can learn what worked
    - Feature importance chart still included

Run from the project root:
    python src/train_model.py
"""

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
DATA_PATH     = "data/processed_car_data.csv"
MODEL_PATH    = "models/car_price_model.pkl"
TARGET_COLUMN = "selling_price"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# Hyperparameter search space for RandomizedSearchCV.
# Each key is an XGBRegressor parameter; the value is the distribution
# to sample from. RandomizedSearchCV will try N_ITER random combinations
# and return the best one (measured by cross-validated R²).
PARAM_DISTRIBUTIONS = {
    "n_estimators":      randint(200, 600),      # number of boosting rounds
    "max_depth":         randint(3, 9),          # tree depth (deeper = more complex)
    "learning_rate":     uniform(0.03, 0.17),    # step size (lower = more robust)
    "subsample":         uniform(0.7, 0.3),      # fraction of rows per tree
    "colsample_bytree":  uniform(0.7, 0.3),      # fraction of columns per tree
    "min_child_weight":  randint(1, 6),          # minimum samples per leaf
    "gamma":             uniform(0, 0.3),        # minimum loss reduction to split
}
N_ITER  = 30    # number of random combinations to try (more = better but slower)
CV_FOLD = 5     # k in k-fold cross validation


# --------------------------------------------------------------------------- #
# STEP 1: LOAD DATA
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
# STEP 4: TUNE + TRAIN XGBoost
# --------------------------------------------------------------------------- #
def tune_and_train(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Find the best XGBoost hyperparameters via RandomizedSearchCV, then
    return the best estimator already fitted on the full training set.

    WHY RandomizedSearchCV over GridSearchCV?
        GridSearch tries every combination -> too slow for 7 parameters.
        RandomizedSearch samples N_ITER random combinations -> much faster
        with only slightly worse coverage. For 30 iterations and 5-fold CV
        this runs ~150 fits, which takes 1-3 minutes on a laptop.

    WHY XGBoost over RandomForest?
        XGBoost builds trees sequentially -- each new tree corrects the
        errors of the previous ones (gradient boosting). This means it
        squeezes more signal from the same features, especially interaction
        terms like km_per_year and age_km_interaction where the relationship
        with price is non-linear and complex.
    """
    base_model = XGBRegressor(
        objective="reg:squarederror",  # standard regression loss
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,                   # suppress XGBoost's own output
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="r2",
        cv=CV_FOLD,
        verbose=1,          # prints one line per fold set so you see progress
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,         # refit best params on full X_train after search
    )

    print(f"\n[INFO] Starting hyperparameter search "
          f"({N_ITER} iterations × {CV_FOLD}-fold CV = {N_ITER * CV_FOLD} fits)...")
    search.fit(X_train, y_train)

    print(f"\n[INFO] Best CV R²  : {search.best_score_:.4f}")
    print(f"[INFO] Best params  :")
    for param, value in search.best_params_.items():
        print(f"         {param:<22}: {value}")

    # search.best_estimator_ is already refitted on the full X_train
    return search.best_estimator_


# --------------------------------------------------------------------------- #
# STEP 5: EVALUATE
# --------------------------------------------------------------------------- #
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "R2 Score": r2_score(y_test, y_pred),
        "MAE":      mean_absolute_error(y_test, y_pred),
        "RMSE":     np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def print_metrics(metrics: dict, previous_r2: float = 0.6486) -> None:
    delta = metrics["R2 Score"] - previous_r2
    arrow = "▲" if delta >= 0 else "▼"

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"R2 Score : {metrics['R2 Score']:.4f}   "
          f"{arrow} {abs(delta):.4f} vs Phase 1 baseline")
    print(f"MAE      : ₹{metrics['MAE']:>12,.2f}")
    print(f"RMSE     : ₹{metrics['RMSE']:>12,.2f}")
    print("=" * 50)


# --------------------------------------------------------------------------- #
# STEP 5b: FEATURE IMPORTANCE
# --------------------------------------------------------------------------- #
def print_feature_importance(model: XGBRegressor, feature_names: list, top_n: int = 15) -> None:
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n{'=' * 50}")
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print(f"{'=' * 50}")
    for _, row in importance_df.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 300)
        print(f"{row['feature']:<35} {row['importance']:.4f}  {bar}")
    print("=" * 50)


# --------------------------------------------------------------------------- #
# STEP 6: SAVE
# --------------------------------------------------------------------------- #
def save_model(model, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\n[INFO] Model saved to '{filepath}'")


# --------------------------------------------------------------------------- #
# MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    print(f"\n[START] Phase 2 training pipeline — "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df                             = load_data(DATA_PATH)
    X, y                           = split_features_target(df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE, RANDOM_STATE)

    model   = tune_and_train(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)
    print_feature_importance(model, list(X.columns), top_n=15)
    save_model(model, MODEL_PATH)

    print(f"\n[DONE] New XGBoost model ready. R² = {metrics['R2 Score']:.4f}")


if __name__ == "__main__":
    main()