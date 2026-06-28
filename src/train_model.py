"""
train_model.py
---------------
Trains a RandomForestRegressor to predict used-car selling prices.
 
Run this script from the project root, e.g.:
    python src/train_model.py
 
Expected project structure:
    data/car_data.csv          -> cleaned, fully numeric dataset (input)
    src/train_model.py         -> this file
    models/car_price_model.pkl -> trained model (output)
"""
 
import os
 
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
 
 
# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
# Keeping paths/settings as constants makes them easy to find and change
# later (e.g. if you rename the dataset or move it to a different folder).
DATA_PATH = "data/car_data.csv"
MODEL_PATH = "models/car_price_model.pkl"
TARGET_COLUMN = "selling_price"
TEST_SIZE = 0.2          # 20% of data held out for testing
RANDOM_STATE = 42        # fixed seed -> same split / same results every run
 
 
# --------------------------------------------------------------------------- #
# STEP 1: LOAD DATA
# --------------------------------------------------------------------------- #
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the preprocessed dataset from a CSV file.
 
    Since data_preprocessing.py has already cleaned and encoded everything,
    this step is intentionally simple -- it just reads the CSV.
    """
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded data from '{filepath}' -> shape: {df.shape}")
    return df
 
 
# --------------------------------------------------------------------------- #
# STEP 2: SPLIT INTO FEATURES (X) AND TARGET (y)
# --------------------------------------------------------------------------- #
def split_features_target(df: pd.DataFrame, target_column: str):
    """
    Separate the dataframe into:
        X -> all the input features the model learns from
        y -> the value the model is trying to predict (selling_price)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y
 
 
# --------------------------------------------------------------------------- #
# STEP 3: TRAIN-TEST SPLIT
# --------------------------------------------------------------------------- #
def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    """
    Split data into training and testing sets.
 
    - The model learns patterns ONLY from the training set.
    - The test set is held back and used purely to check how well the
      model performs on data it has never seen -- this is what tells us
      whether the model has actually learned something useful, rather
      than just memorized the training data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train size: {X_train.shape[0]} rows | Test size: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test
 
 
# --------------------------------------------------------------------------- #
# STEP 4: TRAIN THE MODEL
# --------------------------------------------------------------------------- #
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on the training data.
 
    RandomForest works by building many decision trees on random subsets
    of the data/features, then averaging their predictions. This usually
    gives a strong, stable baseline without needing much tuning -- a good
    fit for a first version of a price-prediction model.
    """
    model = RandomForestRegressor(
        n_estimators=100,    # number of trees in the "forest"
        max_depth=None,      # let trees grow until leaves are pure (default)
        random_state=RANDOM_STATE,
        n_jobs=-1,            # use all available CPU cores -> faster training
    )
 
    print("[INFO] Training RandomForestRegressor...")
    model.fit(X_train, y_train)
    print("[INFO] Training complete.")
    return model
 
 
# --------------------------------------------------------------------------- #
# STEP 5: EVALUATE THE MODEL
# --------------------------------------------------------------------------- #
def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model on the test set using three standard
    regression metrics:
 
        R2 Score -> how much of the variance in price the model explains.
                    1.0 = perfect, 0.0 = no better than guessing the mean.
        MAE      -> Mean Absolute Error. Average ₹ amount the prediction
                    is off by, in the same units as selling_price.
        RMSE     -> Root Mean Squared Error. Similar to MAE, but penalizes
                    large errors more heavily (useful for spotting cars
                    where the model is way off).
    """
    y_pred = model.predict(X_test)
 
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # NOTE: newer scikit-learn versions removed the `squared` argument from
    # mean_squared_error, so we take the square root manually for RMSE --
    # this keeps the code working across scikit-learn versions.
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
 
    metrics = {"R2 Score": r2, "MAE": mae, "RMSE": rmse}
    return metrics
 
 
def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics in a clear, readable format."""
    print("\n" + "=" * 40)
    print("MODEL EVALUATION RESULTS")
    print("=" * 40)
    print(f"R2 Score : {metrics['R2 Score']:.4f}   (closer to 1.0 is better)")
    print(f"MAE      : {metrics['MAE']:,.2f}   (average error, in price units)")
    print(f"RMSE     : {metrics['RMSE']:,.2f}   (penalizes big errors more)")
    print("=" * 40)
 
 
# --------------------------------------------------------------------------- #
# STEP 6: SAVE THE TRAINED MODEL
# --------------------------------------------------------------------------- #
def save_model(model: RandomForestRegressor, filepath: str) -> None:
    """
    Save the trained model to disk using joblib.
 
    joblib is preferred over plain `pickle` for scikit-learn models because
    it's more efficient at storing the large numpy arrays inside trained
    models -- but the saved file still uses the standard .pkl extension.
    """
    # Make sure the target folder (e.g. "models/") exists before saving
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Model saved to '{filepath}'")
 
 
# --------------------------------------------------------------------------- #
# MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    # 1. Load data
    df = load_data(DATA_PATH)
 
    # 2. Split into features (X) and target (y)
    X, y = split_features_target(df, TARGET_COLUMN)
 
    # 3. Train-test split (80-20)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
 
    # 4. Train the RandomForestRegressor
    model = train_model(X_train, y_train)
 
    # 5. Evaluate on unseen test data
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)
 
    # 6. Save the trained model for later use (e.g. in a prediction API/app)
    save_model(model, MODEL_PATH)
 
 
if __name__ == "__main__":
    main()
 