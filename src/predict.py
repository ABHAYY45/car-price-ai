"""
predict.py
----------
Loads the trained RandomForestRegressor and predicts a used car's selling
price for a single, hardcoded example car.
 
Run this script from the project root, e.g.:
    python src/predict.py
 
Expected project structure:
    models/car_price_model.pkl -> the trained model (input)
    src/predict.py             -> this file
"""
 
import joblib
import pandas as pd
 
 
# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
MODEL_PATH = "models/car_price_model.pkl"
 
# Same encoding rules used in data_preprocessing.py -- they MUST match,
# otherwise the model will receive numbers that mean something different
# from what it was trained on.
OWNER_ORDER = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}
TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}
 
 
# --------------------------------------------------------------------------- #
# STEP 1: LOAD THE TRAINED MODEL
# --------------------------------------------------------------------------- #
def load_model(filepath: str):
    """Load the trained RandomForestRegressor model saved with joblib."""
    model = joblib.load(filepath)
    print(f"[INFO] Model loaded from '{filepath}'")
    return model
 
 
# --------------------------------------------------------------------------- #
# STEP 2: GET THE EXACT FEATURE COLUMNS THE MODEL WAS TRAINED ON
# --------------------------------------------------------------------------- #
def get_training_columns(model) -> list:
    """
    Get the exact feature column names -- AND their order -- that the model
    was trained on, straight from the model object itself.
 
    WHY THIS MATTERS:
    The model has no idea what column names mean -- it only sees numbers
    in a fixed order. One-hot encoding a single new car will only create
    columns for the categories THAT car has (e.g. just "fuel_Petrol"),
    not every category the model saw during training (e.g. "fuel_Diesel",
    "fuel_CNG", etc.). So we need the full original column list to safely
    line everything back up in Step 4.
 
    NOTE: We deliberately do NOT re-read data/car_data.csv for this. That
    file can change, get overwritten, or fall out of sync with what the
    model was actually trained on (this is exactly what caused the
    "feature names should match those that were passed during fit" error).
    scikit-learn automatically stores the training column names on the
    fitted model itself (`model.feature_names_in_`), so reading them
    directly from the model is the single source of truth and can never
    drift out of sync.
    """
    return list(model.feature_names_in_)
 
 
# --------------------------------------------------------------------------- #
# STEP 3: DEFINE INPUT AND CONVERT TO DATAFRAME
# --------------------------------------------------------------------------- #
def get_sample_input() -> dict:
    """
    Hardcoded example of a single car's raw, human-readable details.
    Later, this could be replaced with input() calls, a web form, or an API
    request -- the rest of the pipeline doesn't need to change.
    """
    sample_input = {
        "km_driven": 45000,
        "car_age": 5,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "brand": "Maruti",
    }
    return sample_input
 
 
def dict_to_dataframe(input_dict: dict) -> pd.DataFrame:
    """
    Convert the input dictionary into a single-row pandas DataFrame.
    A DataFrame is needed because that's the format the model expects
    (the same format used during training).
    """
    df = pd.DataFrame([input_dict])  # wrap in a list -> creates ONE row
    return df
 
 
# --------------------------------------------------------------------------- #
# STEP 4: ENCODE INPUT + MATCH TRAINING STRUCTURE
# --------------------------------------------------------------------------- #
def encode_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the exact same encoding used in data_preprocessing.py, so the raw
    input means the same thing to the model as the data it was trained on:
        - transmission -> binary label encoding
        - owner        -> ordinal label encoding
        - fuel, seller_type, brand -> one-hot encoding
    """
    df = df.copy()
 
    df["transmission"] = df["transmission"].map(TRANSMISSION_MAP)
    df["owner"] = df["owner"].map(OWNER_ORDER)
 
    nominal_cols = [c for c in ["fuel", "seller_type", "brand"] if c in df.columns]
    df = pd.get_dummies(df, columns=nominal_cols)
 
    return df
 
 
def align_columns(df: pd.DataFrame, training_columns: list) -> pd.DataFrame:
    """
    Reindex the encoded input so it has EXACTLY the same columns, in the
    EXACT same order, as the data the model was trained on:
        - Missing one-hot columns (e.g. "fuel_Diesel" when this car runs on
          Petrol) are added back in and filled with 0.
        - Any unexpected/extra columns are dropped.
    This step is what makes a single hardcoded example "safe" to feed into
    a model trained on 31 specific columns.
    """
    aligned_df = df.reindex(columns=training_columns, fill_value=0)
    # Ensure everything is numeric (reindexing can leave mixed bool/int dtypes)
    aligned_df = aligned_df.astype(float)
    return aligned_df
 
 
# --------------------------------------------------------------------------- #
# STEP 5: PREDICT
# --------------------------------------------------------------------------- #
def predict_price(model, input_df: pd.DataFrame) -> float:
    """Use the trained model to predict the selling price for the input row."""
    prediction = model.predict(input_df)
    return prediction[0]  # predict() returns an array; we only have 1 row
 
 
# --------------------------------------------------------------------------- #
# STEP 6: PRINT RESULT CLEARLY
# --------------------------------------------------------------------------- #
def print_result(raw_input: dict, predicted_price: float) -> None:
    """Print the input details and the predicted price in a readable format."""
    print("\n" + "=" * 45)
    print("CAR PRICE PREDICTION")
    print("=" * 45)
    for key, value in raw_input.items():
        print(f"{key:<15}: {value}")
    print("-" * 45)
    print(f"Predicted Selling Price: ₹{predicted_price:,.2f}")
    print("=" * 45)
 
 
# --------------------------------------------------------------------------- #
# MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    # 1. Load the trained model
    model = load_model(MODEL_PATH)
 
    # 2. Get the exact column structure the model expects (from the model itself)
    training_columns = get_training_columns(model)
 
    # 3. Define the hardcoded input and convert it to a DataFrame
    raw_input = get_sample_input()
    input_df = dict_to_dataframe(raw_input)
 
    # 4. Encode categorical fields, then align to the training structure
    encoded_df = encode_input(input_df)
    final_df = align_columns(encoded_df, training_columns)
 
    # 5. Predict the price
    predicted_price = predict_price(model, final_df)
 
    # 6. Print the result clearly
    print_result(raw_input, predicted_price)
 
 
if __name__ == "__main__":
    main()