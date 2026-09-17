import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def run_preprocessing():

    print("=" * 60)
    print("LAB 1: DATA PREPROCESSING")
    print("=" * 60)

    # Create required directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD DATASET
    # ---------------------------------------------------------

    data_path = "data/raw/churn.csv"

    print("\n[1] Loading dataset...")

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    # ---------------------------------------------------------
    # 2. BASIC DATA ANALYSIS
    # ---------------------------------------------------------

    print("\n[2] Dataset Information")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # ---------------------------------------------------------
    # 3. DATA CLEANING
    # ---------------------------------------------------------

    print("\n[3] Data Cleaning")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing TotalCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Remove customer ID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert target variable
    df["Churn"] = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    # Remove rows with invalid target
    df = df.dropna(subset=["Churn"])

    df["Churn"] = df["Churn"].astype(int)

    print("Data cleaning completed.")

    # ---------------------------------------------------------
    # 4. SEPARATE FEATURES AND TARGET
    # ---------------------------------------------------------

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("\nTarget distribution:")
    print(y.value_counts())

    # ---------------------------------------------------------
    # 5. TRAIN TEST SPLIT
    # ---------------------------------------------------------

    print("\n[4] Train-Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ---------------------------------------------------------
    # 6. IDENTIFY COLUMN TYPES
    # ---------------------------------------------------------

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # ---------------------------------------------------------
    # 7. STANDARD SCALING
    # ---------------------------------------------------------

    print("\n[5] Applying StandardScaler")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train[numerical_columns]
    )

    X_test_scaled = scaler.transform(
        X_test[numerical_columns]
    )

    # ---------------------------------------------------------
    # 8. ONE HOT ENCODING
    # ---------------------------------------------------------

    print("[6] Applying OneHotEncoder")

    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    X_train_encoded = encoder.fit_transform(
        X_train[categorical_columns]
    )

    X_test_encoded = encoder.transform(
        X_test[categorical_columns]
    )

    # ---------------------------------------------------------
    # 9. COMBINE FEATURES
    # ---------------------------------------------------------

    X_train_final = np.hstack(
        (X_train_scaled, X_train_encoded)
    )

    X_test_final = np.hstack(
        (X_test_scaled, X_test_encoded)
    )

    print("\nFinal training shape:", X_train_final.shape)
    print("Final testing shape:", X_test_final.shape)

    # ---------------------------------------------------------
    # 10. SAVE PROCESSED DATA
    # ---------------------------------------------------------

    print("\n[7] Saving processed datasets")

    np.save(
        "data/processed/X_train_final.npy",
        X_train_final
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        y_train.to_numpy(dtype=np.int64)
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy(dtype=np.int64)
    )

    # ---------------------------------------------------------
    # 11. SAVE PREPROCESSING OBJECTS
    # ---------------------------------------------------------

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    joblib.dump(
        encoder,
        "models/ohe.pkl"
    )

    # ---------------------------------------------------------
    # 12. DATASET METADATA
    # ---------------------------------------------------------

    metadata = {
        "dataset_name": "Telco Customer Churn",
        "original_shape": list(pd.read_csv(data_path).shape),
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "target": "Churn",
        "test_size": 0.20,
        "random_state": 42
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("LAB 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()