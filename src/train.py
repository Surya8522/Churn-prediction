import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier


def run_training():

    print("=" * 60)
    print("LAB 2: MODEL TRAINING")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD PROCESSED DATA
    # ---------------------------------------------------------

    X_train = np.load(
        "data/processed/X_train_final.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    print("\nTraining data shape:", X_train.shape)
    print("Target shape:", y_train.shape)

    # ---------------------------------------------------------
    # 2. CREATE RANDOM FOREST MODEL
    # ---------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    print("\nTraining Random Forest...")

    # ---------------------------------------------------------
    # 3. TRAIN
    # ---------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # ---------------------------------------------------------
    # 4. SAVE MODEL
    # ---------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    model_path = (
        "models/random_forest_baseline.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved at:")
    print(model_path)

    print("\n" + "=" * 60)
    print("LAB 2 TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_training()