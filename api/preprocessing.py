import pandas as pd


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw customer data before prediction.
    """
    df = df.copy()

    # Remove customer ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges into numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # Remove target column
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    return df