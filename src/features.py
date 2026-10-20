import pandas as pd


TARGET = "High_Risk_Customer"

REQUIRED_COLUMNS = [
    "Customer_ID",
    "Age",
    "Income_Bracket",
    "Credit_Limit",
    "Current_Balance",
    "Monthly_Spend",
    "Num_Transactions_Monthly",
    "Payment_Delay_Days",
    "Tenure_Months",
    TARGET,
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate customer data and create credit-utilisation features."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()

    # Credit utilisation as a percentage of the credit limit.
    result["Credit_Utilization_Ratio"] = (
        result["Current_Balance"]
        / result["Credit_Limit"].replace(0, float("nan"))
    )

    # Average monthly spend per transaction.
    result["Spend_Per_Transaction"] = (
        result["Monthly_Spend"]
        / result["Num_Transactions_Monthly"].replace(
            0, float("nan")
        )
    )

    # Exclude identifiers and the target from model features.
    result = result.drop(columns=["Customer_ID"])

    return result
