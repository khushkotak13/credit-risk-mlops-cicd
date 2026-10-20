import pandas as pd
import pytest

from src.features import engineer_features


def sample_data():
    return pd.DataFrame({
        "Customer_ID": ["CC-1", "CC-2"],
        "Age": [25, 40],
        "Income_Bracket": ["Low", "High"],
        "Credit_Limit": [1000.0, 2000.0],
        "Current_Balance": [500.0, 1000.0],
        "Monthly_Spend": [200.0, 300.0],
        "Num_Transactions_Monthly": [10, 15],
        "Payment_Delay_Days": [2.0, 0.0],
        "Tenure_Months": [12, 24],
        "High_Risk_Customer": [1, 0],
    })


def test_engineered_features():
    result = engineer_features(sample_data())

    assert result.loc[0, "Credit_Utilization_Ratio"] == pytest.approx(0.5)
    assert result.loc[0, "Spend_Per_Transaction"] == pytest.approx(20.0)


def test_customer_id_removed():
    result = engineer_features(sample_data())
    assert "Customer_ID" not in result.columns


def test_target_preserved():
    result = engineer_features(sample_data())
    assert "High_Risk_Customer" in result.columns


def test_missing_required_column():
    data = sample_data().drop(columns=["Age"])

    with pytest.raises(ValueError, match="Missing required columns"):
        engineer_features(data)
