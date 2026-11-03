import json

import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.joblib"
COLUMNS_PATH = "models/feature_columns.json"


def load_artifacts():
    """Load the saved model and its feature column names."""
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH) as f:
        feature_columns = json.load(f)
    return model, feature_columns


def prepare_input(customer, feature_columns):
    """Turn one raw customer (a dict) into a row with exactly the model's columns."""
    row = pd.DataFrame([customer])
    row = pd.get_dummies(row, dtype=int)  # text -> 0/1 columns
    return row.reindex(columns=feature_columns, fill_value=0)


def predict_churn(customer, model=None, feature_columns=None, threshold=0.5):
    """Return the churn probability and a yes/no prediction for one customer."""
    if model is None or feature_columns is None:
        model, feature_columns = load_artifacts()

    X_new = prepare_input(customer, feature_columns)
    probability = float(model.predict_proba(X_new)[0, 1])
    return {
        "churn_probability": round(probability, 3),
        "will_churn": probability >= threshold,
    }


HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0,
}

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 3900.0,
}


if __name__ == "__main__":
    model, feature_columns = load_artifacts()

    print("High-risk customer:", predict_churn(HIGH_RISK, model, feature_columns))
    print("Low-risk customer: ", predict_churn(LOW_RISK, model, feature_columns))
