from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import load_artifacts, predict_churn

YesNo = Literal["Yes", "No"]
YesNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """The details of one customer. Invalid values are rejected automatically."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
                "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0, "TotalCharges": 170.0,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool


app = FastAPI(title="Customer Churn API", version="0.1.0")

# Load the model once when the server starts, not on every request
model, feature_columns = load_artifacts()


@app.get("/health")
def health():
    """Tell the caller the service is running."""
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    """Return the churn probability for one customer."""
    return predict_churn(customer.model_dump(), model, feature_columns)
