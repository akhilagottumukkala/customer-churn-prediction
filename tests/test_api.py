from fastapi.testclient import TestClient

from api import app
from predict import HIGH_RISK, LOW_RISK

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_probability():
    response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    # the yes/no answer must agree with the probability
    assert body["will_churn"] == (body["churn_probability"] >= 0.5)


def test_high_risk_scores_higher_than_low_risk():
    high = client.post("/predict", json=HIGH_RISK).json()
    low = client.post("/predict", json=LOW_RISK).json()
    assert high["churn_probability"] > low["churn_probability"]
    assert high["will_churn"] is True
    assert low["will_churn"] is False


def test_invalid_contract_is_rejected():
    bad = {**HIGH_RISK, "Contract": "Monthly"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_negative_tenure_is_rejected():
    bad = {**HIGH_RISK, "tenure": -5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_missing_field_is_rejected():
    incomplete = {k: v for k, v in HIGH_RISK.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
