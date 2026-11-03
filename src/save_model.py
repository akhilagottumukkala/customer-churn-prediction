import json
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score

from clean_data import clean_data
from load_data import load_data
from preprocess import encode_features, split_data

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/churn_model.joblib"
COLUMNS_PATH = f"{MODEL_DIR}/feature_columns.json"


def train_final_model(X_train, y_train):
    """Train the tuned balanced random forest from Day 8."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return model.fit(X_train, y_train)


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = clean_data(load_data())
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_final_model(X_train, y_train)

    # Sanity check: the numbers should match Day 8
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(f"Test recall:  {recall_score(y_test, y_pred):.3f}")

    joblib.dump(model, MODEL_PATH)
    with open(COLUMNS_PATH, "w") as f:
        json.dump(list(X.columns), f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved {len(X.columns)} feature names to {COLUMNS_PATH}")
