import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from clean_data import clean_data
from load_data import load_data
from preprocess import encode_features, split_data
from train_baseline import scale_numeric


def score(name, model, X_test, y_test):
    """Return one row of metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


if __name__ == "__main__":
    df = clean_data(load_data())
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Logistic regression needs scaled numbers
    X_train_s, X_test_s = scale_numeric(X_train, X_test)
    logreg = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)

    # Random forests work on the unscaled data
    forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    forest.fit(X_train, y_train)

    forest_bal = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    forest_bal.fit(X_train, y_train)

    results = pd.DataFrame(
        [
            score("Logistic regression", logreg, X_test_s, y_test),
            score("Random forest", forest, X_test, y_test),
            score("Random forest (balanced)", forest_bal, X_test, y_test),
        ]
    ).round(3)

    print(results.to_string(index=False))
