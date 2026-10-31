from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from clean_data import clean_data
from load_data import load_data
from preprocess import encode_features, split_data

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def scale_numeric(X_train, X_test):
    """Scale the numeric columns. The scaler learns from the training data only."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])
    return X_train, X_test


def evaluate(model, X_test, y_test):
    """Print the main classification metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of churn

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1 score:  {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    df = clean_data(load_data())
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = scale_numeric(X_train, X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Baseline logistic regression results on the test set:\n")
    evaluate(model, X_test, y_test)
