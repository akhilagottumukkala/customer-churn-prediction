import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clean_data import clean_data
from load_data import load_data
from preprocess import encode_features, split_data
from train_baseline import NUMERIC_COLS

SCORING = ["roc_auc", "recall", "precision", "f1"]


def build_logreg():
    """Logistic regression with scaling of the numeric columns inside the pipeline."""
    scale = ColumnTransformer(
        [("scale", StandardScaler(), NUMERIC_COLS)], remainder="passthrough"
    )
    return Pipeline([("scale", scale), ("model", LogisticRegression(max_iter=1000))])


def cross_validate_model(name, model, X_train, y_train):
    """Run 5-fold stratified cross-validation and return mean and spread of each metric."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)

    row = {"Model": name}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        row[metric] = f"{values.mean():.3f} +/- {values.std():.3f}"
    return row


if __name__ == "__main__":
    df = clean_data(load_data())
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "Logistic regression": build_logreg(),
        "Random forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Random forest (balanced)": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    rows = [cross_validate_model(name, m, X_train, y_train) for name, m in models.items()]
    print(pd.DataFrame(rows).to_string(index=False))
