import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from clean_data import clean_data
from cross_validate import build_logreg
from load_data import load_data
from preprocess import encode_features, split_data


def tune(name, model, grid, X_train, y_train):
    """Grid search with 5-fold CV on the training set, scored by ROC-AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"\n{name}")
    print("  Best settings:", search.best_params_)
    print(f"  Best cross-validated ROC-AUC: {search.best_score_:.3f}")
    return search.best_estimator_


def test_scores(name, model, X_test, y_test):
    """One row of final test-set metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }


if __name__ == "__main__":
    df = clean_data(load_data())
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_logreg = tune(
        "Logistic regression",
        build_logreg(),
        {"model__C": [0.01, 0.1, 1, 10]},
        X_train, y_train,
    )

    best_forest = tune(
        "Random forest (balanced)",
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        {"max_depth": [5, 8, 12, None], "min_samples_leaf": [1, 5, 10, 20]},
        X_train, y_train,
    )

    results = pd.DataFrame(
        [
            test_scores("Logistic regression (tuned)", best_logreg, X_test, y_test),
            test_scores("Random forest balanced (tuned)", best_forest, X_test, y_test),
        ]
    ).round(3)

    print("\nFinal check on the untouched test set:")
    print(results.to_string(index=False))
