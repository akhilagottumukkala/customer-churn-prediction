import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from clean_data import clean_data
from load_data import load_data
from preprocess import encode_features, split_data

FIG_DIR = "reports/figures"


def get_importances(model, feature_names):
    """Return feature importances as a sorted table."""
    table = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_importances(table, top_n=10):
    """Save a horizontal bar chart of the top features."""
    top = table.head(top_n).iloc[::-1]  # reversed so the biggest bar is on top

    plt.figure(figsize=(8, 5))
    sns.barplot(data=top, x="importance", y="feature")
    plt.title(f"Top {top_n} features by random forest importance")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_feature_importance.png", dpi=150)
    plt.close()
    print("Saved 05_feature_importance.png")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    df = clean_data(load_data())
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    forest.fit(X_train, y_train)

    table = get_importances(forest, X_train.columns)
    print("Top 10 features:")
    print(table.head(10).round(3).to_string(index=False))

    plot_importances(table)
