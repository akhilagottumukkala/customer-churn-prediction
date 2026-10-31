import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from clean_data import clean_data
from load_data import load_data
from preprocess import encode_features, split_data
from train_baseline import scale_numeric

FIG_DIR = "reports/figures"


def plot_confusion_matrix(y_test, y_pred):
    """Save a labelled heatmap of the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted stayed", "Predicted churn"],
        yticklabels=["Actually stayed", "Actually churned"],
    )
    plt.title("Baseline logistic regression")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_baseline_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved 04_baseline_confusion_matrix.png")


def threshold_table(y_test, y_prob, thresholds=(0.5, 0.4, 0.3, 0.2)):
    """Show how precision and recall change as we flag more customers."""
    print("\nThreshold | Precision | Recall |   F1  | Customers flagged")
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        print(
            f"   {t:.1f}    |   {precision_score(y_test, y_pred):.3f}   | "
            f"{recall_score(y_test, y_pred):.3f}  | {f1_score(y_test, y_pred):.3f} | {y_pred.sum()}"
        )


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    df = clean_data(load_data())
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = scale_numeric(X_train, X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, y_pred)
    threshold_table(y_test, y_prob)
