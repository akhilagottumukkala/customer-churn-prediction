import os

import matplotlib.pyplot as plt
import seaborn as sns

from clean_data import clean_data
from load_data import load_data

FIG_DIR = "reports/figures"
CATEGORICAL_COLS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]


def churn_rate_by_category(df, col):
    """Percentage of customers who churned in each category of a column."""
    return (df.groupby(col)["Churn"].mean() * 100).round(1).sort_values(ascending=False)


def plot_categorical_vs_churn(df):
    """Bar charts of the churn rate for each categorical feature."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, col in zip(axes.flatten(), CATEGORICAL_COLS):
        rates = churn_rate_by_category(df, col)
        sns.barplot(x=rates.index, y=rates.values, ax=ax)

        for i, value in enumerate(rates.values):
            ax.text(i, value + 0.5, f"{value}%", ha="center")

        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

        print(f"\nChurn rate by {col}:")
        print(rates)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_categorical_vs_churn.png", dpi=150)
    plt.close()
    print("\nSaved 03_categorical_vs_churn.png")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    df = clean_data(load_data())
    plot_categorical_vs_churn(df)
