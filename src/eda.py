import os

import matplotlib.pyplot as plt
import seaborn as sns

from clean_data import clean_data
from load_data import load_data

FIG_DIR = "reports/figures"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def plot_churn_distribution(df):
    """Bar chart of how many customers stayed vs churned."""
    counts = df["Churn"].value_counts().sort_index()
    labels = ["Stayed (0)", "Churned (1)"]

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=labels, y=counts.values)

    for i, value in enumerate(counts.values):
        pct = value / len(df) * 100
        ax.text(i, value + 50, f"{value} ({pct:.1f}%)", ha="center")

    plt.title("Customer churn distribution")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_churn_distribution.png", dpi=150)
    plt.close()
    print("Saved 01_churn_distribution.png")


def plot_numeric_vs_churn(df):
    """Box plots of each numeric feature, split by churn."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col in zip(axes, NUMERIC_COLS):
        sns.boxplot(data=df, x="Churn", y=col, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_title(f"{col} by churn")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_numeric_vs_churn.png", dpi=150)
    plt.close()
    print("Saved 02_numeric_vs_churn.png")

    print("\nAverage of each feature by churn group:")
    print(df.groupby("Churn")[NUMERIC_COLS].mean().round(1))


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    df = clean_data(load_data())
    plot_churn_distribution(df)
    plot_numeric_vs_churn(df)
