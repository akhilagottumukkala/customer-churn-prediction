import os

import matplotlib.pyplot as plt
import seaborn as sns

from clean_data import clean_data
from load_data import load_data

FIG_DIR = "reports/figures"


def plot_churn_distribution(df):
    """Bar chart of how many customers stayed vs churned."""
    counts = df["Churn"].value_counts().sort_index()
    labels = ["Stayed (0)", "Churned (1)"]

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=labels, y=counts.values)

    # Write the count and percentage above each bar
    for i, value in enumerate(counts.values):
        pct = value / len(df) * 100
        ax.text(i, value + 50, f"{value} ({pct:.1f}%)", ha="center")

    plt.title("Customer churn distribution")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_churn_distribution.png", dpi=150)
    plt.close()
    print("Saved 01_churn_distribution.png")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)  # create the folder if it is missing
    df = clean_data(load_data())
    plot_churn_distribution(df)
