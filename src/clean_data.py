import pandas as pd
from load_data import load_data


def clean_data(df):
    """Return a cleaned copy of the churn DataFrame."""
    df = df.copy()  # don't change the original

    # 1. TotalCharges: text -> number. The 11 blank values become NaN...
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # ...and since those customers have tenure 0 (never billed), their total is 0.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 2. Target: Yes/No -> 1/0, because models need numbers.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 3. customerID is just a label. It has no predictive value.
    df = df.drop(columns=["customerID"])

    return df


if __name__ == "__main__":
    raw = load_data()
    clean = clean_data(raw)

    print("Before:", raw.shape, "| After:", clean.shape)
    print("\nTotalCharges type now:", clean["TotalCharges"].dtype)
    print("Missing values left:", clean.isnull().sum().sum())
    print("\nChurn values:")
    print(clean["Churn"].value_counts())
