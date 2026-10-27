import pandas as pd

DATA_PATH = "data/Telco-Customer-Churn.csv"


def load_data(path=DATA_PATH):
    """Read the churn CSV into a pandas DataFrame."""
    return pd.read_csv(path)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    df = load_data()

    print("Shape (rows, columns):", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nChurn counts:")
    print(df["Churn"].value_counts())
    print("\nChurn percentage:")
    print(df["Churn"].value_counts(normalize=True) * 100)