import pandas as pd

from clean_data import clean_data
from load_data import load_data


def encode_features(df):
    """Split the cleaned data into features X and target y, with text columns one-hot encoded."""
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Every column that is still text (object/str) becomes 0/1 columns.
    text_cols = X.select_dtypes(exclude="number").columns.tolist()
    print("Text columns to encode:", len(text_cols))

    X = pd.get_dummies(X, columns=text_cols, drop_first=True, dtype=int)
    return X, y


if __name__ == "__main__":
    df = clean_data(load_data())
    X, y = encode_features(df)

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)
    print("\nAny text columns left?", len(X.select_dtypes(exclude="number").columns) > 0)
    print("Missing values:", X.isnull().sum().sum())
    print("\nFirst 10 column names:")
    print(X.columns.tolist()[:10])
