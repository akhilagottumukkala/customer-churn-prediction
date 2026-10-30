import pandas as pd
from sklearn.model_selection import train_test_split

from clean_data import clean_data
from load_data import load_data

RANDOM_STATE = 42


def encode_features(df):
    """Split the cleaned data into features X and target y, with text columns one-hot encoded."""
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    text_cols = X.select_dtypes(exclude="number").columns.tolist()
    X = pd.get_dummies(X, columns=text_cols, drop_first=True, dtype=int)
    return X, y


def split_data(X, y, test_size=0.2):
    """Hold out part of the data for testing, keeping the churn rate the same in both parts."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


if __name__ == "__main__":
    df = clean_data(load_data())
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("X shape:", X.shape)
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print(f"\nChurn rate overall: {y.mean() * 100:.1f}%")
    print(f"Churn rate in train: {y_train.mean() * 100:.1f}%")
    print(f"Churn rate in test:  {y_test.mean() * 100:.1f}%")
