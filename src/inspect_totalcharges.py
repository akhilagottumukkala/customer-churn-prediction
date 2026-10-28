import pandas as pd
from load_data import load_data

df = load_data()

print("TotalCharges type:", df["TotalCharges"].dtype)

# Try to turn the column into numbers. Anything that can't be
# converted becomes NaN (missing) instead of crashing.
converted = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Rows where the conversion failed
bad_rows = df[converted.isnull()]

print("\nNumber of problem rows:", len(bad_rows))
print("\nThe problem rows:")
print(bad_rows[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

print("\nWhat the bad values look like:")
print(repr(bad_rows["TotalCharges"].unique()))