# Daily Log

## Day 1: Oct 26, 2026
- Created the repo and folder structure.
- What I learned: git init creates a hidden .git folder that tracks changes.
  A commit is a saved snapshot of my project at one moment.
- What confused me: what to do to connect git with vs code 

## Day 2: Oct 27, 2026

### What I did
- Created a virtual environment (.venv) and installed pandas, matplotlib and seaborn.
- Saved the exact package versions in requirements.txt.
- Downloaded the Telco Customer Churn dataset from IBM's GitHub into the data/ folder.
- Wrote src/load_data.py to load the CSV and print a first summary of the data.

### What I found
- The dataset has 7,043 rows (customers) and 21 columns (features).
- About 26.5% of customers churned (left) and about 73.5% stayed.
- No column shows missing values in df.isnull().sum().
- TotalCharges is stored as text (object) instead of a number (float).
  It is probably because some rows contain a blank space instead of a number,
  so pandas read the whole column as text. I will check this on Day 3.

### What I learned
- A DataFrame is a table of rows and columns in pandas. Each row is one
  customer and each column is one property of the customer.
- A virtual environment is a separate box of Python packages for one project,
  so projects don't break each other when versions differ.
- requirements.txt lists the packages and versions so anyone can recreate
  my setup with pip install -r requirements.txt.
- The classes are imbalanced: far more customers stayed than churned. This
  means accuracy alone can be misleading, since a model that always predicts
  "No churn" would still be about 73% accurate.

### What confused me
- Why the isnull() check shows no missing values but TotalCharges is still
  stored as text. A blank space is not counted as missing by pandas.
- PowerShell commands differ from the ones in many tutorials (for example,
  mkdir with several names needs commas).