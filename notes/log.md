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

  ## Day 3: Oct 28, 2026

### What I did
- Wrote src/inspect_totalcharges.py to find out why TotalCharges was stored as text.
- Used pd.to_numeric(errors="coerce") to find the values that could not be
  converted to numbers, then looked at those rows.
- Found 11 problem rows. All of them have tenure = 0 and a TotalCharges value
  of a single blank space (' ').
- Fixed a truncated dataset download (see mistakes below) and re-ran the checks.
- (Only keep this line if you created and ran it) Wrote src/clean_data.py to
  convert TotalCharges to numbers, set the 11 blanks to 0, convert Churn from
  Yes/No to 1/0, and drop customerID.

### What I found
- TotalCharges showed as text (str) because 11 rows contain a blank space
  instead of a number, so pandas could not read the column as numbers.
- These 11 customers have tenure = 0, so they are brand new and have not been
  billed yet. Their total charges should be 0, which is why filling the blanks
  with 0 is a reasoned choice and not just a guess.
- A blank space is not counted as missing by isnull(), which is why Day 2
  showed no missing values even though the column had a problem.

### Mistakes I made and how I fixed them
1. Truncated download: my first dataset download was interrupted by a slow
   connection, so the CSV had only 4,340 rows instead of 7,043. The sign was
   a row with tenure 38 but NaN in MonthlyCharges, TotalCharges and Churn.
   Fix: deleted the file, downloaded it again, and checked that the shape was
   (7043, 21) and the file size was about 970 KB.
2. Typo in a filename: I created insoect_totalcharges.py instead of
   inspect_totalcharges.py. Fix: renamed it with Move-Item.
3. Nested folder: the file ended up in src/src/ because I typed src/ in the
   name while right-clicking the src folder. Fix: moved it up one level and
   removed the empty extra folder.
4. Running python src/... only works from the project root folder, and the
   virtual environment has to be activated in every new terminal.

### What I learned
- errors="coerce" in pd.to_numeric turns values that cannot be converted into
  NaN instead of crashing, which makes bad values easy to find.
- Always check the row count and file size after downloading data. A cut-off
  file can look fine until you notice odd rows.
- Filling missing values should be based on evidence from the data (here,
  tenure = 0), not just on a default like the mean.
- A green file name in VS Code means git sees a new, uncommitted file.

### What confused me
- Why isnull() did not catch the blank spaces, since a blank string is a real
  value and not a missing one.
- The difference between where a file is created in the Explorer and what
  path I type when I name it.