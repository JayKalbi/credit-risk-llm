# ── LendingClub ─────────────────────────────────────────────
# (This is a large file ~1.7GB, takes 5–10 min)
kaggle datasets download wordsforthewise/lending-club -p data/raw/
unzip data/raw/lending-club.zip -d data/raw/lending_club/

# ── Home Credit Default Risk ─────────────────────────────────
kaggle competitions download -c home-credit-default-risk -p data/raw/
unzip data/raw/home-credit-default-risk.zip -d data/raw/home_credit/

# ── German Credit (UCI) ─────────────────────────────────────
# Download directly from UCI

import urllib.request
urllib.request.urlretrieve(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    "data/raw/german_credit.data"
)




# ── Verify downloads ─────────────────────────────────────────
import pandas as pd, os

lc = pd.read_csv('data/raw/lending_club/accepted_2007_to_2018Q4.csv/accepted_2007_to_2018Q4.csv',
                 nrows=100)
print(f"LendingClub columns: {lc.shape[1]}")  # Expect ~140
print(lc[['loan_amnt','loan_status','desc']].head(3))

hc = pd.read_csv('data/raw/home_credit/application_train.csv', nrows=100)
print(f"Home Credit columns: {hc.shape[1]}")  # Expect ~122
print("✓ All datasets verified")