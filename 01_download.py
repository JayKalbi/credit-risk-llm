"""
HybridCreditLLM — Dataset Download Script
===========================================
Downloads all three datasets required for the project.
Datasets are already present, so this script also verifies them.

Datasets:
    1. LendingClub (2007-2018) — Primary training set
    2. Home Credit Default Risk — Out-of-sample test
    3. German Credit (UCI) — Fairness audit

Usage:
    python 01_download.py
    python 01_download.py --verify-only
"""

import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


def download_german_credit():
    """Download German Credit dataset from UCI if not present."""
    filepath = DATA_DIR / "german_credit.data"
    if filepath.exists():
        print(f"  [OK] German Credit already exists ({filepath})")
        return

    print("  Downloading German Credit from UCI...")
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "statlog/german/german.data")
    os.makedirs(DATA_DIR, exist_ok=True)
    urllib.request.urlretrieve(url, filepath)
    print(f"  [OK] Downloaded: {filepath}")


def verify_datasets():
    """Verify all datasets are present and loadable."""
    print("\n" + "=" * 60)
    print("  DATASET VERIFICATION")
    print("=" * 60)

    # LendingClub
    lc_path = DATA_DIR / "accepted_2007_to_2018Q4.csv.gz"
    if lc_path.exists():
        size_mb = lc_path.stat().st_size / (1024 * 1024)
        lc = pd.read_csv(lc_path, nrows=5, low_memory=False)
        print(f"\n  [OK] LendingClub: {lc_path}")
        print(f"    Size: {size_mb:.0f} MB | Columns: {lc.shape[1]}")
    else:
        print(f"\n  ✗ LendingClub NOT FOUND at {lc_path}")
        print("    Run: kaggle datasets download wordsforthewise/lending-club "
              f"-p {DATA_DIR}/")

    # Home Credit
    hc_path = DATA_DIR / "HC_application_train.csv"
    if hc_path.exists():
        size_mb = hc_path.stat().st_size / (1024 * 1024)
        hc = pd.read_csv(hc_path, nrows=5)
        print(f"\n  [OK] Home Credit: {hc_path}")
        print(f"    Size: {size_mb:.0f} MB | Columns: {hc.shape[1]}")
    else:
        print(f"\n  ✗ Home Credit NOT FOUND at {hc_path}")
        print("    Run: kaggle competitions download -c home-credit-default-risk "
              f"-p {DATA_DIR}/")

    # German Credit
    gc_path = DATA_DIR / "german_credit.data"
    if gc_path.exists():
        size_kb = gc_path.stat().st_size / 1024
        gc = pd.read_csv(gc_path, sep=r"\s+", header=None, nrows=5)
        print(f"\n  [OK] German Credit: {gc_path}")
        print(f"    Size: {size_kb:.0f} KB | Columns: {gc.shape[1]}")
    else:
        print(f"\n  ✗ German Credit NOT FOUND at {gc_path}")

    # Rejected loans (bonus — may use later)
    rej_path = DATA_DIR / "rejected_2007_to_2018Q4.csv.gz"
    if rej_path.exists():
        size_mb = rej_path.stat().st_size / (1024 * 1024)
        print(f"\n  [OK] Rejected Loans: {rej_path}")
        print(f"    Size: {size_mb:.0f} MB (available for future use)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("HybridCreditLLM — Dataset Setup")
    print("=" * 60)

    if "--verify-only" not in sys.argv:
        download_german_credit()

    verify_datasets()
    print("\n[OK] Dataset verification complete")