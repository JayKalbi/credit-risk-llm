"""
HybridCreditLLM — Data Preprocessing Module
==============================================
Handles the entire data pipeline from raw CSV → model-ready splits.

Pipeline:
    1. Load raw LendingClub CSV
    2. Map loan_status → binary target (default=1, non-default=0)
    3. Filter to core features (from config.yaml)
    4. Engineer derived features
    5. Temporal split (Train ≤2015 | Val 2016 | Test 2017-2018)
    6. Impute missing values (median for numeric, 'UNKNOWN' for categorical)
    7. Encode categorical features (ordinal for tree models)
    8. Apply SMOTETomek resampling (training set only)
    9. Save processed splits to data/processed/

Usage:
    from src.preprocess import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    splits = pipeline.run()

    # Or run as script:
    python -m src.preprocess
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> dict:
    """Load project configuration."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# 1. Data Loading
# ============================================================================

def load_lending_club(
    config: dict,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the raw LendingClub CSV.

    Args:
        config: Project configuration dictionary.
        nrows: Optional row limit for development/testing.

    Returns:
        Raw DataFrame with all columns.
    """
    raw_dir = os.path.join(PROJECT_ROOT, config["paths"]["raw_data"])
    filepath = os.path.join(raw_dir, config["datasets"]["lending_club"]["file"])

    print(f"Loading LendingClub data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False, nrows=nrows)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

    return df


def load_home_credit(config: dict) -> pd.DataFrame:
    """Load the Home Credit dataset.

    Args:
        config: Project configuration dictionary.

    Returns:
        Raw DataFrame.
    """
    raw_dir = os.path.join(PROJECT_ROOT, config["paths"]["raw_data"])
    filepath = os.path.join(raw_dir, config["datasets"]["home_credit"]["file"])

    print(f"Loading Home Credit data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

    return df


def load_german_credit(config: dict) -> pd.DataFrame:
    """Load the German Credit dataset.

    Args:
        config: Project configuration dictionary.

    Returns:
        Raw DataFrame with 21 columns (20 features + 1 target).
    """
    raw_dir = os.path.join(PROJECT_ROOT, config["paths"]["raw_data"])
    filepath = os.path.join(raw_dir, config["datasets"]["german_credit"]["file"])

    print(f"Loading German Credit data from {filepath}...")
    df = pd.read_csv(filepath, sep=r"\s+", header=None)

    # Standard column names for German Credit (UCI)
    # Last column (20) is the target: 1=Good, 2=Bad
    col_names = [
        "status_checking", "duration", "credit_history", "purpose_gc",
        "credit_amount", "savings_account", "employment_since",
        "installment_rate", "personal_status", "other_debtors",
        "residence_since", "property", "age", "other_plans",
        "housing", "num_credits", "job", "num_dependents",
        "telephone", "foreign_worker", "target"
    ]
    df.columns = col_names

    # Remap target: 1=Good(0), 2=Bad(1) → standard binary
    df["target"] = (df["target"] == 2).astype(int)

    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Default rate: {df['target'].mean()*100:.1f}%")

    return df


# ============================================================================
# 2. Target Variable Mapping
# ============================================================================

def map_target_variable(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Map loan_status to binary target variable.

    Uses the mapping from config.yaml:
        - default (1): Charged Off, Default, Late (31-120 days), etc.
        - non_default (0): Fully Paid, etc.
        - drop: Current, In Grace Period, etc. (ambiguous)

    Args:
        df: DataFrame with loan_status column.
        config: Project configuration dictionary.

    Returns:
        DataFrame with new 'target' column, ambiguous rows dropped.
    """
    target_col = config["datasets"]["lending_club"]["target_col"]
    mapping = config["target_mapping"]

    print(f"\nMapping target variable from '{target_col}'...")
    print(f"  Original value counts:")

    # Show distribution before mapping
    status_counts = df[target_col].value_counts()
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        print(f"    {status}: {count:,} ({pct:.1f}%)")

    # Build mapping dict
    status_to_label = {}
    for status in mapping.get("default", []):
        status_to_label[status] = 1
    for status in mapping.get("non_default", []):
        status_to_label[status] = 0

    drop_statuses = set(mapping.get("drop", []))

    # Apply mapping
    df = df.copy()
    df = df[~df[target_col].isin(drop_statuses)]
    df["target"] = df[target_col].map(status_to_label)

    # Drop any unmapped statuses
    unmapped = df["target"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped:,} rows with unmapped loan_status — dropping")
        df = df.dropna(subset=["target"])

    df["target"] = df["target"].astype(int)

    n_default = (df["target"] == 1).sum()
    n_non_default = (df["target"] == 0).sum()
    total = len(df)

    print(f"\n  After mapping:")
    print(f"    Total: {total:,}")
    print(f"    Default (1): {n_default:,} ({n_default/total*100:.1f}%)")
    print(f"    Non-default (0): {n_non_default:,} ({n_non_default/total*100:.1f}%)")
    print(f"    Imbalance ratio: 1:{n_non_default/max(n_default,1):.1f}")

    return df


# ============================================================================
# 3. Feature Selection
# ============================================================================

def select_features(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Select core tabular features from config.

    Retains only the numeric and categorical features specified in
    config.yaml, plus the target, issue_d (for temporal split),
    and text columns.

    Args:
        df: Full DataFrame.
        config: Project configuration dictionary.

    Returns:
        DataFrame with selected columns only.
    """
    numeric_cols = config["core_features"]["numeric"]
    categorical_cols = config["core_features"]["categorical"]

    # Columns we always need
    keep_cols = ["target"]

    # Add issue_d for temporal splitting
    if "issue_d" in df.columns:
        keep_cols.append("issue_d")

    # Add text columns for later text pipeline
    text_primary = config["datasets"]["lending_club"].get("text_primary", "desc")
    text_fallback = config["datasets"]["lending_club"].get("text_fallback", [])
    for col in [text_primary] + text_fallback:
        if col in df.columns:
            keep_cols.append(col)

    # Check which configured features actually exist
    available_numeric = [c for c in numeric_cols if c in df.columns]
    missing_numeric = [c for c in numeric_cols if c not in df.columns]

    available_categorical = [c for c in categorical_cols if c in df.columns]
    missing_categorical = [c for c in categorical_cols if c not in df.columns]

    if missing_numeric:
        print(f"  WARNING: Missing numeric features: {missing_numeric}")
    if missing_categorical:
        print(f"  WARNING: Missing categorical features: {missing_categorical}")

    all_cols = keep_cols + available_numeric + available_categorical
    # Deduplicate while preserving order
    seen = set()
    unique_cols = []
    for c in all_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    df_selected = df[unique_cols].copy()

    print(f"\n  Feature selection:")
    print(f"    Numeric: {len(available_numeric)} features")
    print(f"    Categorical: {len(available_categorical)} features")
    print(f"    Total columns: {len(unique_cols)}")

    return df_selected


# ============================================================================
# 4. Feature Engineering
# ============================================================================

def engineer_features(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Create engineered features from config.

    Features:
        - loan_to_income: loan_amnt / (annual_inc + 1)
        - payment_burden: installment / (annual_inc / 12 + 1)
        - credit_util_dti_interaction: revol_util * dti
        - fico_mid: (fico_range_low + fico_range_high) / 2

    Args:
        df: DataFrame with core features.
        config: Project configuration dictionary.

    Returns:
        DataFrame with new engineered features added.
    """
    df = df.copy()
    engineered = config.get("engineered_features", [])

    print(f"\n  Engineering {len(engineered)} derived features...")

    for feat in engineered:
        name = feat["name"]
        try:
            if name == "loan_to_income":
                df[name] = df["loan_amnt"] / (df["annual_inc"].fillna(0) + 1)
            elif name == "payment_burden":
                df[name] = df["installment"] / (df["annual_inc"].fillna(0) / 12 + 1)
            elif name == "credit_util_dti_interaction":
                df[name] = df["revol_util"].fillna(0) * df["dti"].fillna(0)
            elif name == "fico_mid":
                df[name] = (df["fico_range_low"].fillna(0) + df["fico_range_high"].fillna(0)) / 2
            else:
                print(f"    WARNING: Unknown engineered feature '{name}' — skipping")
                continue
            print(f"    [OK] {name} — mean: {df[name].mean():.4f}, "
                  f"nulls: {df[name].isna().sum()}")
        except KeyError as e:
            print(f"    WARNING: Cannot compute '{name}' — missing column {e}")

    return df


# ============================================================================
# 5. Temporal Split
# ============================================================================

def temporal_split(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by time using issue_d column.

    Split boundaries from config:
        - Train: ≤ 2015-12-31
        - Validation: 2016-01-01 to 2016-12-31
        - Test: ≥ 2017-01-01

    Args:
        df: DataFrame with issue_d column.
        config: Project configuration dictionary.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    lc_cfg = config["datasets"]["lending_club"]

    # Parse issue_d to datetime
    if "issue_d" not in df.columns:
        raise ValueError("Cannot perform temporal split: 'issue_d' column not found. "
                         "Make sure to preserve it during feature selection.")

    df = df.copy()
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="mixed", errors="coerce")

    # Drop rows where date couldn't be parsed
    n_bad_dates = df["issue_d"].isna().sum()
    if n_bad_dates > 0:
        print(f"  WARNING: {n_bad_dates:,} rows with unparseable dates — dropping")
        df = df.dropna(subset=["issue_d"])

    train_end = pd.Timestamp(lc_cfg["train_end"])
    val_start = pd.Timestamp(lc_cfg["val_start"])
    val_end = pd.Timestamp(lc_cfg["val_end"])
    test_start = pd.Timestamp(lc_cfg["test_start"])

    train_df = df[df["issue_d"] <= train_end].copy()
    val_df = df[(df["issue_d"] >= val_start) & (df["issue_d"] <= val_end)].copy()
    test_df = df[df["issue_d"] >= test_start].copy()

    # Drop issue_d after splitting (not a feature)
    for split_df in [train_df, val_df, test_df]:
        if "issue_d" in split_df.columns:
            split_df.drop(columns=["issue_d"], inplace=True)

    print(f"\n  Temporal split:")
    print(f"    Train (<= {train_end.date()}): {len(train_df):,} rows "
          f"({train_df['target'].mean()*100:.1f}% default)")
    print(f"    Val ({val_start.date()} to {val_end.date()}): {len(val_df):,} rows "
          f"({val_df['target'].mean()*100:.1f}% default)")
    print(f"    Test (>= {test_start.date()}): {len(test_df):,} rows "
          f"({test_df['target'].mean()*100:.1f}% default)")

    if len(test_df) == 0:
        print("\n  WARNING: Temporal split yielded empty test set (likely due to nrows limit).")
        print("  Falling back to random split (80/10/10)...")
        from sklearn.model_selection import train_test_split
        # All data is in train_df currently
        temp_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(temp_df, test_size=0.1111, random_state=42) # 0.1111 of 0.9 is ~0.1
        print(f"    New Train: {len(train_df):,} rows")
        print(f"    New Val:   {len(val_df):,} rows")
        print(f"    New Test:  {len(test_df):,} rows")

    return train_df, val_df, test_df


# ============================================================================
# 6. Missing Value Imputation
# ============================================================================

def impute_missing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Impute missing values using training set statistics.

    Strategy:
        - Numeric: median imputation (computed on train, applied to all)
        - Categorical: fill with 'UNKNOWN'

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        config: Project configuration dictionary.

    Returns:
        Tuple of (train_df, val_df, test_df, imputation_stats).
    """
    numeric_cols = config["core_features"]["numeric"]
    categorical_cols = config["core_features"]["categorical"]

    # Add engineered features to numeric list
    engineered_names = [f["name"] for f in config.get("engineered_features", [])]
    all_numeric = [c for c in numeric_cols + engineered_names
                   if c in train_df.columns]
    all_categorical = [c for c in categorical_cols if c in train_df.columns]

    print(f"\n  Imputing missing values...")

    # Compute medians from training set only
    imputation_stats = {}
    medians = {}
    for col in all_numeric:
        median_val = train_df[col].median()
        medians[col] = median_val
        n_missing_train = train_df[col].isna().sum()
        n_missing_val = val_df[col].isna().sum() if col in val_df.columns else 0
        n_missing_test = test_df[col].isna().sum() if col in test_df.columns else 0
        if n_missing_train > 0 or n_missing_val > 0 or n_missing_test > 0:
            imputation_stats[col] = {
                "type": "median",
                "value": median_val,
                "missing_train": int(n_missing_train),
                "missing_val": int(n_missing_val),
                "missing_test": int(n_missing_test),
            }

    # Apply numeric imputation
    for col in all_numeric:
        fill_val = medians[col]
        train_df[col] = train_df[col].fillna(fill_val)
        if col in val_df.columns:
            val_df[col] = val_df[col].fillna(fill_val)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(fill_val)

    # Apply categorical imputation
    for col in all_categorical:
        train_df[col] = train_df[col].fillna("UNKNOWN").astype(str)
        if col in val_df.columns:
            val_df[col] = val_df[col].fillna("UNKNOWN").astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("UNKNOWN").astype(str)

    n_imputed = len(imputation_stats)
    print(f"    Imputed {n_imputed} numeric columns (median from train)")
    print(f"    Filled {len(all_categorical)} categorical columns with 'UNKNOWN'")

    # Report top missing columns
    if imputation_stats:
        print(f"    Top missing columns (train):")
        sorted_stats = sorted(imputation_stats.items(),
                              key=lambda x: x[1]["missing_train"], reverse=True)
        for col, stats in sorted_stats[:5]:
            pct = stats["missing_train"] / len(train_df) * 100
            print(f"      {col}: {stats['missing_train']:,} ({pct:.1f}%) -> "
                  f"filled with {stats['value']:.2f}")

    return train_df, val_df, test_df, imputation_stats


# ============================================================================
# 7. Categorical Encoding
# ============================================================================

def encode_categoricals(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Encode categorical features using ordinal encoding.

    Uses ordinal encoding (label encoding) which works well with
    tree-based models (LightGBM, XGBoost). Categories unseen in
    training are mapped to -1.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        config: Project configuration dictionary.

    Returns:
        Tuple of (train_df, val_df, test_df, encoding_maps).
    """
    from sklearn.preprocessing import OrdinalEncoder

    categorical_cols = [c for c in config["core_features"]["categorical"]
                        if c in train_df.columns]

    print(f"\n  Encoding {len(categorical_cols)} categorical features...")

    encoding_maps = {}

    for col in categorical_cols:
        # Build category list from training set
        categories = sorted(train_df[col].unique().tolist())
        cat_to_int = {cat: i for i, cat in enumerate(categories)}
        encoding_maps[col] = cat_to_int

        # Encode training set
        train_df[col] = train_df[col].map(cat_to_int).fillna(-1).astype(int)

        # Encode val/test (unseen categories → -1)
        if col in val_df.columns:
            val_df[col] = val_df[col].map(cat_to_int).fillna(-1).astype(int)
        if col in test_df.columns:
            test_df[col] = test_df[col].map(cat_to_int).fillna(-1).astype(int)

        n_cats = len(categories)
        print(f"    {col}: {n_cats} categories")

    return train_df, val_df, test_df, encoding_maps


# ============================================================================
# 8. Separate Features and Target
# ============================================================================

def separate_features_target(
    df: pd.DataFrame,
    text_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features from target column.

    Drops target and any text columns (handled separately in text pipeline)
    from the feature matrix.

    Args:
        df: DataFrame with features and target.
        text_cols: Text columns to exclude from tabular features.

    Returns:
        Tuple of (X, y).
    """
    drop_cols = ["target"]
    if text_cols:
        drop_cols.extend([c for c in text_cols if c in df.columns])

    y = df["target"].copy()
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y


# ============================================================================
# 9. SMOTETomek Resampling
# ============================================================================

def apply_smote_tomek(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTETomek resampling to training data.

    Combines SMOTE (oversampling minority) with Tomek links
    (cleaning boundary samples). Applied ONLY to training set.

    Args:
        X_train: Training features.
        y_train: Training target.
        config: Project configuration dictionary.

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    from imblearn.combine import SMOTETomek

    seed = config["imbalance"].get("random_state", 42)

    print(f"\n  Applying SMOTETomek resampling...")
    print(f"    Before: {len(X_train):,} samples "
          f"(Default: {(y_train==1).sum():,}, "
          f"Non-default: {(y_train==0).sum():,})")

    smote_tomek = SMOTETomek(random_state=seed, n_jobs=-1)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name="target")

    print(f"    After:  {len(X_resampled):,} samples "
          f"(Default: {(y_resampled==1).sum():,}, "
          f"Non-default: {(y_resampled==0).sum():,})")

    return X_resampled, y_resampled


# ============================================================================
# 10. Save / Load Utilities
# ============================================================================

def save_processed_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict,
    encoding_maps: Optional[dict] = None,
    imputation_stats: Optional[dict] = None,
    text_dfs: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None,
    suffix: str = "",
) -> None:
    """Save processed splits to data/processed/.

    Saves as parquet files for fast I/O and type preservation.
    Also saves encoding maps and imputation stats as pickle.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        config: Project configuration dictionary.
        encoding_maps: Category → int mappings.
        imputation_stats: Imputation values used.
        suffix: Optional suffix for filenames (e.g., '_smote').
    """
    save_dir = os.path.join(PROJECT_ROOT, config["paths"]["processed_data"])
    os.makedirs(save_dir, exist_ok=True)

    # Save feature matrices
    X_train.to_parquet(os.path.join(save_dir, f"X_train{suffix}.parquet"))
    X_val.to_parquet(os.path.join(save_dir, f"X_val{suffix}.parquet"))
    X_test.to_parquet(os.path.join(save_dir, f"X_test{suffix}.parquet"))

    # Save targets as CSV (simple)
    y_train.to_csv(os.path.join(save_dir, f"y_train{suffix}.csv"), index=False)
    y_val.to_csv(os.path.join(save_dir, f"y_val{suffix}.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, f"y_test{suffix}.csv"), index=False)

    # Save metadata
    if encoding_maps:
        with open(os.path.join(save_dir, "encoding_maps.pkl"), "wb") as f:
            pickle.dump(encoding_maps, f)

    if imputation_stats:
        with open(os.path.join(save_dir, "imputation_stats.pkl"), "wb") as f:
            pickle.dump(imputation_stats, f)

    if text_dfs is not None:
        text_train, text_val, text_test = text_dfs
        text_train.to_parquet(os.path.join(save_dir, f"text_train{suffix}.parquet"))
        text_val.to_parquet(os.path.join(save_dir, f"text_val{suffix}.parquet"))
        text_test.to_parquet(os.path.join(save_dir, f"text_test{suffix}.parquet"))
        print(f"    Saved text_dfs to text_train{suffix}.parquet, etc.")

    print(f"\n  Saved processed data to {save_dir}/")
    print(f"    X_train{suffix}: {X_train.shape}")
    print(f"    X_val{suffix}:   {X_val.shape}")
    print(f"    X_test{suffix}:  {X_test.shape}")


def load_processed_data(
    config: dict,
    suffix: str = "",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series]:
    """Load processed splits from data/processed/.

    Args:
        config: Project configuration dictionary.
        suffix: Filename suffix (e.g., '_smote').

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    load_dir = os.path.join(PROJECT_ROOT, config["paths"]["processed_data"])

    X_train = pd.read_parquet(os.path.join(load_dir, f"X_train{suffix}.parquet"))
    X_val = pd.read_parquet(os.path.join(load_dir, f"X_val{suffix}.parquet"))
    X_test = pd.read_parquet(os.path.join(load_dir, f"X_test{suffix}.parquet"))

    y_train = pd.read_csv(os.path.join(load_dir, f"y_train{suffix}.csv")).squeeze()
    y_val = pd.read_csv(os.path.join(load_dir, f"y_val{suffix}.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(load_dir, f"y_test{suffix}.csv")).squeeze()

    print(f"  Loaded processed data from {load_dir}/")
    print(f"    X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"    X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    print(f"    X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# 11. Full Pipeline
# ============================================================================

class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for LendingClub data.

    Orchestrates the full pipeline from raw CSV to model-ready splits:
        Load → Map target → Select features → Engineer features →
        Temporal split → Impute → Encode → (Optional) SMOTE → Save

    Attributes:
        config: Project configuration dictionary.
        encoding_maps: Category encoding mappings (available after run).
        imputation_stats: Imputation statistics (available after run).
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.encoding_maps = None
        self.imputation_stats = None

    def run(
        self,
        nrows: Optional[int] = None,
        apply_smote: bool = True,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Run the full preprocessing pipeline.

        Args:
            nrows: Optional row limit for development.
            apply_smote: Whether to apply SMOTETomek resampling.
            save: Whether to save processed data to disk.

        Returns:
            Dictionary with keys:
                'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
                'X_train_smote', 'y_train_smote' (if apply_smote=True),
                'encoding_maps', 'imputation_stats', 'text_cols'
        """
        print("=" * 60)
        print("  HybridCreditLLM — Preprocessing Pipeline")
        print("=" * 60)

        # 1. Load raw data
        df = load_lending_club(self.config, nrows=nrows)

        # 2. Map target variable
        df = map_target_variable(df, self.config)

        # 3. Select features
        df = select_features(df, self.config)

        # 4. Engineer features
        df = engineer_features(df, self.config)

        # 5. Temporal split
        train_df, val_df, test_df = temporal_split(df, self.config)

        # Identify text columns (keep them in the DataFrame but exclude from X)
        text_primary = self.config["datasets"]["lending_club"].get(
            "text_primary", "desc")
        text_fallback = self.config["datasets"]["lending_club"].get(
            "text_fallback", [])
        text_cols = [c for c in [text_primary] + text_fallback
                     if c in train_df.columns]

        # 6. Impute missing values
        train_df, val_df, test_df, self.imputation_stats = impute_missing(
            train_df, val_df, test_df, self.config
        )

        # Extract text_dfs BEFORE categorical encoding (which changes strings to ints)
        text_dfs = (
            train_df[text_cols].copy(),
            val_df[text_cols].copy(),
            test_df[text_cols].copy()
        )

        # 7. Encode categoricals
        train_df, val_df, test_df, self.encoding_maps = encode_categoricals(
            train_df, val_df, test_df, self.config
        )

        # 8. Separate features and target
        X_train, y_train = separate_features_target(train_df, text_cols)
        X_val, y_val = separate_features_target(val_df, text_cols)
        X_test, y_test = separate_features_target(test_df, text_cols)

        result = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "encoding_maps": self.encoding_maps,
            "imputation_stats": self.imputation_stats,
            "text_cols": text_cols,
            "text_dfs": text_dfs,
        }

        # 9. Optional SMOTETomek
        if apply_smote:
            X_train_smote, y_train_smote = apply_smote_tomek(
                X_train, y_train, self.config
            )
            result["X_train_smote"] = X_train_smote
            result["y_train_smote"] = y_train_smote

        # 10. Save
        if save:
            save_processed_data(
                X_train, y_train, X_val, y_val, X_test, y_test,
                self.config,
                encoding_maps=self.encoding_maps,
                imputation_stats=self.imputation_stats,
                text_dfs=text_dfs,
            )
            if apply_smote:
                save_processed_data(
                    X_train_smote, y_train_smote, X_val, y_val, X_test, y_test,
                    self.config,
                    suffix="_smote",
                )

        print("\n" + "=" * 60)
        print("  Pipeline complete!")
        print("=" * 60)

        return result


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HybridCreditLLM — Data Preprocessing Pipeline"
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Limit rows for development (default: all)"
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Skip SMOTETomek resampling"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving processed data"
    )

    args = parser.parse_args()

    pipeline = PreprocessingPipeline()
    result = pipeline.run(
        nrows=args.nrows,
        apply_smote=not args.no_smote,
        save=not args.no_save,
    )
