"""
HybridCreditLLM — Model Training Module
=========================================
Trains classical ML baselines and manages model persistence.

Models:
    - Logistic Regression (interpretable baseline)
    - XGBoost (gradient boosting baseline)
    - LightGBM (primary tabular model)

All models support Optuna hyperparameter tuning.

Usage:
    from src.models import train_lightgbm, train_xgboost, train_logistic
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
# Logistic Regression
# ============================================================================

def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> Any:
    """Train a Logistic Regression baseline.

    Args:
        X_train: Training features (numeric only).
        y_train: Training target.
        config: Project configuration dictionary.
        X_val: Validation features (for reporting).
        y_val: Validation target (for reporting).

    Returns:
        Fitted LogisticRegression model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    lr_cfg = config["classical_ml"]["logistic"]
    seed = config["classical_ml"]["random_state"]

    # Logistic Regression requires scaled features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        C=lr_cfg["C"],
        random_state=seed,
        solver="lbfgs",
        n_jobs=-1,
    )

    print("Training Logistic Regression...")
    model.fit(X_train_scaled, y_train)

    # Store scaler as attribute for inference
    model._scaler = scaler

    train_score = model.score(X_train_scaled, y_train)
    print(f"  Train accuracy: {train_score:.4f}")

    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        val_score = model.score(X_val_scaled, y_val)
        print(f"  Val accuracy:   {val_score:.4f}")

    return model


def predict_logistic(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict probabilities using a trained Logistic Regression model.

    Args:
        model: Fitted LogisticRegression with attached _scaler.
        X: Features DataFrame.

    Returns:
        Predicted probabilities of the positive class (default).
    """
    X_scaled = model._scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


# ============================================================================
# XGBoost
# ============================================================================

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    tune: bool = False,
) -> Any:
    """Train an XGBoost classifier with optional Optuna tuning.

    Args:
        X_train: Training features.
        y_train: Training target.
        config: Project configuration dictionary.
        X_val: Validation features.
        y_val: Validation target.
        tune: Whether to run Optuna hyperparameter search.

    Returns:
        Fitted XGBClassifier model.
    """
    import xgboost as xgb

    xgb_cfg = config["classical_ml"]["xgboost"]
    seed = config["classical_ml"]["random_state"]

    if tune and X_val is not None:
        print("Running Optuna tuning for XGBoost...")
        best_params = _tune_xgboost(X_train, y_train, X_val, y_val, config)
        xgb_cfg = {**xgb_cfg, **best_params}

    # Compute scale_pos_weight if auto
    if xgb_cfg.get("scale_pos_weight") == "auto":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
    else:
        scale_pos_weight = xgb_cfg.get("scale_pos_weight", 1.0)

    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        learning_rate=xgb_cfg["learning_rate"],
        max_depth=xgb_cfg["max_depth"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        tree_method="hist",
    )

    print("Training XGBoost...")
    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=100,
    )

    return model


def _tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> dict:
    """Run Optuna hyperparameter search for XGBoost.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        config: Project configuration dictionary.

    Returns:
        Dictionary of best hyperparameters.
    """
    import optuna
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = config["classical_ml"]["random_state"]
    n_trials = config["classical_ml"]["optuna"]["n_trials"]
    timeout = config["classical_ml"]["optuna"]["timeout"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()

        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=seed,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print(f"  Best AUC-ROC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ============================================================================
# LightGBM
# ============================================================================

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    tune: bool = False,
) -> Any:
    """Train a LightGBM classifier with optional Optuna tuning.

    Args:
        X_train: Training features.
        y_train: Training target.
        config: Project configuration dictionary.
        X_val: Validation features.
        y_val: Validation target.
        tune: Whether to run Optuna hyperparameter search.

    Returns:
        Fitted LGBMClassifier model.
    """
    import lightgbm as lgb

    lgb_cfg = config["classical_ml"]["lightgbm"]
    seed = config["classical_ml"]["random_state"]

    if tune and X_val is not None:
        print("Running Optuna tuning for LightGBM...")
        best_params = _tune_lightgbm(X_train, y_train, X_val, y_val, config)
        lgb_cfg = {**lgb_cfg, **best_params}

    model = lgb.LGBMClassifier(
        n_estimators=lgb_cfg["n_estimators"],
        learning_rate=lgb_cfg["learning_rate"],
        max_depth=lgb_cfg["max_depth"],
        num_leaves=lgb_cfg["num_leaves"],
        subsample=lgb_cfg["subsample"],
        colsample_bytree=lgb_cfg["colsample_bytree"],
        class_weight=lgb_cfg["class_weight"],
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    print("Training LightGBM...")
    eval_set = [(X_val, y_val)] if X_val is not None else None
    callbacks = []
    if eval_set:
        callbacks.append(lgb.early_stopping(lgb_cfg["early_stopping_rounds"]))
        callbacks.append(lgb.log_evaluation(100))

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=callbacks if callbacks else None,
    )

    return model


def _tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> dict:
    """Run Optuna hyperparameter search for LightGBM.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        config: Project configuration dictionary.

    Returns:
        Dictionary of best hyperparameters.
    """
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = config["classical_ml"]["random_state"]
    n_trials = config["classical_ml"]["optuna"]["n_trials"]
    timeout = config["classical_ml"]["optuna"]["timeout"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = lgb.LGBMClassifier(
            **params,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0),  # silent
            ],
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print(f"  Best AUC-ROC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ============================================================================
# Model Persistence
# ============================================================================

def save_model(model: Any, name: str, config: dict) -> str:
    """Save a trained model to disk.

    Args:
        model: Trained model object.
        name: Model name (used as filename).
        config: Project configuration dictionary.

    Returns:
        Path to the saved model file.
    """
    model_dir = os.path.join(PROJECT_ROOT, config["paths"]["models"])
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, f"{name}.pkl")

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved model: {filepath} ({size_mb:.1f} MB)")
    return filepath


def load_model(name: str, config: dict) -> Any:
    """Load a trained model from disk.

    Args:
        name: Model name (filename without extension).
        config: Project configuration dictionary.

    Returns:
        Loaded model object.
    """
    filepath = os.path.join(
        PROJECT_ROOT, config["paths"]["models"], f"{name}.pkl"
    )
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"  Loaded model: {filepath}")
    return model


# ============================================================================
# Feature Importance
# ============================================================================

def get_feature_importance(
    model: Any,
    feature_names: list,
    top_n: int = 20,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Extract feature importance from tree-based models.

    Args:
        model: Trained LightGBM or XGBoost model.
        feature_names: List of feature names.
        top_n: Number of top features to return.
        importance_type: Type of importance ('gain', 'split', 'weight').

    Returns:
        DataFrame with feature names and importance scores, sorted descending.
    """
    import_attr = getattr(model, "feature_importances_", None)

    if import_attr is not None:
        importance = import_attr
    else:
        raise ValueError("Model does not have feature_importances_ attribute")

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    return df.head(top_n).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot horizontal bar chart of feature importance.

    Args:
        importance_df: DataFrame from get_feature_importance().
        model_name: Model name for title.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        palette="viridis",
        ax=ax,
    )
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved importance plot: {save_path}")

    plt.close(fig)
