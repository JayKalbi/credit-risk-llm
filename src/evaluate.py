"""
HybridCreditLLM — Evaluation Module
======================================
Computes all target metrics and generates evaluation visualizations.

Target Metrics (from config.yaml):
    - AUC-ROC    > 0.82
    - PR-AUC     > 0.65
    - KS Stat    > 0.40
    - Brier Score < 0.12
    - F1 (Default) > 0.70

Usage:
    from src.evaluate import evaluate_model, compare_models, plot_roc_curves
"""

import os
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
# Core Metrics
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all five target metrics for credit risk evaluation.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities of the positive class (default).
        threshold: Decision threshold for classification metrics.

    Returns:
        Dictionary with keys: auc_roc, pr_auc, ks_stat, brier_score, f1_default.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
    )

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "ks_stat": _compute_ks_statistic(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "f1_default": f1_score(y_true, y_pred, pos_label=1),
    }

    return metrics


def _compute_ks_statistic(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Compute the Kolmogorov-Smirnov statistic.

    KS Stat measures the maximum separation between the cumulative
    distributions of default and non-default predicted probabilities.
    Higher KS → better discrimination.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.

    Returns:
        KS statistic value (0 to 1).
    """
    # Get probabilities for each class
    prob_default = y_prob[y_true == 1]
    prob_non_default = y_prob[y_true == 0]

    # Sort all probabilities
    all_probs = np.sort(np.unique(np.concatenate([prob_default, prob_non_default])))

    # Compute CDFs
    ks_values = []
    for t in all_probs:
        cdf_default = np.mean(prob_default <= t)
        cdf_non_default = np.mean(prob_non_default <= t)
        ks_values.append(abs(cdf_default - cdf_non_default))

    return max(ks_values) if ks_values else 0.0


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find the optimal classification threshold.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        metric: Optimization target ('f1', 'youden', 'precision_recall').

    Returns:
        Optimal threshold value.
    """
    from sklearn.metrics import f1_score, roc_curve, precision_recall_curve

    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        return thresholds[best_idx]

    elif metric == "f1":
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, pos_label=1))
        return thresholds[np.argmax(f1_scores)]

    elif metric == "precision_recall":
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # Find threshold where precision ≈ recall
        f1_vals = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_vals[:-1])  # last threshold is always 1
        return thresholds[best_idx]

    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: Optional[dict] = None,
    threshold: float = 0.5,
    print_results: bool = True,
) -> Dict[str, Any]:
    """Evaluate a model against all target metrics.

    Args:
        model_name: Human-readable model name.
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        config: Project configuration (for target values).
        threshold: Decision threshold.
        print_results: Whether to print results.

    Returns:
        Dictionary with metrics and pass/fail against targets.
    """
    if config is None:
        config = load_config()

    metrics = compute_metrics(y_true, y_prob, threshold)
    targets = config.get("target_metrics", {})

    result = {
        "model_name": model_name,
        "threshold": threshold,
        "metrics": metrics,
        "targets": targets,
        "pass_fail": {},
    }

    # Check against targets
    checks = {
        "auc_roc": (">=", targets.get("auc_roc", 0)),
        "pr_auc": (">=", targets.get("pr_auc", 0)),
        "ks_stat": (">=", targets.get("ks_stat", 0)),
        "brier_score": ("<=", targets.get("brier_score", 1)),
        "f1_default": (">=", targets.get("f1_default", 0)),
    }

    for metric_name, (op, target) in checks.items():
        value = metrics[metric_name]
        if op == ">=":
            passed = value >= target
        else:
            passed = value <= target
        result["pass_fail"][metric_name] = passed

    if print_results:
        _print_evaluation(result)

    return result


def _print_evaluation(result: Dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  Evaluation: {result['model_name']}")
    print(f"  Threshold: {result['threshold']:.2f}")
    print(f"{'='*60}")

    metrics = result["metrics"]
    targets = result["targets"]
    pass_fail = result["pass_fail"]

    header = f"  {'Metric':<20} {'Value':>8} {'Target':>10} {'Status':>8}"
    print(header)
    print(f"  {'-'*46}")

    format_map = {
        "auc_roc": ("AUC-ROC", ">="),
        "pr_auc": ("PR-AUC", ">="),
        "ks_stat": ("KS Statistic", ">="),
        "brier_score": ("Brier Score", "<="),
        "f1_default": ("F1 (Default)", ">="),
    }

    for key, (label, op) in format_map.items():
        value = metrics[key]
        target = targets.get(key, "-")
        passed = pass_fail.get(key, None)

        if target != "-":
            target_str = f"{op} {target:.2f}"
        else:
            target_str = "-"

        status = "PASS" if passed else "FAIL" if passed is not None else "-"
        status_symbol = "[OK]" if passed else "[X]" if passed is not None else "[-]"

        print(f"  {label:<20} {value:>8.4f} {target_str:>10} {status_symbol:>8}")

    n_pass = sum(1 for v in pass_fail.values() if v)
    n_total = len(pass_fail)
    print(f"\n  Score: {n_pass}/{n_total} metrics passed")


# ============================================================================
# Multi-Model Comparison
# ============================================================================

def compare_models(
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Create a comparison table across multiple models.

    Args:
        results: List of evaluation result dicts from evaluate_model().

    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    rows = []
    for r in results:
        row = {"Model": r["model_name"]}
        row.update(r["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("Model")

    # Rename columns for display
    col_rename = {
        "auc_roc": "AUC-ROC",
        "pr_auc": "PR-AUC",
        "ks_stat": "KS Stat",
        "brier_score": "Brier",
        "f1_default": "F1 (Def)",
    }
    df = df.rename(columns=col_rename)

    return df


# ============================================================================
# Visualization
# ============================================================================

def plot_roc_curves(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC curves for multiple models.

    Args:
        models: Dict of {model_name: (y_true, y_prob)}.
        save_path: Path to save figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for (name, (y_true, y_prob)), color in zip(models.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved ROC curves: {save_path}")
    plt.close(fig)


def plot_pr_curves(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
) -> None:
    """Plot Precision-Recall curves for multiple models.

    Args:
        models: Dict of {model_name: (y_true, y_prob)}.
        save_path: Path to save figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for (name, (y_true, y_prob)), color in zip(models.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})",
                color=color, linewidth=2)

    # Baseline (random classifier)
    baseline = np.mean(list(models.values())[0][0])
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5,
               label=f"Baseline ({baseline:.2f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved PR curves: {save_path}")
    plt.close(fig)


def plot_calibration_curve(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Plot calibration curves (reliability diagrams).

    Shows how well predicted probabilities match actual frequencies.

    Args:
        models: Dict of {model_name: (y_true, y_prob)}.
        n_bins: Number of probability bins.
        save_path: Path to save figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for (name, (y_true, y_prob)), color in zip(models.items(), colors):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, label=name, color=color,
                linewidth=2, marker="o", markersize=5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1,
            label="Perfect calibration")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved calibration curves: {save_path}")
    plt.close(fig)


def plot_ks_chart(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot the KS (Kolmogorov-Smirnov) chart.

    Shows the separation between default and non-default CDFs.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        model_name: Model name for title.
        save_path: Path to save figure.
    """
    import matplotlib.pyplot as plt

    prob_default = np.sort(y_prob[y_true == 1])
    prob_non_default = np.sort(y_prob[y_true == 0])

    # Compute CDFs
    cdf_default = np.arange(1, len(prob_default) + 1) / len(prob_default)
    cdf_non_default = np.arange(1, len(prob_non_default) + 1) / len(prob_non_default)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(prob_default, cdf_default, label="Default (1)",
            color="#e74c3c", linewidth=2)
    ax.plot(prob_non_default, cdf_non_default, label="Non-default (0)",
            color="#2ecc71", linewidth=2)

    # Find KS point
    ks_stat = _compute_ks_statistic(y_true, y_prob)
    ax.set_title(f"KS Chart — {model_name} (KS={ks_stat:.4f})", fontsize=14)
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Cumulative Proportion", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved KS chart: {save_path}")
    plt.close(fig)


def save_results(
    results: List[Dict[str, Any]],
    config: dict,
    filename: str = "evaluation_results.csv",
) -> str:
    """Save evaluation results to CSV.

    Args:
        results: List of evaluation result dicts.
        config: Project configuration.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    df = compare_models(results)

    save_dir = os.path.join(PROJECT_ROOT, config["paths"]["results"])
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    df.to_csv(filepath)
    print(f"  Saved evaluation results: {filepath}")
    return filepath


def generate_latex_table(
    results: List[Dict[str, Any]],
    bold_best: bool = True,
) -> str:
    """Generate a LaTeX table from evaluation results.

    For direct inclusion in the paper's main results table.

    Args:
        results: List of evaluation result dicts.
        bold_best: Whether to bold the best value in each column.

    Returns:
        LaTeX table string.
    """
    df = compare_models(results)

    # Find best values
    best_cols = {}
    for col in df.columns:
        if col == "Brier":
            best_cols[col] = df[col].idxmin()  # Lower is better
        else:
            best_cols[col] = df[col].idxmax()  # Higher is better

    lines = []
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{AUC-ROC} & \\textbf{PR-AUC} & "
                  "\\textbf{KS Stat} & \\textbf{Brier} & \\textbf{F1 (Def)} \\\\")
    lines.append("\\midrule")

    for model_name, row in df.iterrows():
        cells = [model_name.replace("_", "\\_")]
        for col in df.columns:
            val = f"{row[col]:.4f}"
            if bold_best and best_cols[col] == model_name:
                val = f"\\textbf{{{val}}}"
            cells.append(val)
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)
