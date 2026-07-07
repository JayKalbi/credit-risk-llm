"""
HybridCreditLLM — Explainability Module
==========================================
SHAP explanations, DiCE counterfactuals, faithfulness auditing,
and hallucination detection for LLM rationales.

Components:
    1. SHAP TreeExplainer — Feature attribution for tree models
    2. DiCE-ML — Counterfactual explanations
    3. Faithfulness Audit — Kendall's Tau between LLM rationale and SHAP
    4. Hallucination Detection — Fact-checking LLM rationales against data

Usage:
    from src.explain import (
        compute_shap_values, generate_counterfactuals,
        faithfulness_audit, detect_hallucinations
    )
"""

import os
import re
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
# 1. SHAP Explanations
# ============================================================================

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    config: Optional[dict] = None,
    max_samples: Optional[int] = None,
) -> Any:
    """Compute SHAP values using TreeExplainer.

    Uses TreeExplainer for LightGBM/XGBoost (exact, fast).
    Falls back to KernelExplainer for other model types.

    Args:
        model: Trained tree-based model.
        X: Feature matrix.
        config: Project configuration (for max_samples).
        max_samples: Override max samples to explain.

    Returns:
        shap.Explanation object with .values, .base_values, .data.
    """
    import shap

    if config and max_samples is None:
        max_samples = config.get("xai", {}).get("shap", {}).get("max_samples", 10000)

    if max_samples and len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
        print(f"  Subsampled to {max_samples:,} rows for SHAP computation")
    else:
        X_sample = X

    print(f"  Computing SHAP values for {len(X_sample):,} samples...")

    # Try TreeExplainer first (fast, exact for tree models)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        print(f"  Used TreeExplainer (exact)")
    except Exception:
        # Fallback to KernelExplainer
        print(f"  TreeExplainer failed, falling back to KernelExplainer...")
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values_raw = explainer.shap_values(X_sample)
        # For binary classification, take the positive class
        if isinstance(shap_values_raw, list):
            shap_values_raw = shap_values_raw[1]
        shap_values = shap.Explanation(
            values=shap_values_raw,
            base_values=explainer.expected_value[1] if isinstance(
                explainer.expected_value, list) else explainer.expected_value,
            data=X_sample.values,
            feature_names=X_sample.columns.tolist(),
        )
        print(f"  Used KernelExplainer (approximate)")

    return shap_values


def plot_shap_summary(
    shap_values: Any,
    save_path: Optional[str] = None,
    max_display: int = 20,
) -> None:
    """Plot SHAP summary (beeswarm) plot.

    Args:
        shap_values: SHAP Explanation object.
        save_path: Path to save the figure.
        max_display: Number of top features to show.
    """
    import matplotlib.pyplot as plt
    import shap

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    shap.summary_plot(shap_values, max_display=max_display, show=False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved SHAP summary plot: {save_path}")

    plt.close("all")


def plot_shap_waterfall(
    shap_values: Any,
    idx: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """Plot SHAP waterfall for a single prediction.

    Args:
        shap_values: SHAP Explanation object.
        idx: Index of the sample to explain.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import shap

    shap.waterfall_plot(shap_values[idx], show=False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved SHAP waterfall: {save_path}")

    plt.close("all")


def get_shap_feature_ranking(
    shap_values: Any,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Get features ranked by mean absolute SHAP value.

    Args:
        shap_values: SHAP Explanation object.
        feature_names: Optional feature name override.

    Returns:
        DataFrame with feature names and mean |SHAP| values, sorted desc.
    """
    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    # Handle multi-output (binary classification)
    if values.ndim == 3:
        values = values[:, :, 1]  # Positive class

    mean_abs_shap = np.abs(values).mean(axis=0)

    if feature_names is None:
        if hasattr(shap_values, "feature_names"):
            feature_names = shap_values.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    df["rank"] = range(1, len(df) + 1)

    return df


# ============================================================================
# 2. DiCE Counterfactual Explanations
# ============================================================================

def generate_counterfactuals(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    query_indices: Optional[List[int]] = None,
    num_cf: int = 5,
    config: Optional[dict] = None,
) -> List[pd.DataFrame]:
    """Generate counterfactual explanations using DiCE-ML.

    Answers: "What minimal changes would flip the prediction?"

    Args:
        model: Trained model with predict_proba method.
        X: Feature matrix.
        y: True labels (for context).
        query_indices: Indices of samples to explain (default: first 5 defaults).
        num_cf: Number of counterfactuals per instance.
        config: Project configuration.

    Returns:
        List of DataFrames, one per query instance.
    """
    import dice_ml

    if config is None:
        config = load_config()

    dice_cfg = config.get("xai", {}).get("dice", {})
    num_cf = dice_cfg.get("num_counterfactuals", num_cf)
    diversity = dice_cfg.get("diversity_weight", 1.0)

    print(f"  Generating {num_cf} counterfactuals per instance...")

    # Identify numeric and categorical features
    numeric_cols = [c for c in config["core_features"]["numeric"]
                    if c in X.columns]
    engineered_names = [f["name"] for f in config.get("engineered_features", [])]
    numeric_cols += [c for c in engineered_names if c in X.columns]

    categorical_cols = [c for c in config["core_features"]["categorical"]
                        if c in X.columns]

    # Build a combined DataFrame
    data_df = X.copy()
    data_df["target"] = y.values

    # Create DiCE data object
    d = dice_ml.Data(
        dataframe=data_df,
        continuous_features=numeric_cols,
        outcome_name="target",
    )

    # Create DiCE model object
    m = dice_ml.Model(model=model, backend="sklearn")

    # Create explainer
    exp = dice_ml.Dice(d, m, method="random")

    # Select query instances
    if query_indices is None:
        # Default: explain first 5 default cases
        default_indices = y[y == 1].index[:5].tolist()
        query_indices = default_indices

    results = []
    for idx in query_indices:
        query = X.loc[[idx]]
        try:
            cf = exp.generate_counterfactuals(
                query,
                total_CFs=num_cf,
                desired_class="opposite",
                diversity_weight=diversity,
            )
            cf_df = cf.cf_examples_list[0].final_cfs_df
            results.append(cf_df)
            print(f"    Instance {idx}: generated {len(cf_df)} counterfactuals")
        except Exception as e:
            print(f"    Instance {idx}: FAILED — {e}")
            results.append(None)

    return results


# ============================================================================
# 3. Faithfulness Audit
# ============================================================================

def faithfulness_audit(
    shap_ranking: pd.DataFrame,
    llm_rationale: str,
    feature_names: List[str],
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Audit LLM rationale faithfulness against SHAP attributions.

    Measures alignment between the features the LLM cites as important
    and the features SHAP identifies as actually important.

    Uses Kendall's Tau rank correlation.

    Args:
        shap_ranking: DataFrame from get_shap_feature_ranking().
        llm_rationale: LLM-generated rationale text.
        feature_names: List of all feature names.
        config: Project configuration.

    Returns:
        Dict with 'kendall_tau', 'top3_overlap', 'cited_features',
        'shap_top_features', 'passed'.
    """
    from scipy.stats import kendalltau

    if config is None:
        config = load_config()

    threshold = config.get("xai", {}).get("faithfulness", {}).get("threshold", 0.50)

    # Extract features mentioned in the LLM rationale
    cited_features = _extract_cited_features(llm_rationale, feature_names)

    # Get SHAP top features
    shap_top = shap_ranking.head(10)["feature"].tolist()

    # Build rank vectors for features that appear in both
    all_mentioned = list(set(cited_features + shap_top))

    if len(all_mentioned) < 2:
        return {
            "kendall_tau": 0.0,
            "p_value": 1.0,
            "top3_overlap": 0,
            "cited_features": cited_features,
            "shap_top_features": shap_top[:3],
            "passed": False,
            "message": "Too few features to compare",
        }

    # Build rank vectors
    shap_ranks = {}
    for _, row in shap_ranking.iterrows():
        shap_ranks[row["feature"]] = row["rank"]

    # LLM rank = order of first mention
    llm_ranks = {}
    for i, feat in enumerate(cited_features):
        if feat not in llm_ranks:
            llm_ranks[feat] = i + 1

    # Build aligned rank vectors
    common_features = [f for f in all_mentioned if f in shap_ranks and f in llm_ranks]

    if len(common_features) < 2:
        top3_overlap = len(set(cited_features[:3]) & set(shap_top[:3]))
        return {
            "kendall_tau": 0.0,
            "p_value": 1.0,
            "top3_overlap": top3_overlap,
            "cited_features": cited_features,
            "shap_top_features": shap_top[:3],
            "passed": False,
            "message": "Insufficient overlap between SHAP and LLM features",
        }

    shap_rank_vec = [shap_ranks[f] for f in common_features]
    llm_rank_vec = [llm_ranks[f] for f in common_features]

    tau, p_value = kendalltau(shap_rank_vec, llm_rank_vec)

    top3_overlap = len(set(cited_features[:3]) & set(shap_top[:3]))

    result = {
        "kendall_tau": tau,
        "p_value": p_value,
        "top3_overlap": top3_overlap,
        "cited_features": cited_features,
        "shap_top_features": shap_top[:3],
        "common_features": common_features,
        "passed": tau >= threshold,
        "threshold": threshold,
    }

    return result


def _extract_cited_features(
    rationale: str,
    feature_names: List[str],
) -> List[str]:
    """Extract feature names mentioned in LLM rationale text.

    Maps natural language to feature names. Handles common synonyms:
        - "FICO score" → fico_mid or fico_range_low
        - "DTI" → dti
        - "interest rate" → int_rate
        - etc.

    Args:
        rationale: LLM rationale text.
        feature_names: List of valid feature names.

    Returns:
        List of mentioned feature names, in order of first appearance.
    """
    rationale_lower = rationale.lower()

    # Mapping from natural language → feature name
    synonym_map = {
        "fico": ["fico_mid", "fico_range_low", "fico_range_high"],
        "fico score": ["fico_mid"],
        "credit score": ["fico_mid"],
        "dti": ["dti"],
        "debt-to-income": ["dti"],
        "debt to income": ["dti"],
        "interest rate": ["int_rate"],
        "loan amount": ["loan_amnt"],
        "annual income": ["annual_inc"],
        "income": ["annual_inc"],
        "revolving": ["revol_util", "revol_bal"],
        "revolving utilization": ["revol_util"],
        "revolving balance": ["revol_bal"],
        "grade": ["grade", "sub_grade"],
        "loan grade": ["grade"],
        "employment": ["emp_length"],
        "employment length": ["emp_length"],
        "home ownership": ["home_ownership"],
        "delinquency": ["delinq_2yrs"],
        "delinquencies": ["delinq_2yrs"],
        "inquiries": ["inq_last_6mths"],
        "public records": ["pub_rec"],
        "total accounts": ["total_acc"],
        "loan-to-income": ["loan_to_income"],
        "payment burden": ["payment_burden"],
        "installment": ["installment"],
        "purpose": ["purpose"],
    }

    found_features = []
    found_positions = {}

    # Check synonyms
    for phrase, features in synonym_map.items():
        pos = rationale_lower.find(phrase)
        if pos >= 0:
            for feat in features:
                if feat in feature_names and feat not in found_positions:
                    found_positions[feat] = pos

    # Also check exact feature names
    for feat in feature_names:
        feat_lower = feat.lower().replace("_", " ")
        pos = rationale_lower.find(feat_lower)
        if pos >= 0 and feat not in found_positions:
            found_positions[feat] = pos

    # Sort by position of first mention
    found_features = sorted(found_positions.keys(),
                            key=lambda f: found_positions[f])

    return found_features


def batch_faithfulness_audit(
    shap_values: Any,
    X: pd.DataFrame,
    rationales: List[str],
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Run faithfulness audit across multiple instances.

    Args:
        shap_values: SHAP Explanation object.
        X: Feature matrix (aligned with shap_values).
        rationales: List of LLM rationale strings.
        config: Project configuration.

    Returns:
        DataFrame with per-instance audit results.
    """
    feature_names = X.columns.tolist()
    results = []

    for i, rationale in enumerate(rationales):
        if i >= len(shap_values.values):
            break

        # Get per-instance SHAP ranking
        instance_shap = np.abs(shap_values.values[i])
        ranking = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": instance_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        ranking["rank"] = range(1, len(ranking) + 1)

        audit = faithfulness_audit(ranking, rationale, feature_names, config)
        audit["instance_idx"] = i
        results.append(audit)

    df = pd.DataFrame(results)
    n_passed = df["passed"].sum()
    n_total = len(df)
    avg_tau = df["kendall_tau"].mean()

    print(f"\n  Batch Faithfulness Audit:")
    print(f"    Instances: {n_total}")
    print(f"    Passed: {n_passed}/{n_total} ({n_passed/n_total*100:.1f}%)")
    print(f"    Mean Kendall's Tau: {avg_tau:.4f}")

    return df


# ============================================================================
# 4. Hallucination Detection
# ============================================================================

def detect_hallucinations(
    rationale: str,
    row: pd.Series,
    tolerance: float = 0.10,
) -> Dict[str, Any]:
    """Detect factual hallucinations in LLM rationales.

    Checks whether numerical claims in the rationale match the
    actual data values. Catches:
        - Wrong dollar amounts
        - Wrong percentages (DTI, interest rate, utilization)
        - Wrong FICO scores
        - Fabricated facts

    Args:
        rationale: LLM-generated rationale text.
        row: Original data row with true values.
        tolerance: Relative tolerance for number matching (0.10 = 10%).

    Returns:
        Dict with 'hallucinations' list, 'num_claims', 'num_hallucinated',
        'hallucination_rate', 'is_clean'.
    """
    claims = _extract_numerical_claims(rationale)

    hallucinations = []
    verified = 0
    total_claims = len(claims)

    for claim in claims:
        check = _verify_claim(claim, row, tolerance)
        if check["status"] == "hallucinated":
            hallucinations.append(check)
        elif check["status"] == "verified":
            verified += 1

    result = {
        "num_claims": total_claims,
        "num_verified": verified,
        "num_hallucinated": len(hallucinations),
        "hallucination_rate": len(hallucinations) / max(total_claims, 1),
        "hallucinations": hallucinations,
        "is_clean": len(hallucinations) == 0,
    }

    return result


def _extract_numerical_claims(rationale: str) -> List[Dict[str, Any]]:
    """Extract numerical claims from rationale text.

    Patterns detected:
        - Dollar amounts: $50,000 / $50000
        - Percentages: 15.5% / 15%
        - FICO scores: FICO of 720 / score of 720
        - Plain numbers near financial terms

    Args:
        rationale: LLM rationale text.

    Returns:
        List of claim dicts with 'value', 'context', 'type'.
    """
    claims = []

    # Dollar amounts
    for match in re.finditer(r"\$[\d,]+(?:\.\d+)?", rationale):
        value_str = match.group().replace("$", "").replace(",", "")
        try:
            value = float(value_str)
            start = max(0, match.start() - 30)
            end = min(len(rationale), match.end() + 30)
            context = rationale[start:end]
            claims.append({
                "value": value,
                "context": context,
                "type": "dollar",
                "raw": match.group(),
            })
        except ValueError:
            pass

    # Percentages
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", rationale):
        try:
            value = float(match.group(1))
            start = max(0, match.start() - 40)
            end = min(len(rationale), match.end() + 20)
            context = rationale[start:end]
            claims.append({
                "value": value,
                "context": context,
                "type": "percentage",
                "raw": match.group(),
            })
        except ValueError:
            pass

    # FICO-related numbers
    fico_pattern = r"(?:fico|credit\s+score|score)\s+(?:of\s+)?(\d{3})"
    for match in re.finditer(fico_pattern, rationale, re.IGNORECASE):
        try:
            value = float(match.group(1))
            if 300 <= value <= 850:  # Valid FICO range
                start = max(0, match.start() - 20)
                end = min(len(rationale), match.end() + 20)
                context = rationale[start:end]
                claims.append({
                    "value": value,
                    "context": context,
                    "type": "fico",
                    "raw": match.group(),
                })
        except ValueError:
            pass

    return claims


def _verify_claim(
    claim: Dict[str, Any],
    row: pd.Series,
    tolerance: float,
) -> Dict[str, Any]:
    """Verify a single numerical claim against the data.

    Args:
        claim: Claim dict from _extract_numerical_claims().
        row: Original data row.
        tolerance: Relative tolerance for matching.

    Returns:
        Dict with 'status' ('verified', 'hallucinated', 'unverifiable'),
        'claim', 'actual_value', 'message'.
    """
    claim_type = claim["type"]
    claimed_value = claim["value"]
    context = claim["context"].lower()

    # Map claim type to potential source columns
    check_cols = []

    if claim_type == "dollar":
        if "loan" in context or "amount" in context:
            check_cols = ["loan_amnt", "funded_amnt"]
        elif "income" in context or "annual" in context:
            check_cols = ["annual_inc"]
        elif "payment" in context or "installment" in context:
            check_cols = ["installment", "last_pymnt_amnt", "total_pymnt"]
        elif "balance" in context:
            check_cols = ["revol_bal", "tot_cur_bal"]
        else:
            check_cols = ["loan_amnt", "annual_inc", "installment",
                          "revol_bal", "total_pymnt"]

    elif claim_type == "percentage":
        if "dti" in context or "debt" in context:
            check_cols = ["dti"]
        elif "interest" in context or "rate" in context:
            check_cols = ["int_rate"]
        elif "revolving" in context or "util" in context:
            check_cols = ["revol_util", "bc_util"]
        else:
            check_cols = ["dti", "int_rate", "revol_util"]

    elif claim_type == "fico":
        check_cols = ["fico_mid", "fico_range_low", "fico_range_high"]

    # Check against each potential column
    for col in check_cols:
        actual = row.get(col)
        if actual is not None and not pd.isna(actual):
            actual = float(actual)
            if actual == 0:
                if claimed_value == 0:
                    return {
                        "status": "verified",
                        "claim": claim,
                        "matched_col": col,
                        "actual_value": actual,
                    }
                continue

            rel_error = abs(claimed_value - actual) / abs(actual)
            if rel_error <= tolerance:
                return {
                    "status": "verified",
                    "claim": claim,
                    "matched_col": col,
                    "actual_value": actual,
                }

    # If we checked columns but couldn't match
    if check_cols:
        actuals = {col: row.get(col) for col in check_cols
                   if col in row.index and not pd.isna(row.get(col))}
        return {
            "status": "hallucinated",
            "claim": claim,
            "actual_values": actuals,
            "message": f"Claimed {claim['raw']} but actual values are {actuals}",
        }

    return {
        "status": "unverifiable",
        "claim": claim,
        "message": "Could not determine source column for claim",
    }


def batch_hallucination_detection(
    rationales: List[str],
    df: pd.DataFrame,
    tolerance: float = 0.10,
) -> pd.DataFrame:
    """Run hallucination detection across multiple instances.

    Args:
        rationales: List of LLM rationale strings.
        df: DataFrame with original data (aligned with rationales).
        tolerance: Relative tolerance for number matching.

    Returns:
        DataFrame with per-instance hallucination results.
    """
    results = []

    for i, rationale in enumerate(rationales):
        if i >= len(df):
            break

        row = df.iloc[i]
        detection = detect_hallucinations(rationale, row, tolerance)

        results.append({
            "instance_idx": i,
            "num_claims": detection["num_claims"],
            "num_verified": detection["num_verified"],
            "num_hallucinated": detection["num_hallucinated"],
            "hallucination_rate": detection["hallucination_rate"],
            "is_clean": detection["is_clean"],
        })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        n_clean = results_df["is_clean"].sum()
        n_total = len(results_df)
        avg_rate = results_df["hallucination_rate"].mean()

        print(f"\n  Batch Hallucination Detection:")
        print(f"    Instances: {n_total}")
        print(f"    Clean (no hallucinations): {n_clean}/{n_total} "
              f"({n_clean/n_total*100:.1f}%)")
        print(f"    Mean hallucination rate: {avg_rate*100:.1f}%")

    return results_df
