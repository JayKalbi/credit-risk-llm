"""
HybridCreditLLM — Source Package
================================
Modular utilities for the hybrid ML + LLM credit risk pipeline.

Modules:
    preprocess  — Data loading, cleaning, temporal splitting, feature engineering
    models      — Model training (LightGBM, XGBoost, Logistic Regression)
    evaluate    — Metrics computation (AUC-ROC, PR-AUC, KS, Brier, F1) and visualization
    explain     — SHAP, DiCE counterfactuals, faithfulness audit, hallucination detection
    text_utils  — Text cleaning, synthesis, and FinBERT embedding extraction
    llm_utils   — LLaMA loading, prompt formatting, rationale generation
"""

__version__ = "0.1.0"
__author__ = "JayKalbi"

