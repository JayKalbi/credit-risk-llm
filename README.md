# 🏦 Credit Risk Assessment with LLM-Enhanced Features

> **Independent Research Project**
> Enhancing traditional credit scoring with Large Language Model embeddings and explainable AI.

---

## 📋 Project Overview

This project explores whether **LLM-derived text features** (from loan descriptions, employment titles, etc.) can improve credit default prediction beyond traditional tabular models. The system combines:

- **Traditional ML baselines** (XGBoost, LightGBM, Logistic Regression)
- **LLM text embeddings** (FinBERT / QLoRA fine-tuned models)
- **Fusion architectures** (tabular + text feature combination)
- **Explainable AI** (SHAP, LIME for model interpretability)
- **Fairness analysis** (bias detection across protected groups)

## 🗂️ Project Structure

```
credit_risk_llm/
├── data/
│   ├── raw/                  # Original datasets (not tracked)
│   ├── processed/            # Cleaned & engineered features
│   └── embeddings/           # LLM-generated embeddings
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_baselines.ipynb    # Traditional ML models
│   ├── 04_text_pipeline.ipynb
│   ├── 05_finbert.ipynb      # FinBERT embeddings
│   ├── 06_qlora_finetune.ipynb  # QLoRA fine-tuning (Kaggle GPU)
│   ├── 07_fusion.ipynb       # Tabular + text fusion
│   ├── 08_xai.ipynb          # SHAP / LIME explanations
│   └── 09_fairness.ipynb     # Bias & fairness audit
├── src/
│   ├── preprocess.py         # Data cleaning & feature engineering
│   ├── models.py             # Model definitions & training
│   ├── evaluate.py           # Metrics & evaluation utilities
│   └── explain.py            # XAI utilities
├── figures/                  # Generated plots & visualizations
├── models/                   # Saved model checkpoints (not tracked)
├── results/                  # Evaluation results (not tracked)
├── requirements.txt
└── README.md
```

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| **ML / DL** | scikit-learn, XGBoost, LightGBM, PyTorch |
| **NLP / LLM** | HuggingFace Transformers, FinBERT, PEFT (QLoRA) |
| **XAI** | SHAP, LIME |
| **Data** | pandas, numpy, polars |
| **Visualization** | matplotlib, seaborn, plotly |
| **GPU Runtime** | Kaggle Notebooks (T4/P100) |

## 📊 Datasets

| Dataset | Rows | Features | Text Field | Purpose |
|---------|------|----------|------------|---------|
| **LendingClub** | ~2.2M | ~140 | ✅ `desc` | Primary — LLM text features |
| **Home Credit** | ~307K | ~122 | ❌ | Tabular-only benchmark |
| **German Credit** | 1,000 | 20 | ❌ | Quick prototyping |

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/JayKalbi/credit-risk-llm.git
cd credit-risk-llm

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download datasets (requires Kaggle API key)
python 01_download.py
```

## 📄 License

This project is an independent research study on hybrid credit risk assessment.

---

*🚀 Project Status: **Phase 1 (Tabular Baselines) Complete** — Transitioning to Phase 2 (Text Pipeline & LLM Finetuning).*
