import os
import json

def create_notebook(filename, title, markdown_intro, code_cells):
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", f"\n{markdown_intro}"]},
    ]
    
    # Optional auto-load autoreload block for development
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["%load_ext autoreload\n", "%autoreload 2\n", "import sys\n", "sys.path.append('..')"]
    })
    
    for code in code_cells:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })
        
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "credit_risk_llm_venv",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filename, "w") as f:
        json.dump(nb, f, indent=2)


NOTEBOOKS = {
    "notebooks/01_eda.ipynb": {
        "title": "01: Exploratory Data Analysis (EDA)",
        "intro": "Analyze the raw LendingClub dataset to understand distributions and missing values.",
        "cells": [
            "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom src.preprocess import load_config, load_lending_club\n\nconfig = load_config()\ndf = load_lending_club(config, nrows=50000)\ndf.head()",
            "sns.countplot(y='loan_status', data=df)\nplt.title('Loan Status Distribution')\nplt.show()"
        ]
    },
    "notebooks/02_preprocessing.ipynb": {
        "title": "02: End-to-End Preprocessing Pipeline",
        "intro": "Run the data through our `src.preprocess` module to generate train/val/test splits.",
        "cells": [
            "from src.preprocess import PreprocessingPipeline\n\npipeline = PreprocessingPipeline()\nresults = pipeline.run(nrows=100000, apply_smote=True, save=True)",
            "print(f\"Train size: {results['X_train'].shape}\")\nprint(f\"Test size: {results['X_test'].shape}\")"
        ]
    },
    "notebooks/03_baselines.ipynb": {
        "title": "03: Classical ML Baselines",
        "intro": "Train Logistic Regression, XGBoost, and LightGBM models.",
        "cells": [
            "import pandas as pd\nfrom src.preprocess import load_config, load_processed_data\nfrom src.models import train_lightgbm, train_xgboost\nfrom src.evaluate import evaluate_model, compare_models, plot_roc_curves\n\nconfig = load_config()\nX_train, y_train, X_val, y_val, X_test, y_test = load_processed_data(config)",
            "# Example: Train LightGBM\nmodel_lgb = train_lightgbm(X_train, y_train, X_val, y_val, config)\npreds_lgb = model_lgb.predict_proba(X_test)[:, 1]\n\nres_lgb = evaluate_model('LightGBM', y_test, preds_lgb, config)",
            "plot_roc_curves({'LightGBM': (y_test, preds_lgb)})"
        ]
    },
    "notebooks/04_text_pipeline.ipynb": {
        "title": "04: Text Pipeline & Synthesis",
        "intro": "Synthesize textual rationales from tabular features for LLM fine-tuning.",
        "cells": [
            "from src.text_utils import synthesize_text_column\nfrom src.preprocess import load_processed_data, load_config\n\nconfig = load_config()\nX_train, y_train, _, _, _, _ = load_processed_data(config)\n\nX_train_text = synthesize_text_column(X_train.head(100), config)\nX_train_text[['desc', 'synthesized_rationale']].head()"
        ]
    },
    "notebooks/05_finbert.ipynb": {
        "title": "05: FinBERT Embeddings",
        "intro": "Extract 768-dimensional embeddings from textual loan descriptions.",
        "cells": [
            "from src.text_utils import FinBERTExtractor\nfrom src.preprocess import load_config, load_processed_data\n\nconfig = load_config()\nextractor = FinBERTExtractor(config)\n\n# Example texts\ntexts = ['The borrower has a stable income but high DTI.', 'Excellent credit history.']\nembeddings = extractor.extract_embeddings(texts, batch_size=2)\nprint(embeddings.shape)"
        ]
    },
    "notebooks/06_qlora_finetune.ipynb": {
        "title": "06: LLaMA-3 QLoRA Fine-Tuning",
        "intro": "Fine-tune LLaMA-3 to generate credit risk rationales (run this on Kaggle T4).",
        "cells": [
            "from src.llm_utils import setup_unsloth_model, format_dataset_for_llm, train_qlora\nfrom src.preprocess import load_config\n\nconfig = load_config()\nprint('Ready to initialize Unsloth for LLaMA-3 8B Instruct!')"
        ]
    },
    "notebooks/07_fusion.ipynb": {
        "title": "07: Late Fusion Model",
        "intro": "Combine tabular predictions (LightGBM) with textual embeddings/rationales.",
        "cells": [
            "from sklearn.linear_model import LogisticRegression\nimport numpy as np\n\n# Placeholder for meta-learner fusion\nprint('Train meta-learner on concatenated [pred_tabular, pred_text]')\n"
        ]
    },
    "notebooks/08_xai.ipynb": {
        "title": "08: Explainability (SHAP & DiCE)",
        "intro": "Extract global/local insights using TreeExplainer and generate counterfactuals.",
        "cells": [
            "from src.explain import compute_shap_values, plot_shap_summary, generate_counterfactuals\nfrom src.preprocess import load_processed_data, load_config\nimport lightgbm as lgb\nimport joblib\n\nconfig = load_config()\nX_train, y_train, _, _, X_test, y_test = load_processed_data(config)\n\n# Assuming model exists\n# model = joblib.load('../models/lightgbm_tuned.pkl')\n# shap_values = compute_shap_values(model, X_test.head(1000))\n# plot_shap_summary(shap_values)"
        ]
    },
    "notebooks/09_fairness.ipynb": {
        "title": "09: Fairness & Bias Audits",
        "intro": "Audit the model using Aequitas and detect LLM hallucinations.",
        "cells": [
            "from src.explain import batch_hallucination_detection\nimport pandas as pd\n\nprint('Run Aequitas Group/Fairness metrics here')\n# detect_hallucinations(rationale, row)"
        ]
    }
}

if __name__ == "__main__":
    for filepath, nb_data in NOTEBOOKS.items():
        create_notebook(
            os.path.join("d:/credit_risk_llm", filepath),
            nb_data["title"],
            nb_data["intro"],
            nb_data["cells"]
        )
    print("Successfully populated all 9 notebooks with starter code!")
