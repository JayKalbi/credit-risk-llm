"""
HybridCreditLLM — LLM Utilities Module
========================================
Handles LLaMA-3-8B loading via Unsloth, QLoRA configuration,
instruction prompt formatting, and rationale generation.

Designed for Kaggle T4 (16GB VRAM) with strict memory management:
    - 4-bit NF4 quantization
    - Gradient checkpointing
    - Paged AdamW 8-bit optimizer
    - batch_size=1 with gradient_accumulation=8

Usage:
    from src.llm_utils import (
        load_model_unsloth, format_credit_prompt,
        generate_rationale, prepare_training_data
    )
"""

import json
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
# Prompt Templates
# ============================================================================

CREDIT_RISK_SYSTEM_PROMPT = """You are a credit risk analyst. Given a loan application with financial data and a borrower description, assess the default risk. Provide:
1. A risk classification (HIGH or LOW)
2. A detailed rationale explaining which factors drive the risk assessment
3. Reference specific numbers from the financial profile in your explanation"""

CREDIT_RISK_INSTRUCTION_TEMPLATE = """### Loan Application

**Borrower Description:** {description}

**Financial Profile:**
- Loan Amount: ${loan_amnt:,.0f}
- Interest Rate: {int_rate:.1f}%
- Annual Income: ${annual_inc:,.0f}
- DTI Ratio: {dti:.1f}%
- FICO Score: {fico_mid:.0f}
- Revolving Utilization: {revol_util:.1f}%
- Employment Length: {emp_length}
- Home Ownership: {home_ownership}
- Loan Purpose: {purpose}
- Loan Grade: {grade}

Assess the default risk and provide your rationale."""

CREDIT_RISK_RESPONSE_TEMPLATE = """**Risk Assessment: {risk_label}**

**Rationale:** {rationale}"""


def format_credit_prompt(
    row: pd.Series,
    text_col: str = "text",
    include_response: bool = False,
    risk_label: Optional[str] = None,
    rationale: Optional[str] = None,
) -> str:
    """Format a single loan application into an instruction prompt.

    Args:
        row: DataFrame row with loan features.
        text_col: Name of the text column.
        include_response: Whether to include the response (for training).
        risk_label: 'HIGH' or 'LOW' (for training data).
        rationale: Chain-of-thought rationale (for training data).

    Returns:
        Formatted prompt string.
    """
    # Safely extract values with defaults
    def safe_get(col, default=0):
        val = row.get(col, default)
        if pd.isna(val):
            return default
        return val

    description = safe_get(text_col, "No description provided")
    if not isinstance(description, str) or len(description.strip()) == 0:
        description = "No description provided"

    instruction = CREDIT_RISK_INSTRUCTION_TEMPLATE.format(
        description=description,
        loan_amnt=safe_get("loan_amnt", 0),
        int_rate=safe_get("int_rate", 0),
        annual_inc=safe_get("annual_inc", 0),
        dti=safe_get("dti", 0),
        revol_util=safe_get("revol_util", 0),
        fico_mid=safe_get("fico_mid", 0),
        emp_length=safe_get("emp_length", "Unknown"),
        home_ownership=safe_get("home_ownership", "Unknown"),
        purpose=safe_get("purpose", "Unknown"),
        grade=safe_get("grade", "Unknown"),
    )

    # Build the full prompt
    prompt = f"[INST] {CREDIT_RISK_SYSTEM_PROMPT}\n\n{instruction} [/INST]\n"

    if include_response and risk_label:
        response = CREDIT_RISK_RESPONSE_TEMPLATE.format(
            risk_label=risk_label,
            rationale=rationale or "Based on the financial profile analysis.",
        )
        prompt += f"\n{response}"

    return prompt


def generate_training_rationale(row: pd.Series) -> str:
    """Generate a training rationale based on actual tabular features.

    Creates a ground-truth chain-of-thought explanation from the
    structured data. This serves as the target for fine-tuning.

    Args:
        row: DataFrame row with loan features and target.

    Returns:
        Chain-of-thought rationale string.
    """
    factors = []

    # FICO analysis
    fico = row.get("fico_mid", 0)
    if pd.notna(fico) and fico > 0:
        if fico >= 740:
            factors.append(f"The FICO score of {fico:.0f} is excellent, "
                          "indicating strong creditworthiness")
        elif fico >= 670:
            factors.append(f"The FICO score of {fico:.0f} is good, "
                          "suggesting acceptable credit history")
        else:
            factors.append(f"The FICO score of {fico:.0f} is below average, "
                          "indicating elevated credit risk")

    # DTI analysis
    dti = row.get("dti", 0)
    if pd.notna(dti) and dti > 0:
        if dti > 35:
            factors.append(f"The DTI ratio of {dti:.1f}% is high, "
                          "suggesting the borrower may be overleveraged")
        elif dti < 20:
            factors.append(f"The DTI ratio of {dti:.1f}% is healthy, "
                          "indicating manageable debt levels")
        else:
            factors.append(f"The DTI ratio of {dti:.1f}% is moderate")

    # Interest rate analysis
    int_rate = row.get("int_rate", 0)
    if pd.notna(int_rate) and int_rate > 0:
        if int_rate > 15:
            factors.append(f"The interest rate of {int_rate:.1f}% is high, "
                          "reflecting the lender's assessment of elevated risk")
        elif int_rate < 8:
            factors.append(f"The low interest rate of {int_rate:.1f}% "
                          "suggests the lender sees low risk")

    # Income vs loan amount
    annual_inc = row.get("annual_inc", 0)
    loan_amnt = row.get("loan_amnt", 0)
    if pd.notna(annual_inc) and annual_inc > 0 and pd.notna(loan_amnt):
        ratio = loan_amnt / annual_inc
        if ratio > 0.5:
            factors.append(f"The loan-to-income ratio of {ratio:.2f} is high, "
                          "as the ${loan_amnt:,.0f} loan is significant "
                          f"relative to ${annual_inc:,.0f} annual income")

    # Revolving utilization
    revol_util = row.get("revol_util", 0)
    if pd.notna(revol_util) and revol_util > 0:
        if revol_util > 80:
            factors.append(f"Revolving credit utilization of {revol_util:.1f}% "
                          "is very high, indicating heavy reliance on credit")
        elif revol_util < 30:
            factors.append(f"Revolving credit utilization of {revol_util:.1f}% "
                          "is low, suggesting responsible credit use")

    # Grade
    grade = row.get("grade", "")
    if pd.notna(grade) and isinstance(grade, str):
        if grade in ["A", "B"]:
            factors.append(f"The loan grade '{grade}' reflects low risk assessment")
        elif grade in ["F", "G"]:
            factors.append(f"The loan grade '{grade}' reflects very high risk")

    if not factors:
        factors.append("Based on the overall financial profile analysis")

    return ". ".join(factors) + "."


# ============================================================================
# Training Data Preparation
# ============================================================================

def prepare_training_data(
    df: pd.DataFrame,
    text_col: str = "text",
    target_col: str = "target",
    max_samples: Optional[int] = None,
    balance: bool = True,
) -> List[Dict[str, str]]:
    """Prepare instruction-tuning data for QLoRA fine-tuning.

    Creates input-output pairs where:
        - Input: Formatted credit risk assessment prompt
        - Output: Risk label + chain-of-thought rationale

    Args:
        df: DataFrame with features, text, and target columns.
        text_col: Name of the text column.
        target_col: Name of the target column.
        max_samples: Maximum total samples (None = use all).
        balance: Whether to balance classes.

    Returns:
        List of dicts with 'instruction' and 'output' keys.
    """
    if balance:
        # Balance by undersampling the majority class
        n_pos = (df[target_col] == 1).sum()
        n_neg = (df[target_col] == 0).sum()
        n_per_class = min(n_pos, n_neg)

        if max_samples:
            n_per_class = min(n_per_class, max_samples // 2)

        pos_df = df[df[target_col] == 1].sample(n=n_per_class, random_state=42)
        neg_df = df[df[target_col] == 0].sample(n=n_per_class, random_state=42)
        df_balanced = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42)
    else:
        df_balanced = df
        if max_samples:
            df_balanced = df_balanced.sample(n=min(max_samples, len(df)),
                                             random_state=42)

    training_data = []
    for _, row in df_balanced.iterrows():
        risk_label = "HIGH" if row[target_col] == 1 else "LOW"
        rationale = generate_training_rationale(row)

        prompt = format_credit_prompt(
            row,
            text_col=text_col,
            include_response=True,
            risk_label=risk_label,
            rationale=rationale,
        )

        training_data.append({
            "text": prompt,
            "risk_label": risk_label,
        })

    print(f"Prepared {len(training_data):,} training samples")
    print(f"  HIGH risk: {sum(1 for d in training_data if d['risk_label'] == 'HIGH'):,}")
    print(f"  LOW risk:  {sum(1 for d in training_data if d['risk_label'] == 'LOW'):,}")

    return training_data


def save_training_data(
    data: List[Dict[str, str]],
    filepath: str,
) -> None:
    """Save training data as JSONL for Unsloth/HuggingFace.

    Args:
        data: List of training samples.
        filepath: Output JSONL file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Saved {len(data):,} samples to {filepath}")


# ============================================================================
# Model Loading (Unsloth)
# ============================================================================

def load_model_unsloth(config: dict, use_fallback: bool = False) -> Tuple[Any, Any]:
    """Load a quantized LLM via Unsloth for QLoRA fine-tuning.

    Designed for Kaggle T4 (16GB VRAM):
        - 4-bit NF4 quantization
        - Loads model + tokenizer

    Args:
        config: Project configuration dictionary.
        use_fallback: If True, use fallback model (Mistral-7B).

    Returns:
        Tuple of (model, tokenizer).
    """
    from unsloth import FastLanguageModel

    qlora_cfg = config["qlora"]
    model_name = qlora_cfg["fallback_model"] if use_fallback else qlora_cfg["base_model"]

    print(f"Loading model: {model_name}")
    print(f"  4-bit quantization: {qlora_cfg['load_in_4bit']}")
    print(f"  Max sequence length: {qlora_cfg['max_seq_length']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=qlora_cfg["max_seq_length"],
        dtype=None,  # Auto-detect
        load_in_4bit=qlora_cfg["load_in_4bit"],
    )

    return model, tokenizer


def apply_qlora_adapters(model: Any, config: dict) -> Any:
    """Apply QLoRA adapters to the loaded model.

    Args:
        model: Loaded base model from Unsloth.
        config: Project configuration dictionary.

    Returns:
        Model with QLoRA adapters applied.
    """
    from unsloth import FastLanguageModel

    lora_cfg = config["qlora"]["lora"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",  # Unsloth-optimized
        random_state=42,
    )

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} "
          f"({trainable / total * 100:.2f}%)")

    return model


# ============================================================================
# Inference
# ============================================================================

def generate_rationale(
    model: Any,
    tokenizer: Any,
    row: pd.Series,
    text_col: str = "text",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a risk rationale for a single loan application.

    Args:
        model: Fine-tuned model.
        tokenizer: Model tokenizer.
        row: DataFrame row with loan features.
        text_col: Name of the text column.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated rationale text.
    """
    import torch

    prompt = format_credit_prompt(row, text_col=text_col, include_response=False)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # Decode only the generated portion (skip the prompt)
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return generated.strip()


def batch_generate_rationales(
    model: Any,
    tokenizer: Any,
    df: pd.DataFrame,
    text_col: str = "text",
    max_new_tokens: int = 256,
    max_samples: Optional[int] = None,
) -> List[str]:
    """Generate rationales for multiple loan applications.

    Args:
        model: Fine-tuned model.
        tokenizer: Model tokenizer.
        df: DataFrame with loan features.
        text_col: Name of the text column.
        max_new_tokens: Maximum tokens per generation.
        max_samples: Limit number of samples.

    Returns:
        List of generated rationale strings.
    """
    from tqdm import tqdm

    if max_samples:
        df = df.head(max_samples)

    rationales = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating rationales"):
        rationale = generate_rationale(
            model, tokenizer, row,
            text_col=text_col,
            max_new_tokens=max_new_tokens,
        )
        rationales.append(rationale)

    print(f"  Generated {len(rationales)} rationales")
    return rationales
