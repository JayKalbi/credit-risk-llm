"""
HybridCreditLLM — Text Utilities Module
=========================================
Text cleaning, synthesis, and FinBERT embedding extraction.

Handles the critical dual-text strategy:
    1. Real `desc` text (available for ~5.6% of rows, mostly 2007-2013)
    2. Synthesized text from emp_title + purpose + title (100% coverage)

Usage:
    from src.text_utils import (
        clean_text, synthesize_text_column,
        extract_finbert_embeddings, TextPipeline
    )
"""

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Text Cleaning
# ============================================================================

def clean_text(text: str) -> str:
    """Clean a single text string for NLP processing.

    Operations:
        1. Strip HTML tags
        2. Remove URLs
        3. Remove email addresses
        4. Normalize whitespace
        5. Strip leading/trailing whitespace
        6. Convert to lowercase

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip and lowercase
    text = text.strip().lower()

    return text


def clean_text_series(series: pd.Series, show_progress: bool = True) -> pd.Series:
    """Clean a pandas Series of text strings.

    Args:
        series: Series of raw text.
        show_progress: Whether to show a progress bar.

    Returns:
        Series of cleaned text.
    """
    if show_progress:
        tqdm.pandas(desc="Cleaning text")
        result = series.fillna("").progress_apply(clean_text)
    else:
        result = series.fillna("").apply(clean_text)

    n_empty = (result.str.len() == 0).sum()
    n_total = len(result)
    print(f"  Cleaned {n_total:,} texts | {n_empty:,} empty ({n_empty/n_total*100:.1f}%)")
    print(f"  Avg length: {result.str.len().mean():.0f} chars")

    return result


# ============================================================================
# Text Synthesis (Fallback Strategy)
# ============================================================================

def synthesize_text_column(df: pd.DataFrame) -> pd.Series:
    """Synthesize a text column from structured fields.

    Creates natural language descriptions from:
        - emp_title (employment title)
        - purpose (loan purpose)
        - title (borrower's title for the loan)

    Template: "Borrower employed as {emp_title} seeking a loan for {purpose}. {title}"

    Args:
        df: DataFrame with emp_title, purpose, title columns.

    Returns:
        Series of synthesized text strings.
    """
    emp_title = df.get("emp_title", pd.Series(["unknown"] * len(df))).fillna("unknown")
    purpose = df.get("purpose", pd.Series(["unspecified"] * len(df))).fillna("unspecified")
    title = df.get("title", pd.Series([""] * len(df))).fillna("")

    # Clean up purpose field (replace underscores)
    purpose_clean = purpose.str.replace("_", " ", regex=False).str.strip()

    # Build synthesized text
    text = (
        "Borrower employed as " + emp_title.str.strip()
        + " seeking a loan for " + purpose_clean
        + ". " + title.str.strip()
    )

    # Normalize
    text = text.str.replace(r"\s+", " ", regex=True)
    text = text.str.replace(r"\.\s*\.", ".", regex=True)
    text = text.str.strip()

    print(f"  Synthesized text for {len(text):,} rows")
    print(f"  Avg length: {text.str.len().mean():.0f} chars")

    return text


def build_text_column(df: pd.DataFrame, use_desc: bool = False) -> pd.Series:
    """Build the final text column using the dual strategy.

    Strategy:
        - If use_desc=True and desc is available: prefer real desc,
          fallback to synthesized text where desc is empty.
        - If use_desc=False: always use synthesized text.

    Args:
        df: LendingClub DataFrame.
        use_desc: Whether to attempt using the desc column.

    Returns:
        Series of text strings for the FinBERT pipeline.
    """
    # Always generate synthesized text as baseline
    synth_text = synthesize_text_column(df)

    if use_desc and "desc" in df.columns:
        desc_clean = clean_text_series(df["desc"], show_progress=False)
        has_desc = desc_clean.str.len() > 0

        # Prefer real desc where available, fallback to synthesized
        text = synth_text.copy()
        text[has_desc] = desc_clean[has_desc]

        n_real = has_desc.sum()
        print(f"  Dual strategy: {n_real:,} real desc + "
              f"{len(text) - n_real:,} synthesized")
    else:
        text = synth_text
        print(f"  Using synthesized text only (desc not available/used)")

    # Final clean pass
    text = text.apply(clean_text)

    return text


# ============================================================================
# FinBERT Embedding Extraction
# ============================================================================

def extract_finbert_embeddings(
    texts: List[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 64,
    max_length: int = 128,
    device: str = "cuda",
    show_progress: bool = True,
) -> np.ndarray:
    """Extract CLS embeddings from FinBERT for a list of texts.

    Uses the [CLS] token representation from the last hidden state
    as a 768-dimensional embedding for each text.

    Args:
        texts: List of text strings.
        model_name: HuggingFace model identifier.
        batch_size: Texts per batch (reduce if OOM).
        max_length: Max token length (128 is fine for short descriptions).
        device: 'cuda' or 'cpu'.
        show_progress: Whether to show progress bar.

    Returns:
        numpy array of shape (n_texts, 768).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading FinBERT ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting embeddings")

    with torch.no_grad():
        for i in iterator:
            batch_texts = texts[i : i + batch_size]

            # Replace empty strings with a placeholder
            batch_texts = [t if len(t.strip()) > 0 else "no description" for t in batch_texts]

            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**tokens)

            # CLS token is the first token [0] of the last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    embeddings = np.vstack(all_embeddings)
    print(f"  Extracted {embeddings.shape[0]:,} embeddings of dim {embeddings.shape[1]}")

    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    name: str,
    save_dir: str = "data/embeddings",
) -> str:
    """Save embeddings as a numpy file.

    Args:
        embeddings: numpy array of embeddings.
        name: Filename (without extension).
        save_dir: Directory to save to.

    Returns:
        Path to saved file.
    """
    import os

    save_dir = os.path.join(PROJECT_ROOT, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{name}.npy")
    np.save(filepath, embeddings)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved embeddings: {filepath} ({size_mb:.1f} MB)")
    return filepath


def load_embeddings(name: str, save_dir: str = "data/embeddings") -> np.ndarray:
    """Load embeddings from a numpy file.

    Args:
        name: Filename (without extension).
        save_dir: Directory to load from.

    Returns:
        numpy array of embeddings.
    """
    import os

    filepath = os.path.join(PROJECT_ROOT, save_dir, f"{name}.npy")
    embeddings = np.load(filepath)
    print(f"  Loaded embeddings: {filepath} — shape {embeddings.shape}")
    return embeddings


# ============================================================================
# Text Statistics
# ============================================================================

def text_statistics(series: pd.Series, name: str = "text") -> dict:
    """Compute statistics for a text column.

    Args:
        series: Series of text strings.
        name: Label for the text column.

    Returns:
        Dictionary of text statistics.
    """
    lengths = series.str.len()
    word_counts = series.str.split().str.len()

    stats = {
        "name": name,
        "total": len(series),
        "non_empty": (lengths > 0).sum(),
        "empty": (lengths == 0).sum(),
        "coverage_pct": (lengths > 0).sum() / len(series) * 100,
        "avg_char_length": lengths.mean(),
        "median_char_length": lengths.median(),
        "max_char_length": lengths.max(),
        "avg_word_count": word_counts.mean(),
        "median_word_count": word_counts.median(),
    }

    print(f"\n--- Text Statistics: {name} ---")
    print(f"  Coverage: {stats['non_empty']:,}/{stats['total']:,} "
          f"({stats['coverage_pct']:.1f}%)")
    print(f"  Avg chars: {stats['avg_char_length']:.0f} | "
          f"Median: {stats['median_char_length']:.0f} | "
          f"Max: {stats['max_char_length']:.0f}")
    print(f"  Avg words: {stats['avg_word_count']:.0f} | "
          f"Median: {stats['median_word_count']:.0f}")

    return stats
