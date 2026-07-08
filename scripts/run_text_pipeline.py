import os
import pandas as pd
from pathlib import Path

# Fix pythonpath issue if run directly
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.text_utils import build_text_column, text_statistics
from src.preprocess import load_config

def main():
    print("=" * 60)
    print("  HybridCreditLLM — Local Text Synthesis Pipeline")
    print("=" * 60)
    
    config = load_config()
    data_dir = os.path.join(PROJECT_ROOT, config["paths"]["processed_data"])
    
    # Check if text_dfs were saved by preprocess.py
    if not os.path.exists(os.path.join(data_dir, "text_train.parquet")):
        print("Error: text_train.parquet not found.")
        print("Please run preprocess.py first to extract the text columns.")
        return
        
    for split in ["train", "val", "test"]:
        filepath = os.path.join(data_dir, f"text_{split}.parquet")
        if not os.path.exists(filepath):
            continue
            
        print(f"\nProcessing {split} split...")
        df = pd.read_parquet(filepath)
        
        # Build the cleaned, synthesized text
        text_series = build_text_column(df, use_desc=True)
        text_series.name = "text"
        
        # Calculate and print statistics
        stats = text_statistics(text_series, name=f"{split} text")
        
        # Save as a single-column parquet file ready for Kaggle / FinBERT
        save_path = os.path.join(data_dir, f"text_{split}_clean.parquet")
        text_series.to_frame().to_parquet(save_path)
        print(f"  Saved clean text to {save_path}")

if __name__ == "__main__":
    main()
