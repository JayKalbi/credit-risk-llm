"""RAG evaluation: golden dataset generation and LLM-as-judge scoring."""

from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.golden_dataset import GoldenDatasetGenerator

__all__ = ["RAGEvaluator", "GoldenDatasetGenerator"]
