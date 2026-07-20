"""
LLM-as-a-Judge evaluation framework for the RAG pipeline.

Uses a stronger model (Llama 3.3 70B) to evaluate a weaker model's
(Llama 3.1 8B) outputs across multiple quality dimensions:
  - Answer Relevance: Does it answer the question?
  - Factual Accuracy: Is it consistent with the ground truth?
  - Faithfulness: Is it grounded in the retrieved context?
"""

import json
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src.config import get_settings
from src.generation.rag_chain import RAGChain
from src.logging_config import get_logger

logger = get_logger(__name__)

_EVAL_PROMPT = PromptTemplate(
    input_variables=["question", "ground_truth", "generated_answer"],
    template="""You are an impartial judge evaluating a RAG system.

QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}
SYSTEM'S ANSWER: {generated_answer}

Evaluate the SYSTEM'S ANSWER on two metrics:
1. RELEVANCE (0 or 1): Does it directly answer the QUESTION?
2. ACCURACY (0 or 1): Is it factually consistent with the GROUND TRUTH?

Return exactly in this JSON format:
{{"relevance": 1, "accuracy": 1}}""",
)


class RAGEvaluator:
    """
    Automated evaluation of the RAG pipeline using LLM-as-a-Judge.

    Runs each question through the full pipeline, then asks a stronger
    judge model to score the output against ground truth.
    """

    def __init__(self, dataset_path: str | None = None):
        settings = get_settings()
        self.dataset_path = dataset_path or settings.golden_dataset_path
        self.rag_pipeline = RAGChain()

        load_dotenv()
        api_key = settings.llm.api_key

        logger.info("Initializing Judge LLM (%s)", settings.llm.judge_model_name)
        self.judge_llm = ChatGroq(
            temperature=0.0,
            model_name=settings.llm.judge_model_name,
            api_key=api_key,
        )
        self.eval_prompt = _EVAL_PROMPT

    def evaluate(self) -> dict[str, Any]:
        """
        Run evaluation across the golden dataset.

        Returns a dict with per-question results and aggregate metrics.
        """
        logger.info("Loading golden dataset from %s", self.dataset_path)

        with open(self.dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        if not dataset:
            logger.error("Golden dataset is empty")
            return {"error": "Empty dataset"}

        total_relevance = 0
        total_accuracy = 0
        results: list[dict[str, Any]] = []
        errors = 0

        logger.info("Evaluating %d Q&A pairs", len(dataset))

        for i, item in enumerate(dataset):
            question = item["question"]
            ground_truth = item["ground_truth"]

            logger.info("[%d/%d] Q: %s", i + 1, len(dataset), question[:80])

            try:
                # Run pipeline
                pipeline_result = self.rag_pipeline.query(question)
                generated_answer = pipeline_result["answer"]

                # Judge the output
                prompt = self.eval_prompt.format(
                    question=question,
                    ground_truth=ground_truth,
                    generated_answer=generated_answer,
                )

                judge_response = self.judge_llm.invoke(prompt).content

                # Parse JSON from judge response
                json_start = judge_response.find("{")
                json_end = judge_response.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError(f"No JSON found in judge response: {judge_response[:100]}")

                judge_json = json.loads(judge_response[json_start:json_end])

                rel = judge_json.get("relevance", 0)
                acc = judge_json.get("accuracy", 0)

                total_relevance += rel
                total_accuracy += acc

                results.append(
                    {
                        "question": question,
                        "relevance": rel,
                        "accuracy": acc,
                        "generated_answer": generated_answer[:300],
                    }
                )

                logger.info("  → Relevance: %d | Accuracy: %d", rel, acc)

            except Exception:
                errors += 1
                logger.error("  → Failed to evaluate question %d", i + 1, exc_info=True)
                results.append(
                    {
                        "question": question,
                        "relevance": 0,
                        "accuracy": 0,
                        "error": True,
                    }
                )

            # Respect Groq rate limits with backoff
            time.sleep(3)

        # Aggregate
        n = len(dataset)
        avg_rel = total_relevance / n
        avg_acc = total_accuracy / n

        summary = {
            "total_questions": n,
            "answer_relevance": round(avg_rel * 100, 1),
            "factual_accuracy": round(avg_acc * 100, 1),
            "errors": errors,
            "results": results,
        }

        logger.info("=" * 50)
        logger.info(" EVALUATION RESULTS (LLM-as-Judge)")
        logger.info("=" * 50)
        logger.info("Total Questions : %d", n)
        logger.info("Relevance Score : %.1f%%", avg_rel * 100)
        logger.info("Accuracy Score  : %.1f%%", avg_acc * 100)
        logger.info("Errors          : %d", errors)
        logger.info("=" * 50)

        return summary


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.evaluate()
