"""
End-to-end RAG Chain orchestrator.

Coordinates the full Retrieval-Augmented Generation pipeline:
  1. Hybrid Retrieval (Dense + Sparse + RRF + Rerank)
  2. Grounded LLM Generation with citations
  3. Result formatting with source provenance
"""

import argparse
import time
from typing import Any

from src.config import get_settings
from src.generation.generator import GroundedGenerator
from src.logging_config import get_logger
from src.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger(__name__)


class RAGChain:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Orchestrates the Hybrid Retriever and Grounded Generator
    into a single query interface.
    """

    def __init__(self):
        settings = get_settings()
        self.retriever = HybridRetriever(final_top_k=settings.retrieval.final_top_k)
        self.generator = GroundedGenerator(temperature=settings.llm.temperature)

    def query(self, question: str) -> dict[str, Any]:
        """
        Execute the full RAG pipeline for a given question.

        Returns a dict containing the answer, sources, and pipeline metrics.
        """
        start = time.time()
        logger.info("RAG query: '%s'", question[:100])

        # Stage 1: Hybrid Retrieval
        retrieval_result = self.retriever.retrieve(question)
        retrieved_docs = retrieval_result["documents"]
        retrieval_metrics = retrieval_result["metrics"]

        # Stage 2: Grounded Generation
        logger.info("Generating response with Groq LLM")
        answer = self.generator.generate(question, retrieved_docs)

        total_latency_ms = (time.time() - start) * 1000

        # Stage 3: Format results
        result = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "filename": doc.metadata.get("filename"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "text_preview": doc.page_content[:200] + "...",
                    "rerank_score": doc.metadata.get("rerank_score"),
                }
                for doc in retrieved_docs
            ],
            "metrics": {
                **retrieval_metrics,
                "total_latency_ms": round(total_latency_ms, 1),
            },
        }

        logger.info(
            "RAG pipeline complete in %.0fms (%d sources)",
            total_latency_ms,
            len(retrieved_docs),
        )
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the End-to-End RAG Chain")
    parser.add_argument("--query", type=str, required=True, help="Question to ask the system")
    args = parser.parse_args()

    chain = RAGChain()
    result = chain.query(args.query)

    print("\n" + "=" * 50)
    print(" FINAL ANSWER")
    print("=" * 50)
    print(result["answer"])
    print("\n" + "=" * 50)
    print(" SOURCES USED")
    print("=" * 50)
    for source in result["sources"]:
        score = source.get("rerank_score", 0)
        print(f"  • {source['filename']} (Chunk {source['chunk_index']}, Score: {score:.3f})")
    print("\n" + "=" * 50)
    print(" PIPELINE METRICS")
    print("=" * 50)
    for key, val in result["metrics"].items():
        print(f"  {key}: {val}")
