"""
Hybrid retriever combining dense, sparse, and reranking stages.

Implements a production-grade retrieval pipeline:
  1. Dense Vector Search  (ChromaDB + Sentence-Transformers)
  2. Sparse Keyword Search (BM25)
  3. Reciprocal Rank Fusion (RRF) to merge ranked lists
  4. Cross-Encoder Reranking for final refinement
"""

import time
from typing import Dict, List

from langchain_core.documents import Document

from src.config import get_settings
from src.logging_config import get_logger
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse_retriever import SparseRetriever

logger = get_logger(__name__)


class HybridRetriever:
    """
    Orchestrates multi-stage hybrid retrieval with RRF fusion.

    Pipeline: Dense + Sparse → RRF Merge → Cross-Encoder Rerank → Top-K
    """

    def __init__(
        self,
        rrf_k: int | None = None,
        final_top_k: int | None = None,
    ):
        settings = get_settings()
        self.rrf_k = rrf_k or settings.retrieval.rrf_k
        self.final_top_k = final_top_k or settings.retrieval.final_top_k
        self.rerank_candidates = settings.retrieval.rerank_candidates

        self.dense_retriever = DenseRetriever(
            top_k=settings.retrieval.dense_top_k
        )
        self.sparse_retriever = SparseRetriever(
            top_k=settings.retrieval.sparse_top_k
        )
        self.reranker = CrossEncoderReranker(top_k=self.final_top_k)

    def retrieve(self, query: str) -> Dict:
        """
        Execute the full hybrid retrieval pipeline.

        Returns a dict containing the final documents and pipeline metrics.
        """
        start = time.time()
        logger.info("Hybrid retrieval for: '%s'", query[:100])

        # Stage 1: Independent candidate retrieval
        dense_docs = self.dense_retriever.retrieve(query)
        sparse_docs = self.sparse_retriever.retrieve(query)

        logger.info(
            "Stage 1 — Dense: %d candidates, Sparse: %d candidates",
            len(dense_docs),
            len(sparse_docs),
        )

        # Stage 2: Reciprocal Rank Fusion
        fused_docs = self._reciprocal_rank_fusion(dense_docs, sparse_docs)
        logger.info(
            "Stage 2 — RRF fusion: %d unique candidates", len(fused_docs)
        )

        # Stage 3: Cross-Encoder reranking (on top N candidates)
        candidates = fused_docs[: self.rerank_candidates]
        final_docs = self.reranker.rerank(
            query, candidates, top_k=self.final_top_k
        )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Stage 3 — Reranked to %d final documents (%.0fms total)",
            len(final_docs),
            elapsed_ms,
        )

        return {
            "documents": final_docs,
            "metrics": {
                "dense_candidates": len(dense_docs),
                "sparse_candidates": len(sparse_docs),
                "fused_candidates": len(fused_docs),
                "final_count": len(final_docs),
                "retrieval_latency_ms": round(elapsed_ms, 1),
            },
        }

    def _reciprocal_rank_fusion(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
    ) -> List[Document]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = Σ 1 / (k + rank) across all lists where the
        document appears.  Constant k (default 60) controls how
        much weight is given to lower-ranked items.
        """
        doc_map: Dict[str, Document] = {}
        rrf_scores: Dict[str, float] = {}

        def _doc_id(doc: Document) -> str:
            return doc.metadata.get(
                "content_hash",
                f"{doc.metadata.get('filename', 'unknown')}"
                f"_{doc.metadata.get('chunk_index', 0)}",
            )

        # Accumulate RRF scores from dense rankings
        for rank, doc in enumerate(dense_docs):
            did = _doc_id(doc)
            doc_map[did] = doc
            rrf_scores[did] = 1.0 / (self.rrf_k + rank + 1)

        # Accumulate RRF scores from sparse rankings
        for rank, doc in enumerate(sparse_docs):
            did = _doc_id(doc)
            if did not in doc_map:
                doc_map[did] = doc
                rrf_scores[did] = 0.0
            rrf_scores[did] += 1.0 / (self.rrf_k + rank + 1)

        # Sort by combined RRF score (descending)
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        merged: List[Document] = []
        for did in sorted_ids:
            doc = doc_map[did]
            doc.metadata["rrf_score"] = rrf_scores[did]
            merged.append(doc)

        return merged
