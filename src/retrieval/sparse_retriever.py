"""
Sparse keyword retriever using BM25 (Okapi variant).

Performs exact keyword matching over a pre-built BM25 index to find
chunks that share the most query terms.
"""

import os
import pickle
from typing import List

from langchain_core.documents import Document

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class SparseRetriever:
    """
    BM25-based keyword retrieval.

    Loads a pre-built BM25 index from disk and scores documents
    against the tokenized query.

    WARNING: Uses pickle for deserialization. Ensure the index file
    originates from a trusted source (i.e., your own ingestion pipeline).
    """

    def __init__(
        self,
        bm25_path: str | None = None,
        top_k: int | None = None,
    ):
        settings = get_settings()
        self.bm25_path = bm25_path or settings.ingestion.bm25_path
        self.top_k = top_k or settings.retrieval.sparse_top_k

        if not os.path.exists(self.bm25_path):
            raise FileNotFoundError(
                f"BM25 index not found at {self.bm25_path}. Run the ingestion pipeline first."
            )

        logger.info("Loading BM25 index from %s", self.bm25_path)

        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301 — trusted local file
            self.bm25 = data["bm25"]
            self.chunks: List[Document] = data["chunks"]

        logger.info(
            "BM25 index loaded: %d documents in corpus", len(self.chunks)
        )

    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        """
        Perform keyword search and return the top-k matching chunks.

        Each returned document has a ``sparse_score`` field in its metadata.
        Only chunks with a positive BM25 score are returned.
        """
        k = top_k or self.top_k

        # Tokenize query identically to how the corpus was tokenized
        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        # Sort by descending score and take top-k
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        documents: List[Document] = []
        for i in top_indices:
            if scores[i] > 0:
                doc = self.chunks[i]
                doc.metadata["sparse_score"] = float(scores[i])
                documents.append(doc)

        logger.debug(
            "Sparse retrieval returned %d results for query", len(documents)
        )
        return documents
