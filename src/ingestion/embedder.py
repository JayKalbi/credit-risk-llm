"""
Hybrid embedding and indexing for dense (ChromaDB) and sparse (BM25) retrieval.

Handles generation of both vector embeddings and keyword indices,
with idempotent document ingestion using content-hash-based IDs.
"""

import hashlib
import os
import pickle
from typing import List

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class HybridEmbedder:
    """
    Manages dual-index creation for hybrid retrieval:
    - Dense: Sentence-Transformer embeddings stored in ChromaDB
    - Sparse: BM25Okapi keyword index stored via pickle

    Supports idempotent ingestion via content-hash-based document IDs.
    """

    def __init__(
        self,
        chroma_path: str | None = None,
        bm25_path: str | None = None,
    ):
        settings = get_settings()
        self.chroma_path = chroma_path or settings.ingestion.chroma_dir
        self.bm25_path = bm25_path or settings.ingestion.bm25_path

        os.makedirs(self.chroma_path, exist_ok=True)
        bm25_dir = os.path.dirname(self.bm25_path)
        if bm25_dir:
            os.makedirs(bm25_dir, exist_ok=True)

        model_name = settings.embedding.model_name
        logger.info("Initializing Sentence-Transformers (%s)", model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def ingest_dense(self, chunks: List[Document]) -> None:
        """
        Embed chunks into dense vectors and store in ChromaDB.

        Uses content-hash-based IDs for idempotent ingestion — running
        the pipeline twice will NOT create duplicates.
        """
        logger.info(
            "Generating dense embeddings for %d chunks → ChromaDB", len(chunks)
        )

        vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings,
        )

        # Generate stable IDs from content hashes for idempotent upsert
        doc_ids = []
        for chunk in chunks:
            content_hash = chunk.metadata.get("content_hash")
            if not content_hash:
                content_hash = hashlib.sha256(
                    chunk.page_content.encode("utf-8")
                ).hexdigest()
            doc_ids.append(content_hash)

        vectorstore.add_documents(documents=chunks, ids=doc_ids)
        logger.info("Successfully indexed %d chunks in ChromaDB", len(chunks))

    def ingest_sparse(self, chunks: List[Document]) -> None:
        """
        Create a BM25 keyword index and persist it to disk.

        NOTE: BM25Okapi is serialized via pickle. The integrity of this
        file should be verified in production environments. See the
        SparseRetriever for the corresponding load + verification logic.
        """
        logger.info("Building BM25 sparse index for %d chunks", len(chunks))

        tokenized_corpus = [
            chunk.page_content.lower().split() for chunk in chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)

        # Store BM25 model alongside original chunks
        # (BM25Okapi doesn't retain the source text internally)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks}, f)

        logger.info("Saved BM25 index to %s", self.bm25_path)
