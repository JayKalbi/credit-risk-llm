"""
Dense vector retriever using ChromaDB and Sentence-Transformers.

Performs semantic similarity search over document embeddings to find
chunks whose meaning is closest to the query.
"""

import os

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Semantic vector search using ChromaDB.

    Uses the same embedding model as the ingestion pipeline to ensure
    query-document embedding compatibility.
    """

    def __init__(
        self,
        chroma_path: str | None = None,
        top_k: int | None = None,
    ):
        settings = get_settings()
        self.chroma_path = chroma_path or settings.ingestion.chroma_dir
        self.top_k = top_k or settings.retrieval.dense_top_k

        if not os.path.exists(self.chroma_path):
            raise FileNotFoundError(
                f"ChromaDB not found at {self.chroma_path}. Run the ingestion pipeline first."
            )

        model_name = settings.embedding.model_name
        logger.info("Initializing Dense Retriever (ChromaDB + %s)", model_name)

        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """
        Perform similarity search and return the top-k matching chunks.

        Each returned document has a ``dense_score`` field in its metadata.
        """
        k = top_k or self.top_k

        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

        documents: list[Document] = []
        for doc, score in results:
            doc.metadata["dense_score"] = score
            documents.append(doc)

        logger.debug("Dense retrieval returned %d results for query", len(documents))
        return documents
