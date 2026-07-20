"""
Cross-encoder reranker for second-stage retrieval refinement.

Re-scores candidate documents by computing full cross-attention
between the query and each document simultaneously, which is far
more accurate than bi-encoder similarity.
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidate documents using a Cross-Encoder model.

    Cross-Encoders jointly encode the (query, document) pair and
    produce a relevance score, unlike Bi-Encoders which encode
    them independently.
    """

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
    ):
        settings = get_settings()
        self.top_k = top_k or settings.retrieval.final_top_k
        model = model_name or settings.embedding.cross_encoder_model

        logger.info("Initializing Cross-Encoder Reranker (%s)", model)
        self.model = CrossEncoder(model)

    def rerank(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> list[Document]:
        """
        Score all documents against the query and return the top-k.

        Each returned document has a ``rerank_score`` field in its metadata.
        """
        if not documents:
            return []

        k = top_k or self.top_k

        # Cross-encoder expects (query, passage) pairs
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores, strict=False):
            doc.metadata["rerank_score"] = float(score)

        ranked_docs = sorted(documents, key=lambda d: d.metadata["rerank_score"], reverse=True)

        logger.debug("Reranked %d candidates → returning top %d", len(documents), k)
        return ranked_docs[:k]
