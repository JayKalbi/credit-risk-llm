"""Unit tests for the CrossEncoderReranker."""

from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.documents import Document

from src.retrieval.reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Test reranking logic (with mocked model)."""

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_sorts_by_score(self, mock_ce_class):
        """Should return documents sorted by descending rerank score."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker(top_k=3)

        docs = [
            Document(page_content="Low relevance", metadata={}),
            Document(page_content="High relevance", metadata={}),
            Document(page_content="Medium relevance", metadata={}),
        ]

        result = reranker.rerank("test query", docs)

        assert len(result) == 3
        assert result[0].page_content == "High relevance"
        assert result[0].metadata["rerank_score"] == 0.9
        assert result[1].page_content == "Medium relevance"
        assert result[2].page_content == "Low relevance"

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_top_k(self, mock_ce_class):
        """Should respect top_k parameter."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3, 0.7, 0.1, 0.9, 0.5])
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker(top_k=2)
        docs = [Document(page_content=f"Doc {i}", metadata={}) for i in range(5)]

        result = reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_empty_input(self, mock_ce_class):
        """Should return empty list for empty input."""
        mock_ce_class.return_value = MagicMock()
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [])
        assert result == []

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_attaches_scores(self, mock_ce_class):
        """Every document should have rerank_score in metadata."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.42, 0.78])
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker(top_k=5)
        docs = [
            Document(page_content="A", metadata={}),
            Document(page_content="B", metadata={}),
        ]

        result = reranker.rerank("query", docs)
        for doc in result:
            assert "rerank_score" in doc.metadata
            assert isinstance(doc.metadata["rerank_score"], float)
