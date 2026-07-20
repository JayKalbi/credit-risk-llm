"""Unit tests for the HybridRetriever's RRF fusion logic."""

from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever


class TestReciprocalRankFusion:
    """Test the RRF merging algorithm in isolation."""

    def _make_doc(self, content_hash: str, text: str = "test") -> Document:
        """Helper to create a Document with a content_hash."""
        return Document(
            page_content=text,
            metadata={"content_hash": content_hash, "filename": "test.pdf", "chunk_index": 0},
        )

    def test_rrf_combines_scores(self):
        """Documents in both lists should have combined RRF scores."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60

        doc_a = self._make_doc("aaa", "Document A")
        doc_b = self._make_doc("bbb", "Document B")

        # A appears in both lists, B only in dense
        dense = [doc_a, doc_b]
        sparse = [self._make_doc("aaa", "Document A")]

        result = retriever._reciprocal_rank_fusion(dense, sparse)

        scores = {doc.metadata["content_hash"]: doc.metadata["rrf_score"] for doc in result}

        # A should have higher score (appears in both)
        assert scores["aaa"] > scores["bbb"]

    def test_rrf_deduplicates(self):
        """Same document appearing in both lists should appear once in output."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60

        doc = self._make_doc("same_hash")
        result = retriever._reciprocal_rank_fusion([doc], [self._make_doc("same_hash")])

        hashes = [d.metadata["content_hash"] for d in result]
        assert len(hashes) == 1

    def test_rrf_empty_lists(self):
        """Should handle empty input lists."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60

        result = retriever._reciprocal_rank_fusion([], [])
        assert result == []

    def test_rrf_single_list(self):
        """Should work with one empty and one populated list."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60

        docs = [self._make_doc("x"), self._make_doc("y")]
        result = retriever._reciprocal_rank_fusion(docs, [])

        assert len(result) == 2
        for doc in result:
            assert "rrf_score" in doc.metadata
            assert doc.metadata["rrf_score"] > 0

    def test_rrf_score_formula(self):
        """Verify the RRF formula: score = 1/(k + rank)."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60

        doc = self._make_doc("only_doc")
        result = retriever._reciprocal_rank_fusion([doc], [])

        expected_score = 1.0 / (60 + 1)  # rank 0, so k + 0 + 1 = 61
        assert abs(result[0].metadata["rrf_score"] - expected_score) < 1e-9

    def test_rrf_ordering_is_descending(self):
        """Output should be sorted by RRF score in descending order."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60

        docs = [self._make_doc(f"doc_{i}") for i in range(5)]
        result = retriever._reciprocal_rank_fusion(docs, [])

        scores = [d.metadata["rrf_score"] for d in result]
        assert scores == sorted(scores, reverse=True)
