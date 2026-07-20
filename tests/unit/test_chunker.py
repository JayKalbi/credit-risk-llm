"""Unit tests for the DocumentChunker."""

from langchain_core.documents import Document

from src.ingestion.chunker import DocumentChunker


class TestDocumentChunker:
    """Test document chunking and deduplication."""

    def test_basic_chunking(self, sample_documents):
        """Should split documents into chunks smaller than chunk_size."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents(sample_documents)

        assert len(chunks) > len(sample_documents)
        for chunk in chunks:
            # Allow some tolerance for recursive splitter
            assert len(chunk.page_content) <= 150

    def test_chunk_index_assigned(self, sample_documents):
        """Every chunk should have a sequential chunk_index in metadata."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents(sample_documents)

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_content_hash_assigned(self, sample_documents):
        """Every chunk should have a content_hash in metadata."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents(sample_documents)

        for chunk in chunks:
            assert "content_hash" in chunk.metadata
            assert len(chunk.metadata["content_hash"]) == 64  # SHA-256

    def test_deduplication_removes_exact_copies(self):
        """Identical documents should be deduplicated."""
        identical_text = "This is an identical document. " * 20
        docs = [
            Document(page_content=identical_text, metadata={"source": "a.txt"}),
            Document(page_content=identical_text, metadata={"source": "b.txt"}),
        ]
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk_documents(docs)

        # Same text → same hash → deduplication should remove duplicates
        hashes = [c.metadata["content_hash"] for c in chunks]
        assert len(hashes) == len(set(hashes))

    def test_empty_input(self):
        """Should handle empty document list gracefully."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents([])
        assert chunks == []

    def test_preserves_original_metadata(self, sample_documents):
        """Chunking should preserve the original document metadata."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents(sample_documents)

        # All chunks should retain their source filename
        for chunk in chunks:
            assert "filename" in chunk.metadata

    def test_custom_chunk_size(self):
        """Custom chunk sizes should be respected."""
        # Use unique words so deduplication doesn't collapse identical chunks
        doc = Document(
            page_content=" ".join(f"word{i}" for i in range(1000)),
            metadata={"filename": "test.txt"},
        )
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk_documents([doc])
        assert len(chunks) > 10  # ~6000 unique chars / 50 = many chunks
