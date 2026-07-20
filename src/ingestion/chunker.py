"""
Document chunking with deduplication for the ingestion pipeline.

Splits documents into semantically coherent chunks using recursive
character splitting and removes exact duplicates via SHA-256 hashing.
"""

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """
    Splits LangChain Documents into fixed-size chunks with overlap
    and removes exact-duplicate chunks via content hashing.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.ingestion.chunk_size
        self.chunk_overlap = chunk_overlap or settings.ingestion.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks and deduplicate."""
        logger.info(
            "Splitting %d documents (chunk_size=%d, overlap=%d)",
            len(documents),
            self.chunk_size,
            self.chunk_overlap,
        )

        raw_chunks = self.text_splitter.split_documents(documents)
        logger.info("Generated %d raw chunks", len(raw_chunks))

        unique_chunks = self._deduplicate(raw_chunks)
        logger.info(
            "Retained %d unique chunks after deduplication (removed %d duplicates)",
            len(unique_chunks),
            len(raw_chunks) - len(unique_chunks),
        )

        # Add sequential chunk index for citation tracking
        for i, chunk in enumerate(unique_chunks):
            chunk.metadata["chunk_index"] = i

        return unique_chunks

    @staticmethod
    def _deduplicate(chunks: list[Document]) -> list[Document]:
        """Remove chunks with identical text content using SHA-256 hashing."""
        seen_hashes: set = set()
        unique_chunks: list[Document] = []

        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                chunk.metadata["content_hash"] = content_hash
                unique_chunks.append(chunk)

        return unique_chunks


if __name__ == "__main__":
    test_docs = [
        Document(
            page_content="This is a test document. " * 100,
            metadata={"source": "test.txt"},
        ),
        Document(
            page_content="This is a test document. " * 100,
            metadata={"source": "test_duplicate.txt"},
        ),
    ]
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    result = chunker.chunk_documents(test_docs)
    logger.info("Test complete. Output chunks: %d", len(result))
