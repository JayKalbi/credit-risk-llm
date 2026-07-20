"""
Document ingestion pipeline orchestrator.

Coordinates the full ingestion workflow:
  1. Load documents from a directory
  2. Chunk and deduplicate
  3. Index into dense (ChromaDB) and sparse (BM25) stores
"""

import argparse
import time

from src.config import get_settings
from src.ingestion.chunker import DocumentChunker
from src.ingestion.document_loader import MultiFormatDocumentLoader
from src.ingestion.embedder import HybridEmbedder
from src.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the full ingestion pipeline with CLI argument support."""
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Hybrid RAG Engine — Document Ingestion Pipeline")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=settings.ingestion.documents_dir,
        help="Directory containing raw documents",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=settings.ingestion.chunk_size,
        help="Size of each text chunk",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=settings.ingestion.chunk_overlap,
        help="Overlap between consecutive chunks",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Hybrid RAG Engine — Ingestion Pipeline")
    logger.info("=" * 60)

    start_time = time.time()

    # Step 1: Load Documents
    logger.info("Step 1/3: Loading documents from %s", args.input_dir)
    loader = MultiFormatDocumentLoader(directory_path=args.input_dir)
    documents = loader.load()

    if not documents:
        logger.error("No documents loaded. Exiting pipeline.")
        return

    # Step 2: Chunk & Deduplicate
    logger.info("Step 2/3: Chunking & Deduplication")
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = chunker.chunk_documents(documents)

    # Step 3: Embed & Index (Dense + Sparse)
    logger.info("Step 3/3: Hybrid Indexing (Dense + Sparse)")
    embedder = HybridEmbedder()
    embedder.ingest_sparse(chunks)
    embedder.ingest_dense(chunks)

    elapsed = time.time() - start_time
    logger.info("Pipeline complete. Processed %d chunks in %.2f seconds.", len(chunks), elapsed)


if __name__ == "__main__":
    main()
