"""
Multi-format document loader for the ingestion pipeline.

Scans a directory and loads PDF, TXT, HTML, and Markdown files into
LangChain Document objects while preserving source metadata.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from src.logging_config import get_logger

logger = get_logger(__name__)

# Mapping of file extensions to their LangChain loader classes
_LOADER_MAP = {
    ".pdf": lambda p: PyPDFLoader(str(p)),
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".html": lambda p: UnstructuredHTMLLoader(str(p)),
    ".htm": lambda p: UnstructuredHTMLLoader(str(p)),
    ".md": lambda p: UnstructuredMarkdownLoader(str(p)),
}


class MultiFormatDocumentLoader:
    """
    Scans a directory recursively and loads all supported document formats
    (PDF, TXT, HTML, MD) into LangChain Document objects.

    Each document's metadata is enriched with the source filename for
    downstream citation tracking.
    """

    SUPPORTED_EXTENSIONS = set(_LOADER_MAP.keys())

    def __init__(self, directory_path: str):
        self.directory_path = Path(directory_path)

    def load(self) -> List[Document]:
        """Load all supported documents from the configured directory."""
        if not self.directory_path.exists():
            logger.error("Directory does not exist: %s", self.directory_path)
            return []

        documents: List[Document] = []
        skipped = 0
        errors = 0

        logger.info("Scanning directory: %s", self.directory_path)

        for file_path in sorted(self.directory_path.rglob("*")):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                skipped += 1
                continue

            docs = self._load_single_file(file_path)
            if docs:
                for doc in docs:
                    doc.metadata["filename"] = file_path.name
                    doc.metadata["source_path"] = str(file_path)
                documents.extend(docs)
            else:
                errors += 1

        logger.info(
            "Loaded %d pages/documents (skipped %d unsupported, %d errors)",
            len(documents),
            skipped,
            errors,
        )
        return documents

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single file using the appropriate LangChain loader."""
        ext = file_path.suffix.lower()
        loader_factory = _LOADER_MAP.get(ext)

        if loader_factory is None:
            return []

        try:
            loader = loader_factory(file_path)
            return loader.load()
        except Exception:
            logger.warning("Failed to load %s", file_path.name, exc_info=True)
            return []


if __name__ == "__main__":
    from src.config import get_settings

    settings = get_settings()
    loader = MultiFormatDocumentLoader(settings.ingestion.documents_dir)
    docs = loader.load()
    if docs:
        logger.info(
            "Sample from %s: %s...",
            docs[0].metadata["filename"],
            docs[0].page_content[:100],
        )
