"""Unit tests for the MultiFormatDocumentLoader."""

import pytest
from pathlib import Path

from src.ingestion.document_loader import MultiFormatDocumentLoader


class TestMultiFormatDocumentLoader:
    """Test document loading from filesystem."""

    def test_nonexistent_directory_returns_empty(self):
        """Should return empty list for non-existent directory."""
        loader = MultiFormatDocumentLoader("/this/path/does/not/exist")
        docs = loader.load()
        assert docs == []

    def test_empty_directory_returns_empty(self, temp_dir):
        """Should return empty list for empty directory."""
        loader = MultiFormatDocumentLoader(str(temp_dir))
        docs = loader.load()
        assert docs == []

    def test_loads_txt_files(self, temp_dir):
        """Should successfully load .txt files."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("This is test content.", encoding="utf-8")

        loader = MultiFormatDocumentLoader(str(temp_dir))
        docs = loader.load()

        assert len(docs) == 1
        assert "This is test content." in docs[0].page_content
        assert docs[0].metadata["filename"] == "test.txt"

    def test_skips_unsupported_formats(self, temp_dir):
        """Should skip files with unsupported extensions."""
        (temp_dir / "data.csv").write_text("a,b,c", encoding="utf-8")
        (temp_dir / "image.png").write_bytes(b"\x89PNG")

        loader = MultiFormatDocumentLoader(str(temp_dir))
        docs = loader.load()
        assert docs == []

    def test_supported_extensions_defined(self):
        """Should have a defined set of supported extensions."""
        assert ".pdf" in MultiFormatDocumentLoader.SUPPORTED_EXTENSIONS
        assert ".txt" in MultiFormatDocumentLoader.SUPPORTED_EXTENSIONS
        assert ".html" in MultiFormatDocumentLoader.SUPPORTED_EXTENSIONS
        assert ".htm" in MultiFormatDocumentLoader.SUPPORTED_EXTENSIONS
        assert ".md" in MultiFormatDocumentLoader.SUPPORTED_EXTENSIONS

    def test_filename_in_metadata(self, temp_dir):
        """Every loaded document should have filename in metadata."""
        (temp_dir / "report.txt").write_text("Report content", encoding="utf-8")

        loader = MultiFormatDocumentLoader(str(temp_dir))
        docs = loader.load()

        for doc in docs:
            assert "filename" in doc.metadata
            assert doc.metadata["filename"] == "report.txt"

    def test_loads_multiple_files(self, temp_dir):
        """Should load all supported files from a directory."""
        (temp_dir / "file1.txt").write_text("Content 1", encoding="utf-8")
        (temp_dir / "file2.txt").write_text("Content 2", encoding="utf-8")

        loader = MultiFormatDocumentLoader(str(temp_dir))
        docs = loader.load()
        assert len(docs) == 2

    def test_recursive_loading(self, temp_dir):
        """Should load files from subdirectories recursively."""
        subdir = temp_dir / "subfolder"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content", encoding="utf-8")

        loader = MultiFormatDocumentLoader(str(temp_dir))
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "nested.txt"
