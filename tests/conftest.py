"""
Shared test fixtures and configuration.

Provides mock data, temporary directories, and reusable fixtures
for unit and integration tests.
"""

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── Environment Setup ───────────────────────────────────────────────
# Set test env vars BEFORE importing any src modules
os.environ.setdefault("GROQ_API_KEY", "test_key_not_real")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("LOG_LEVEL", "WARNING")


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_documents() -> List[Document]:
    """Provide a list of sample LangChain Documents for testing."""
    return [
        Document(
            page_content="Apple Inc. reported total net sales of $383.29 billion "
            "for the fiscal year ended September 2023. The company's "
            "iPhone segment contributed $200.58 billion in revenue.",
            metadata={"filename": "Apple 10-K.pdf", "source_path": "/data/Apple 10-K.pdf"},
        ),
        Document(
            page_content="Tesla's total automotive revenues were $77.07 billion "
            "in 2024, representing a decrease of 6% compared to 2023. "
            "Energy generation and storage revenue increased by 67%.",
            metadata={"filename": "tsla-10k.pdf", "source_path": "/data/tsla-10k.pdf"},
        ),
        Document(
            page_content="The company maintains a strong balance sheet with "
            "cash and cash equivalents of $29.97 billion. Long-term "
            "debt obligations totaled $95.28 billion as of year-end.",
            metadata={"filename": "Apple 10-K.pdf", "source_path": "/data/Apple 10-K.pdf"},
        ),
    ]


@pytest.fixture
def sample_chunks() -> List[Document]:
    """Provide pre-chunked documents with metadata."""
    return [
        Document(
            page_content="Apple's total net sales were $383.29 billion.",
            metadata={
                "filename": "Apple 10-K.pdf",
                "chunk_index": 0,
                "content_hash": "abc123",
            },
        ),
        Document(
            page_content="Tesla's automotive revenues were $77.07 billion.",
            metadata={
                "filename": "tsla-10k.pdf",
                "chunk_index": 1,
                "content_hash": "def456",
            },
        ),
        Document(
            page_content="The company had cash of $29.97 billion.",
            metadata={
                "filename": "Apple 10-K.pdf",
                "chunk_index": 2,
                "content_hash": "ghi789",
            },
        ),
    ]


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_sources() -> list:
    """Provide sample source metadata as returned by RAGChain.query()."""
    return [
        {
            "filename": "Apple 10-K.pdf",
            "chunk_index": 5,
            "text_preview": "Apple's total net sales were $383.29 billion...",
            "rerank_score": 0.892,
        },
        {
            "filename": "tsla-10k.pdf",
            "chunk_index": 12,
            "text_preview": "Tesla's automotive revenues were $77.07 billion...",
            "rerank_score": 0.743,
        },
    ]
