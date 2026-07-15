"""Unit tests for the CitationExtractor."""

import pytest

from src.generation.citation_extractor import CitationExtractor


class TestCitationExtractor:
    """Test citation parsing and validation."""

    def setup_method(self):
        self.extractor = CitationExtractor()

    def test_extracts_valid_citation(self, sample_sources):
        """Should extract and validate a known citation."""
        response = (
            "Apple's revenue was $383.29 billion "
            "[Source: Apple 10-K.pdf, Chunk: 5]."
        )
        result = self.extractor.extract_and_validate(response, sample_sources)

        assert len(result["citations"]) == 1
        assert result["citations"][0]["filename"] == "Apple 10-K.pdf"
        assert result["citations"][0]["chunk_index"] == 5
        assert result["citations"][0]["is_hallucinated"] is False

    def test_detects_hallucinated_citation(self, sample_sources):
        """Should flag citations not in the source list."""
        response = "Revenue data [Source: Fake_Report.pdf, Chunk: 99]."
        result = self.extractor.extract_and_validate(response, sample_sources)

        assert len(result["citations"]) == 1
        assert result["citations"][0]["is_hallucinated"] is True
        assert result["hallucinated_count"] == 1

    def test_multiple_citations(self, sample_sources):
        """Should extract all citations from response."""
        response = (
            "Apple had $383B [Source: Apple 10-K.pdf, Chunk: 5]. "
            "Tesla had $77B [Source: tsla-10k.pdf, Chunk: 12]."
        )
        result = self.extractor.extract_and_validate(response, sample_sources)
        assert len(result["citations"]) == 2
        assert result["hallucinated_count"] == 0

    def test_no_citations(self, sample_sources):
        """Should handle responses with no citations."""
        response = "I don't have enough information to answer this."
        result = self.extractor.extract_and_validate(response, sample_sources)

        assert result["citations"] == []
        assert result["hallucinated_count"] == 0

    def test_clean_text_strips_citations(self, sample_sources):
        """clean_text should have citation markers removed."""
        response = "Revenue was $383B [Source: Apple 10-K.pdf, Chunk: 5]."
        result = self.extractor.extract_and_validate(response, sample_sources)

        assert "[Source:" not in result["clean_text"]
        assert "Revenue" in result["clean_text"]

    def test_formatted_text_preserves_citations(self, sample_sources):
        """formatted_text should keep the original citation markers."""
        response = "Revenue [Source: Apple 10-K.pdf, Chunk: 5]."
        result = self.extractor.extract_and_validate(response, sample_sources)
        assert "[Source:" in result["formatted_text"]

    def test_mixed_valid_and_hallucinated(self, sample_sources):
        """Should correctly identify a mix of valid and fake citations."""
        response = (
            "Real [Source: Apple 10-K.pdf, Chunk: 5]. "
            "Fake [Source: NoFile.pdf, Chunk: 0]."
        )
        result = self.extractor.extract_and_validate(response, sample_sources)

        assert len(result["citations"]) == 2
        assert result["hallucinated_count"] == 1
        valid = [c for c in result["citations"] if not c["is_hallucinated"]]
        assert len(valid) == 1

    def test_empty_response(self, sample_sources):
        """Should handle empty response string."""
        result = self.extractor.extract_and_validate("", sample_sources)
        assert result["citations"] == []
        assert result["hallucinated_count"] == 0
