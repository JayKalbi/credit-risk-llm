"""
Citation extraction and validation from LLM-generated responses.

Parses inline citations of the form [Source: filename, Chunk: index]
and validates them against the actually-retrieved source documents
to detect hallucinated citations.
"""

import re
from typing import Any, Dict, List

from src.logging_config import get_logger

logger = get_logger(__name__)

_CITATION_PATTERN = re.compile(r"\[Source:\s*(.*?),\s*Chunk:\s*(\d+)\]")


class CitationExtractor:
    """
    Parses LLM output to extract inline citations and validates
    that each cited source was actually in the retrieved context.

    Hallucinated citations (citing sources not provided) are flagged.
    """

    def extract_and_validate(
        self,
        llm_response: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extract all citations from the response and validate them.

        Args:
            llm_response: The raw LLM-generated text with inline citations.
            sources: List of source metadata dicts from retrieval.

        Returns:
            Dict containing:
              - clean_text: Response with citation markers stripped
              - citations: List of citation dicts with validation flags
              - hallucinated_count: Number of citations referencing
                                   sources not in the retrieved context
        """
        citations: List[Dict[str, Any]] = []
        hallucinated_count = 0

        matches = _CITATION_PATTERN.findall(llm_response)

        for filename, chunk_idx in matches:
            valid_source = None
            for s in sources:
                if (
                    s.get("filename") == filename
                    and str(s.get("chunk_index")) == chunk_idx
                ):
                    valid_source = s
                    break

            if valid_source:
                citations.append({
                    "filename": filename,
                    "chunk_index": int(chunk_idx),
                    "text_preview": valid_source.get("text_preview"),
                    "is_hallucinated": False,
                })
            else:
                hallucinated_count += 1
                citations.append({
                    "filename": filename,
                    "chunk_index": int(chunk_idx),
                    "text_preview": None,
                    "is_hallucinated": True,
                })

        if hallucinated_count:
            logger.warning(
                "Detected %d hallucinated citations in LLM response",
                hallucinated_count,
            )

        # Strip citation markers from clean text
        clean_text = _CITATION_PATTERN.sub("", llm_response).strip()

        return {
            "clean_text": clean_text,
            "formatted_text": llm_response,
            "citations": citations,
            "hallucinated_count": hallucinated_count,
        }
