"""
Grounded LLM generator with enforced citation and hallucination prevention.

Uses Groq's API (Llama 3.1) with a strict system prompt that enforces:
  - Context-only grounding (no external knowledge)
  - Inline citation format [Source: filename, Chunk: index]
  - Explicit "I don't know" for unanswerable questions
"""

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert financial analyst assistant. Your task is to answer the user's question based strictly on the provided context documents.

RULES:
1. ONLY use the information provided in the Context below. Do not use outside knowledge.
2. If the answer is not contained in the context, you MUST say exactly: "I don't have enough information to answer this." Do not attempt to guess or hallucinate.
3. For EVERY factual claim you make, you MUST include an inline citation referencing the exact document it came from. Use the format [Source: <filename>, Chunk: <chunk_index>].

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):""",
)


class GroundedGenerator:
    """
    LLM generator that enforces strict grounding on retrieved context.

    All answers must cite their sources. Unanswerable questions receive
    an explicit "I don't know" response instead of hallucination.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
    ):
        settings = get_settings()
        model = model_name or settings.llm.model_name
        temp = temperature if temperature is not None else settings.llm.temperature
        api_key = settings.llm.api_key

        if not api_key:
            logger.warning("GROQ_API_KEY not set — generation will fail at runtime")

        logger.info("Initializing Grounded Generator (Groq %s)", model)
        self.llm = ChatGroq(
            temperature=temp,
            model_name=model,
            api_key=api_key,
        )
        self.prompt_template = _SYSTEM_PROMPT

    def generate(self, question: str, retrieved_docs: list[Document]) -> str:
        """
        Generate a grounded answer based on retrieved documents.

        Args:
            question: The user's question.
            retrieved_docs: Documents retrieved by the hybrid retriever.

        Returns:
            The LLM-generated answer string with inline citations.
        """
        if not retrieved_docs:
            return "I don't have enough information to answer this."

        # Format context with source metadata so the LLM can cite correctly
        context_parts = []
        for doc in retrieved_docs:
            filename = doc.metadata.get("filename", "Unknown_File")
            chunk_idx = doc.metadata.get("chunk_index", "0")
            header = f"--- SOURCE: filename={filename}, chunk_index={chunk_idx} ---"
            context_parts.append(f"{header}\n{doc.page_content}")

        formatted_context = "\n\n".join(context_parts)

        prompt_text = self.prompt_template.format(context=formatted_context, question=question)

        logger.debug("Sending prompt to Groq (%d chars)", len(prompt_text))
        response = self.llm.invoke(prompt_text)
        return response.content
