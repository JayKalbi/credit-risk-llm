"""
Synthetic golden dataset generator for RAG evaluation.

Samples chunks from the vector store and uses an LLM to generate
question-answer pairs that can be used as ground truth for
automated evaluation.
"""

import json
import random
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

_QA_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""You are an expert financial analyst. Read the following excerpt from a company's SEC 10-K filing.

EXCERPT:
{context}

TASK:
Write one highly specific, factual question that can be answered SOLELY using this excerpt.
Then, provide the exact, correct answer based ONLY on the text.
Format your output exactly like this, with no other conversational text:
QUESTION: <your question>
ANSWER: <your answer>""",
)


class GoldenDatasetGenerator:
    """
    Generates synthetic Q&A pairs by sampling chunks from ChromaDB
    and prompting an LLM to create questions and ground-truth answers.
    """

    def __init__(
        self,
        chroma_path: str | None = None,
        num_samples: int = 20,
        seed: int = 42,
    ):
        settings = get_settings()
        self.chroma_path = chroma_path or settings.ingestion.chroma_dir
        self.num_samples = num_samples
        self.seed = seed

        model_name = settings.embedding.model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings,
        )

        load_dotenv()
        self.llm = ChatGroq(
            temperature=0.7,
            model_name=settings.llm.model_name,
            api_key=settings.llm.api_key,
        )
        self.prompt = _QA_PROMPT

    def generate(self, output_path: str | None = None) -> List[dict]:
        """Generate the golden dataset and save to JSON."""
        settings = get_settings()
        output_path = output_path or settings.golden_dataset_path

        logger.info("Fetching chunks from ChromaDB for Q&A generation")

        # Retrieve a broad sample of chunks
        collection = self.vectorstore._collection
        all_docs = collection.get(include=["documents", "metadatas"])

        if not all_docs["documents"]:
            logger.error("No documents found in ChromaDB")
            return []

        # Build (text, metadata) pairs and sample
        random.seed(self.seed)
        indices = list(range(len(all_docs["documents"])))
        sample_size = min(self.num_samples, len(indices))
        sampled_indices = random.sample(indices, sample_size)

        dataset = []
        logger.info("Generating %d Q&A pairs with Groq LLM", sample_size)

        for i, idx in enumerate(sampled_indices):
            text = all_docs["documents"][idx]
            metadata = all_docs["metadatas"][idx] if all_docs["metadatas"] else {}

            try:
                prompt_text = self.prompt.format(context=text)
                response = self.llm.invoke(prompt_text).content

                # Parse structured output
                question = ""
                answer = ""
                for line in response.split("\n"):
                    if line.startswith("QUESTION:"):
                        question = line.replace("QUESTION:", "").strip()
                    elif line.startswith("ANSWER:"):
                        answer = line.replace("ANSWER:", "").strip()

                if question and answer:
                    dataset.append({
                        "question": question,
                        "ground_truth": answer,
                        "source_chunk": text,
                        "source_file": metadata.get("filename", "unknown"),
                    })
                    logger.info(
                        "  [%d/%d] Generated Q&A pair", i + 1, sample_size
                    )
                else:
                    logger.warning(
                        "  [%d/%d] Failed to parse LLM output", i + 1, sample_size
                    )

            except Exception:
                logger.error(
                    "  [%d/%d] Error generating Q&A", i + 1, sample_size, exc_info=True
                )

        # Save to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)

        logger.info(
            "Golden dataset saved: %d pairs → %s", len(dataset), output_path
        )
        return dataset


if __name__ == "__main__":
    generator = GoldenDatasetGenerator(num_samples=10, seed=42)
    generator.generate()
