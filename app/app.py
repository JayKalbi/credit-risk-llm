"""
Flask Application — Hybrid RAG Engine API Server.

Production-hardened with:
  - Health check and readiness endpoints
  - CORS configuration
  - Request validation and rate limiting (via middleware)
  - Structured error handling
  - Lazy model loading
  - Security headers
"""

import time
from typing import Optional

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from src.config import get_settings
from src.generation.citation_extractor import CitationExtractor
from src.logging_config import configure_root_logger, get_logger
from src.middleware import get_audit_logger, register_middleware

logger = get_logger(__name__)

# ── Application Factory ─────────────────────────────────────────────
app = Flask(__name__)
settings = get_settings()

# Configure logging
configure_root_logger(
    level=settings.log_level,
    json_format=(settings.environment == "production"),
)

# CORS
CORS(app, origins=settings.app.cors_origins.split(","))

# Register middleware (rate limiting, validation, security headers)
register_middleware(app)

# ── Lazy-loaded RAG Components ──────────────────────────────────────
_rag_chain = None
_citation_extractor = None


def _get_rag_chain():
    """Lazy-initialize the RAG chain (avoids blocking app startup)."""
    global _rag_chain
    if _rag_chain is None:
        from src.generation.rag_chain import RAGChain

        logger.info("Initializing RAG Chain (loading ML models)...")
        _rag_chain = RAGChain()
        logger.info("RAG Chain ready")
    return _rag_chain


def _get_citation_extractor():
    """Lazy-initialize the citation extractor."""
    global _citation_extractor
    if _citation_extractor is None:
        _citation_extractor = CitationExtractor()
    return _citation_extractor


# ── Health & Readiness ──────────────────────────────────────────────


@app.route("/health")
def health():
    """Liveness probe — always returns 200 if the process is running."""
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route("/ready")
def ready():
    """Readiness probe — returns 200 only when ML models are loaded."""
    if _rag_chain is not None:
        return jsonify({"status": "ready", "models_loaded": True})
    return jsonify({"status": "not_ready", "models_loaded": False}), 503


# ── Main UI ─────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Render the multi-panel dashboard."""
    return render_template("index.html")


# ── API Endpoint ────────────────────────────────────────────────────


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    Process a question through the Hybrid RAG pipeline.

    Request Body:
        {"question": "What is Apple's total revenue?"}

    Response:
        {
            "answer": "...",
            "formatted_answer": "...",
            "citations": [...],
            "sources": [...],
            "metrics": {...}
        }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("question"):
        return jsonify({"error": "Missing 'question' in request body"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    audit = get_audit_logger()
    start = time.time()

    try:
        rag = _get_rag_chain()
        extractor = _get_citation_extractor()

        # Execute pipeline
        raw_result = rag.query(question)

        # Extract and validate citations
        extracted = extractor.extract_and_validate(
            llm_response=raw_result["answer"],
            sources=raw_result["sources"],
        )

        latency_ms = (time.time() - start) * 1000

        response_data = {
            "answer": extracted["formatted_text"],
            "clean_answer": extracted["clean_text"],
            "citations": extracted["citations"],
            "hallucinated_citations": extracted["hallucinated_count"],
            "sources": raw_result["sources"],
            "metrics": {
                **raw_result.get("metrics", {}),
                "generation_latency_ms": round(latency_ms, 1),
            },
        }

        # Audit log
        audit.log_query(
            request_id=getattr(request, "request_id", "unknown"),
            client_ip=request.remote_addr or "unknown",
            question=question,
            answer=extracted["clean_text"],
            sources=raw_result["sources"],
            latency_ms=latency_ms,
        )

        return jsonify(response_data)

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.error("Pipeline error: %s", str(e), exc_info=True)

        audit.log_query(
            request_id=getattr(request, "request_id", "unknown"),
            client_ip=request.remote_addr or "unknown",
            question=question,
            error=str(e),
            latency_ms=latency_ms,
        )

        return jsonify({"error": "An internal error occurred. Please try again."}), 500


# ── Error Handlers ──────────────────────────────────────────────────


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Development server — use waitress for production:
    #   waitress-serve --host=0.0.0.0 --port=5000 app.app:app
    logger.info(
        "Starting development server on %s:%d",
        settings.app.host,
        settings.app.port,
    )
    app.run(
        host=settings.app.host,
        port=settings.app.port,
        debug=False,  # NEVER enable debug in production
    )
