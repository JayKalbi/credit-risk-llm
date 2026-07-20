"""
Production Middleware for Hybrid RAG Engine.

Provides request-level security, validation, rate limiting, and audit
logging as Flask before/after request hooks.
"""

import time
import uuid
from collections import defaultdict

from flask import Flask, g, jsonify, request

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token-bucket rate limiter keyed by client IP.

    Limits each IP to `max_requests` within a rolling `window_seconds` window.
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if a request from the given IP is allowed."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Prune expired timestamps
        self._requests[client_ip] = [ts for ts in self._requests[client_ip] if ts > cutoff]

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for a client IP."""
        now = time.time()
        cutoff = now - self.window_seconds
        active = [ts for ts in self._requests[client_ip] if ts > cutoff]
        return max(0, self.max_requests - len(active))


class AuditLogger:
    """
    Immutable audit trail for all API interactions.

    Logs every query, response, sources used, and timing data.
    Critical for financial services compliance.
    """

    def __init__(self):
        self._audit_logger = get_logger("audit_trail")

    def log_query(
        self,
        request_id: str,
        client_ip: str,
        question: str,
        answer: str | None = None,
        sources: list | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        """Log a complete query lifecycle event."""
        entry = {
            "request_id": request_id,
            "client_ip": client_ip,
            "question": question[:500],  # Truncate for log safety
            "answer_length": len(answer) if answer else 0,
            "source_count": len(sources) if sources else 0,
            "latency_ms": round(latency_ms, 2) if latency_ms else None,
            "error": error,
            "timestamp": time.time(),
        }
        self._audit_logger.info("AUDIT | %s", entry)


# Module-level singletons
_rate_limiter: RateLimiter | None = None
_audit_logger: AuditLogger | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the rate limiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = RateLimiter(
            max_requests=settings.app.rate_limit_requests,
            window_seconds=settings.app.rate_limit_window_seconds,
        )
    return _rate_limiter


def get_audit_logger() -> AuditLogger:
    """Get or create the audit logger singleton."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def register_middleware(app: Flask) -> None:
    """
    Register all middleware hooks on the Flask application.

    This is the single entry point for middleware setup, called from app.py.
    """
    settings = get_settings()
    rate_limiter = get_rate_limiter()

    @app.before_request
    def before_request_handler():
        """Attach request ID, validate input, enforce rate limits."""
        # Attach unique request ID for traceability
        g.request_id = str(uuid.uuid4())[:8]
        g.start_time = time.time()

        # Skip middleware for health/static endpoints
        if request.path in ("/health", "/ready", "/"):
            return None

        # Rate limiting
        client_ip = request.remote_addr or "unknown"
        if not rate_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for IP %s on %s", client_ip, request.path)
            return (
                jsonify(
                    {
                        "error": "Rate limit exceeded. Please try again later.",
                        "retry_after_seconds": settings.app.rate_limit_window_seconds,
                    }
                ),
                429,
            )

        # Input validation for POST endpoints
        if request.method == "POST" and request.is_json:
            data = request.get_json(silent=True)
            if data and "question" in data:
                question = data["question"]
                if len(question) > settings.app.max_query_length:
                    max_len = settings.app.max_query_length
                    return (
                        jsonify({"error": f"Query too long. Max {max_len} chars."}),
                        400,
                    )

        return None

    @app.after_request
    def after_request_handler(response):
        """Add security headers and log request completion."""
        # Security headers
        response.headers["X-Request-ID"] = getattr(g, "request_id", "unknown")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Rate limit headers
        client_ip = request.remote_addr or "unknown"
        response.headers["X-RateLimit-Remaining"] = str(rate_limiter.get_remaining(client_ip))

        # Log request latency
        if hasattr(g, "start_time"):
            latency = (time.time() - g.start_time) * 1000
            logger.info(
                "[%s] %s %s → %d (%.1fms)",
                getattr(g, "request_id", "?"),
                request.method,
                request.path,
                response.status_code,
                latency,
            )

        return response
