"""Integration tests for the Flask API endpoints."""

import json

import pytest

from app.app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoints:
    """Test health and readiness probes."""

    def test_health_returns_200(self, client):
        """Health endpoint should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"

    def test_ready_returns_status(self, client):
        """Ready endpoint should return model loading status."""
        response = client.get("/ready")
        data = json.loads(response.data)
        assert "status" in data
        assert "models_loaded" in data

    def test_root_returns_html(self, client):
        """Root endpoint should serve the dashboard HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Hybrid RAG Intelligence" in response.data


class TestAskEndpoint:
    """Test the /api/ask endpoint validation."""

    def test_missing_question_returns_400(self, client):
        """Should return 400 when question is missing."""
        response = client.post(
            "/api/ask",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_empty_question_returns_400(self, client):
        """Should return 400 for empty question string."""
        response = client.post(
            "/api/ask",
            data=json.dumps({"question": "   "}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_no_json_body_returns_400(self, client):
        """Should return 400 when no JSON body is sent."""
        response = client.post("/api/ask", data="not json")
        assert response.status_code == 400


class TestErrorHandlers:
    """Test custom error handlers."""

    def test_404_returns_json(self, client):
        """404 should return JSON error, not HTML."""
        response = client.get("/nonexistent/path")
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data

    def test_405_returns_json(self, client):
        """405 should return JSON error for wrong HTTP method."""
        response = client.get("/api/ask")  # GET instead of POST
        assert response.status_code == 405
        data = json.loads(response.data)
        assert "error" in data


class TestSecurityHeaders:
    """Test that security headers are present."""

    def test_security_headers_on_response(self, client):
        """Responses should include security headers."""
        response = client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert "X-Request-ID" in response.headers
