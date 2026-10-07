import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_health():
    """Verify that root /health endpoint responds with 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "project" in data


def test_api_v1_health():
    """Verify that /api/v1/health returns complete status metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data


def test_api_v1_readiness():
    """Verify that /api/v1/health/ready reports system readiness."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "checks" in data


def test_404_error_formatting():
    """Verify that non-existent routes return standardized JSON error format."""
    response = client.get("/api/v1/non_existent_route")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_404"
