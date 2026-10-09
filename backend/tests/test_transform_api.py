import pytest
from starlette.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_get_transform_options_api():
    resp = client.get("/api/v1/transform/options")
    assert resp.status_code == 200
    data = resp.json()
    assert "output_types" in data
    assert "audiences" in data
    assert "tones" in data
    assert "detail_levels" in data

    output_vals = [item["value"] for item in data["output_types"]]
    assert "linkedin" in output_vals
    assert "summary" in output_vals
    assert "advisory" in output_vals
    assert "presentation" in output_vals
    assert "video_package" in output_vals


def test_transform_document_api_success():
    # 1. Ingest a document
    doc_payload = {
        "text": (
            "The Department of Telecommunications has announced the rollout of advanced 5G networks "
            "across 100 enterprise hubs. The project achieved a 40% reduction in network latency "
            "and deployed quantum-safe encryption across all core routing nodes. Operations require "
            "compliance with the National Cybersecurity Framework within 90 days."
        ),
        "filename": "telecom_directive.txt",
    }
    upload_resp = client.post("/api/v1/documents/raw", json=doc_payload)
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["doc_id"]

    # 2. Trigger transformation for LinkedIn and Summary
    transform_payload = {
        "document_id": doc_id,
        "output_types": ["linkedin", "summary"],
        "audience": "executive",
        "tone": "authoritative",
        "detail_level": "standard",
        "objective": "Highlight quantum-safe security and regulatory compliance.",
    }
    trans_resp = client.post("/api/v1/transform", json=transform_payload)
    assert trans_resp.status_code == 200
    data = trans_resp.json()

    assert data["document_id"] == doc_id
    assert data["status"] == "completed"
    assert data["linkedin"] is not None
    assert data["summary"] is not None
    assert data["advisory"] is None
    assert data["presentation"] is None
    assert data["video_package"] is None

    # Check LinkedIn deliverable structure
    linkedin = data["linkedin"]
    assert "headline" in linkedin
    assert "content" in linkedin
    assert "call_to_action" in linkedin
    assert len(linkedin["source_chunk_ids"]) > 0

    # Check Summary deliverable structure
    summary = data["summary"]
    assert "title" in summary
    assert "executive_brief" in summary
    assert len(summary["key_findings"]) > 0

    # Check Citations and Grounding
    assert len(data["citations"]) > 0
    assert data["grounding_score"] > 0.0
    assert "elapsed_ms" in data["metadata"]


def test_transform_all_five_deliverables_api():
    # 1. Ingest a document
    doc_payload = {
        "text": (
            "Autonomous systems reduce operational costs by 35% through algorithmic decisioning. "
            "All autonomous operations must maintain human-in-the-loop oversight to ensure safety."
        ),
        "filename": "autonomous_guidelines.txt",
    }
    upload_resp = client.post("/api/v1/documents/raw", json=doc_payload)
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["doc_id"]

    # 2. Request all 5 deliverable formats
    transform_payload = {
        "document_id": doc_id,
        "output_types": [
            "linkedin",
            "summary",
            "advisory",
            "presentation",
            "video_package",
        ],
        "detail_level": "high_level",
    }
    trans_resp = client.post("/api/v1/transform", json=transform_payload)
    assert trans_resp.status_code == 200
    data = trans_resp.json()

    assert data["linkedin"] is not None
    assert data["summary"] is not None
    assert data["advisory"] is not None
    assert data["presentation"] is not None
    assert data["video_package"] is not None

    # Verify presentation slides
    assert len(data["presentation"]["slides"]) >= 1
    # Verify video script scenes
    assert len(data["video_package"]["scenes"]) >= 1


def test_transform_document_not_indexed_returns_404():
    transform_payload = {
        "document_id": "doc_unindexed_99999",
        "output_types": ["summary"],
    }
    resp = client.post("/api/v1/transform", json=transform_payload)
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"]["code"] == "DOCUMENT_NOT_INDEXED"


def test_transform_document_validation_errors():
    # Empty output_types
    resp1 = client.post("/api/v1/transform", json={
        "document_id": "doc_123",
        "output_types": [],
    })
    assert resp1.status_code == 422

    # Blank document_id
    resp2 = client.post("/api/v1/transform", json={
        "document_id": "   ",
        "output_types": ["summary"],
    })
    assert resp2.status_code == 422

    # Invalid output_type string
    resp3 = client.post("/api/v1/transform", json={
        "document_id": "doc_123",
        "output_types": ["invalid_format_xyz"],
    })
    assert resp3.status_code == 422
