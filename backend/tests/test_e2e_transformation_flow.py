import io
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.tests.test_docx_parser import create_sample_docx_bytes

client = TestClient(app)


def test_end_to_end_docx_ingestion_indexing_and_transformation():
    """
    Validates complete lifecycle:
    Document Upload (DOCX) -> Auto-Indexing -> Vector Storage -> Multi-Channel Transformation -> Citations & Grounding
    """
    # 1. Prepare multi-section DOCX with realistic governmental / enterprise directive text
    paragraphs = [
        "MINISTRY OF COMMUNICATIONS & DIGITAL GOVERNANCE",
        "Directive on Sovereign Artificial Intelligence Infrastructure.",
        (
            "The strategic initiative mandates all public departments to deploy localized, "
            "air-gapped language models to guarantee zero data leakage outside sovereign borders."
        ),
        "OPERATIONAL BENCHMARKS & TIMELINES",
        (
            "Phase 1 requires 100 enterprise pilot installations by December 2026. "
            "Latency benchmarks must not exceed 250 milliseconds for localized vector search. "
            "Departments failing to meet compliance will face administrative reviews."
        ),
    ]
    docx_bytes = create_sample_docx_bytes(paragraphs)
    files = {
        "file": (
            "sovereign_ai_directive.docx",
            io.BytesIO(docx_bytes),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }

    # 2. Upload and Ingest -> Auto-indexed into persistent vector database
    upload_res = client.post("/api/v1/documents/upload", files=files)
    assert upload_res.status_code == 201
    doc_data = upload_res.json()
    doc_id = doc_data["doc_id"]
    assert doc_data["filename"] == "sovereign_ai_directive.docx"
    assert doc_data["file_type"] == "docx"
    assert doc_data["metadata"]["indexed"] is True
    assert doc_data["metadata"]["chunks_count"] >= 1

    # 3. Transform into all 5 deliverables
    transform_payload = {
        "document_id": doc_id,
        "output_types": [
            "linkedin",
            "summary",
            "advisory",
            "presentation",
            "video_package",
        ],
        "audience": "executive",
        "tone": "authoritative",
        "detail_level": "standard",
        "objective": "Highlight localized deployment timelines and latency benchmarks.",
    }
    trans_res = client.post("/api/v1/transform", json=transform_payload)
    assert trans_res.status_code == 200
    res_data = trans_res.json()

    # 4. Verify orchestration metadata
    assert res_data["document_id"] == doc_id
    assert res_data["status"] == "completed"
    assert res_data["grounding_score"] > 0.0
    assert len(res_data["citations"]) >= 1
    assert any("sovereign_ai_directive.docx" in c for c in res_data["citations"])

    # 5. Verify LinkedIn Deliverable
    linkedin = res_data["linkedin"]
    assert linkedin is not None
    assert linkedin["headline"]
    assert linkedin["content"]
    assert linkedin["call_to_action"]
    assert len(linkedin["source_chunk_ids"]) >= 1

    # 6. Verify Executive Summary Deliverable
    summary = res_data["summary"]
    assert summary is not None
    assert summary["title"]
    assert summary["executive_brief"]
    assert len(summary["key_findings"]) >= 1
    assert len(summary["action_items"]) >= 1

    # 7. Verify Official Advisory Deliverable
    advisory = res_data["advisory"]
    assert advisory is not None
    assert advisory["reference_number"]
    assert advisory["subject"]
    assert advisory["urgency_level"]
    assert len(advisory["advisory_details"]) >= 1

    # 8. Verify Presentation Deliverable
    presentation = res_data["presentation"]
    assert presentation is not None
    assert presentation["presentation_title"]
    assert len(presentation["slides"]) >= 1
    first_slide = presentation["slides"][0]
    assert first_slide["title"]
    assert len(first_slide["bullet_points"]) >= 1

    # 9. Verify Video Package Deliverable
    video = res_data["video_package"]
    assert video is not None
    assert video["video_title"]
    assert video["hook"]
    assert len(video["scenes"]) >= 1
    assert video["outro_cta"]
