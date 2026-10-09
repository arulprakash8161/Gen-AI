import io
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.vector_store.sqlite_store import SQLiteVectorStore
from backend.app.services.vector_store.factory import get_vector_store
from backend.app.services.rag.retriever import get_rag_retriever
from backend.tests.test_docx_parser import create_sample_docx_bytes

client = TestClient(app)


def test_end_to_end_ingestion_indexing_retrieval_and_persistence(tmp_path):
    # 1. Prepare sample DOCX with realistic domain content
    docx_paragraphs = [
        "Smart India Hackathon 2024: Problem Statement Detailed Brief.",
        "The AI Content Transformation Platform ingests multi-format source data including PDF, DOCX, and advisories.",
        "It synthesizes tailored communication deliverables: LinkedIn posts, executive summaries, official advisories, slide presentations, and video packages.",
        "Strict source grounding is enforced via vector retrieval to prevent hallucinated assertions.",
    ]
    docx_bytes = create_sample_docx_bytes(docx_paragraphs)
    files = {"file": ("sih_brief.docx", io.BytesIO(docx_bytes), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}

    # 2. Upload and Ingest -> Triggers extraction, chunking, embedding, vector store persistence
    upload_res = client.post("/api/v1/documents/upload", files=files)
    assert upload_res.status_code == 201
    doc_data = upload_res.json()
    doc_id = doc_data["doc_id"]
    assert doc_data["filename"] == "sih_brief.docx"
    assert doc_data["file_type"] == "docx"
    assert doc_data["metadata"]["indexed"] is True
    assert doc_data["metadata"]["chunks_count"] >= 1

    # 3. Inspect stored chunks via RAG API
    chunks_res = client.get(f"/api/v1/rag/chunks/{doc_id}")
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    assert len(chunks) == doc_data["metadata"]["chunks_count"]
    assert chunks[0]["doc_id"] == doc_id
    assert "Strict source grounding" in chunks[0]["text"]
    assert "embedding" not in chunks[0]  # Verify embedding not exposed in API

    # 4. Semantic RAG Query with Citations
    query_payload = {
        "query": "How is source grounding enforced in the platform?",
        "top_k": 3,
        "score_threshold": 0.0,
        "doc_id": doc_id,
    }
    query_res = client.post("/api/v1/rag/query", json=query_payload)
    assert query_res.status_code == 200
    query_data = query_res.json()

    assert query_data["total_chunks_found"] >= 1
    top_hit = query_data["retrieved_chunks"][0]
    assert top_hit["doc_id"] == doc_id
    assert "Strict source grounding" in top_hit["text"]
    assert "citation_header" in top_hit
    assert "[SOURCE CHUNK 1 — Document: sih_brief.docx — Page: 1 — Chunk ID:" in top_hit["citation_header"]
    assert top_hit["citation_header"] in query_data["formatted_context"]

    # 5. Verify Persistence across simulated backend restart
    vector_store = get_vector_store()
    fresh_store = SQLiteVectorStore(db_path=vector_store.db_path)
    import asyncio
    persisted_chunks = asyncio.run(fresh_store.get_chunks_by_doc_id(doc_id))
    assert len(persisted_chunks) == doc_data["metadata"]["chunks_count"]
    assert persisted_chunks[0].chunk_id == chunks[0]["chunk_id"]

    # 6. Delete document vectors via RAG endpoint
    del_res = client.delete(f"/api/v1/rag/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_chunks"] >= 1

    # 7. Verify chunks are gone from vector store
    get_after_del = client.get(f"/api/v1/rag/chunks/{doc_id}")
    assert get_after_del.status_code == 404
    assert get_after_del.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
