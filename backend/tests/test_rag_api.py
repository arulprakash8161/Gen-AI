import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.vector_store.factory import get_vector_store

client = TestClient(app)


def test_rag_query_success_and_citations():
    # 1. Ingest a document with rich context
    raw_payload = {
        "text": (
            "The Smart India Hackathon 2024 problem statement focuses on AI Content Transformation. "
            "The system processes PDFs and DOCX files into multi-channel deliverables such as LinkedIn posts, "
            "presentations with speaker notes, executive summaries, and video scripts."
        ),
        "title": "sih_problem_statement.txt",
    }
    ingest_res = client.post("/api/v1/documents/raw", json=raw_payload)
    assert ingest_res.status_code == 201
    doc_id = ingest_res.json()["doc_id"]

    # 2. Semantic query
    query_payload = {
        "query": "What are the target deliverables in the SIH problem statement?",
        "top_k": 3,
        "score_threshold": 0.0,
        "doc_id": doc_id,
    }
    query_res = client.post("/api/v1/rag/query", json=query_payload)
    assert query_res.status_code == 200
    data = query_res.json()

    assert data["query"] == query_payload["query"]
    assert data["total_chunks_found"] >= 1
    assert len(data["retrieved_chunks"]) >= 1

    first_chunk = data["retrieved_chunks"][0]
    assert first_chunk["doc_id"] == doc_id
    assert "similarity_score" in first_chunk
    assert "text" in first_chunk
    assert "LinkedIn posts" in first_chunk["text"]
    assert "citation_header" in first_chunk
    assert "[SOURCE CHUNK 1 — Document: sih_problem_statement.txt — Page: 1 — Chunk ID:" in first_chunk["citation_header"]

    # Verify formatted context
    assert "[SOURCE CHUNK 1" in data["formatted_context"]
    assert "LinkedIn posts" in data["formatted_context"]


def test_rag_query_with_doc_id_filter():
    # Ingest two documents
    doc1_res = client.post("/api/v1/documents/raw", json={"text": "Unique apple fruit content.", "title": "apples.txt"})
    doc2_res = client.post("/api/v1/documents/raw", json={"text": "Unique banana fruit content.", "title": "bananas.txt"})
    doc1_id = doc1_res.json()["doc_id"]
    doc2_id = doc2_res.json()["doc_id"]

    # Query targeting only doc1
    res1 = client.post("/api/v1/rag/query", json={"query": "fruit content", "doc_id": doc1_id})
    assert res1.status_code == 200
    hits1 = res1.json()["retrieved_chunks"]
    assert all(h["doc_id"] == doc1_id for h in hits1)

    # Query targeting only doc2
    res2 = client.post("/api/v1/rag/query", json={"query": "fruit content", "doc_id": doc2_id})
    assert res2.status_code == 200
    hits2 = res2.json()["retrieved_chunks"]
    assert all(h["doc_id"] == doc2_id for h in hits2)


def test_rag_inspect_stored_chunks_and_deletion():
    # Ingest document
    doc_res = client.post("/api/v1/documents/raw", json={"text": "Chunk metadata inspection sample.", "title": "inspect.txt"})
    doc_id = doc_res.json()["doc_id"]

    # Inspect stored chunks
    chunks_res = client.get(f"/api/v1/rag/chunks/{doc_id}")
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    assert len(chunks) >= 1
    assert chunks[0]["doc_id"] == doc_id
    assert chunks[0]["text"] == "Chunk metadata inspection sample."
    # Ensure large embedding vectors are NOT returned
    assert "embedding" not in chunks[0]

    # Delete vectors for document
    del_res = client.delete(f"/api/v1/rag/documents/{doc_id}")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["status"] == "deleted"
    assert del_data["doc_id"] == doc_id
    assert del_data["deleted_chunks"] >= 1

    # Verify 404 after deletion
    chunks_after_del = client.get(f"/api/v1/rag/chunks/{doc_id}")
    assert chunks_after_del.status_code == 404
    assert chunks_after_del.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_rag_query_invalid_inputs():
    # Empty query string
    res_empty = client.post("/api/v1/rag/query", json={"query": ""})
    assert res_empty.status_code == 422

    # Whitespace-only query string
    res_ws = client.post("/api/v1/rag/query", json={"query": "    "})
    assert res_ws.status_code == 422

    # Invalid top_k (< 1)
    res_top_k = client.post("/api/v1/rag/query", json={"query": "valid query", "top_k": 0})
    assert res_top_k.status_code == 422

    # Invalid score_threshold (> 1.0)
    res_thresh = client.post("/api/v1/rag/query", json={"query": "valid query", "score_threshold": 1.5})
    assert res_thresh.status_code == 422


def test_rag_missing_document_handling():
    # Query with non-existent doc_id
    res_query = client.post("/api/v1/rag/query", json={"query": "test query", "doc_id": "non_existent_doc_id_999"})
    assert res_query.status_code == 404
    assert res_query.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    # Chunks inspection for non-existent doc_id
    res_chunks = client.get("/api/v1/rag/chunks/non_existent_doc_id_999")
    assert res_chunks.status_code == 404
    assert res_chunks.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    # Deletion for non-existent doc_id
    res_del = client.delete("/api/v1/rag/documents/non_existent_doc_id_999")
    assert res_del.status_code == 404
    assert res_del.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_rag_storage_error_handling(monkeypatch):
    # Ingest document first
    doc_res = client.post("/api/v1/documents/raw", json={"text": "Error handling test.", "title": "err.txt"})
    doc_id = doc_res.json()["doc_id"]

    vector_store = get_vector_store()

    # Monkeypatch similarity_search to simulate a database/storage failure
    async def failing_similarity_search(*args, **kwargs):
        raise RuntimeError("Database connection lock failure")

    monkeypatch.setattr(vector_store, "similarity_search", failing_similarity_search)

    res = client.post("/api/v1/rag/query", json={"query": "Error query"})
    assert res.status_code == 500
    assert res.json()["error"]["code"] == "VECTOR_SEARCH_FAILED"
