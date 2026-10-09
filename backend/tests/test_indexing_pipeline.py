import io
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.document import ProcessedDocument, PageContent
from backend.app.services.embeddings.mock_provider import DeterministicLocalEmbeddingProvider
from backend.app.services.vector_store.sqlite_store import SQLiteVectorStore
from backend.app.services.vector_store.indexing import index_document, IndexingResult
from backend.app.services.vector_store.factory import get_vector_store
from backend.app.core.exceptions import AppException
from backend.tests.test_pdf_parser import create_sample_pdf_bytes

client = TestClient(app)


@pytest.mark.asyncio
async def test_indexing_pipeline_complete_flow(tmp_path):
    test_db = str(tmp_path / "test_indexing.db")
    store = SQLiteVectorStore(db_path=test_db)
    provider = DeterministicLocalEmbeddingProvider(dimension=64)

    doc = ProcessedDocument(
        doc_id="doc_test_101",
        filename="ai_overview.txt",
        file_type="txt",
        total_pages=1,
        text="Generative AI transforms text into multi-channel outputs like presentations and advisories.",
        character_count=87,
        pages=[PageContent(page_number=1, text="Generative AI transforms text into multi-channel outputs like presentations and advisories.", character_count=87)],
    )

    result = await index_document(
        document=doc,
        embedding_provider=provider,
        vector_store=store,
    )

    assert isinstance(result, IndexingResult)
    assert result.doc_id == "doc_test_101"
    assert result.chunks_indexed >= 1
    assert result.dimension == 64
    assert result.status == "indexed"

    # Verify vector store holds the chunk
    stored_chunks = await store.get_chunks_by_doc_id("doc_test_101")
    assert len(stored_chunks) == result.chunks_indexed
    assert stored_chunks[0].doc_id == "doc_test_101"
    assert len(stored_chunks[0].embedding) == 64
    assert "Generative AI" in stored_chunks[0].text

    # Verify document metadata was updated
    assert doc.metadata["indexed"] is True
    assert doc.metadata["chunks_count"] == result.chunks_indexed


@pytest.mark.asyncio
async def test_indexing_pipeline_empty_document_raises_error(tmp_path):
    test_db = str(tmp_path / "test_empty.db")
    store = SQLiteVectorStore(db_path=test_db)
    provider = DeterministicLocalEmbeddingProvider()

    doc = ProcessedDocument(
        doc_id="doc_empty",
        filename="empty.txt",
        file_type="txt",
        total_pages=1,
        text="   ",
        character_count=3,
    )

    with pytest.raises(AppException) as exc_info:
        await index_document(document=doc, embedding_provider=provider, vector_store=store)

    assert exc_info.value.code == "EMPTY_DOCUMENT_ERROR"


@pytest.mark.asyncio
async def test_indexing_pipeline_embedding_failure_handling(tmp_path):
    test_db = str(tmp_path / "test_emb_fail.db")
    store = SQLiteVectorStore(db_path=test_db)

    class FailingEmbeddingProvider:
        @property
        def model_name(self):
            return "failing-model"

        @property
        def dimension(self):
            return 128

        async def embed_documents(self, texts):
            raise RuntimeError("Embedding network connection refused")

    doc = ProcessedDocument(
        doc_id="doc_fail",
        filename="fail.txt",
        file_type="txt",
        total_pages=1,
        text="Valid text content that fails during embedding generation.",
        character_count=58,
    )

    with pytest.raises(AppException) as exc_info:
        await index_document(
            document=doc,
            embedding_provider=FailingEmbeddingProvider(),  # type: ignore[arg-type]
            vector_store=store,
        )

    assert exc_info.value.code == "EMBEDDING_FAILED"
    # Verify no chunks were indexed in vector store
    assert len(await store.get_chunks_by_doc_id("doc_fail")) == 0


def test_api_upload_and_auto_indexing():
    text_content = (
        b"SIH Hackathon Problem Statement. Generative AI content transformation pipeline. "
        b"This platform ingests multi-format documents and synthesizes tailored communication deliverables."
    )
    files = {"file": ("sih_guidelines.txt", io.BytesIO(text_content), "text/plain")}

    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()

    assert data["filename"] == "sih_guidelines.txt"
    assert data["metadata"]["indexed"] is True
    assert data["metadata"]["chunks_count"] >= 1
    doc_id = data["doc_id"]

    # Verify indexed in default vector store
    vector_store = get_vector_store()
    import asyncio
    chunks = asyncio.run(vector_store.get_chunks_by_doc_id(doc_id))
    assert len(chunks) == data["metadata"]["chunks_count"]

    # Test re-index endpoint
    reindex_res = client.post(f"/api/v1/documents/{doc_id}/index?chunk_size=300&chunk_overlap=50")
    assert reindex_res.status_code == 200
    reindex_data = reindex_res.json()
    assert reindex_data["doc_id"] == doc_id
    assert reindex_data["status"] == "indexed"

    # Delete document and verify vector store deletion
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
    chunks_after_del = asyncio.run(vector_store.get_chunks_by_doc_id(doc_id))
    assert len(chunks_after_del) == 0
