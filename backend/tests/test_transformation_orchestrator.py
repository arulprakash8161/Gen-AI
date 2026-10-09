import pytest
from unittest.mock import AsyncMock
from backend.app.core.exceptions import AppException
from backend.app.models.request import (
    OutputType,
    TransformationRequest,
    DetailLevel,
)
from backend.app.models.deliverable import TransformationResponse
from backend.app.services.rag.retriever import RAGRetriever, RAGRetrievalResult, RetrievedChunk
from backend.app.services.llm.mock_client import MockLLMClient
from backend.app.services.vector_store.sqlite_store import SQLiteVectorStore
from backend.app.services.transformation.orchestrator import TransformationOrchestrator
from backend.app.services.transformation.factory import get_transformation_orchestrator


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock(spec=RAGRetriever)
    retriever.vector_store = AsyncMock(spec=SQLiteVectorStore)

    # Standard retrieval result
    chunk1 = RetrievedChunk(
        chunk_id="doc_1_chunk_0",
        doc_id="doc_1",
        chunk_index=0,
        text="FastAPI provides high-performance asynchronous execution.",
        similarity_score=0.92,
        page_number=1,
        document_name="architecture.pdf",
        citation_header="[SOURCE CHUNK 0 — Document: architecture.pdf — Page: 1 — Chunk ID: doc_1_chunk_0]",
    )
    chunk2 = RetrievedChunk(
        chunk_id="doc_1_chunk_1",
        doc_id="doc_1",
        chunk_index=1,
        text="Deterministic embeddings ensure zero external API dependencies.",
        similarity_score=0.88,
        page_number=2,
        document_name="architecture.pdf",
        citation_header="[SOURCE CHUNK 1 — Document: architecture.pdf — Page: 2 — Chunk ID: doc_1_chunk_1]",
    )

    retriever.retrieve.return_value = RAGRetrievalResult(
        query="test query",
        retrieved_chunks=[chunk1, chunk2],
        formatted_context=(
            f"{chunk1.citation_header}\n{chunk1.text}\n\n"
            f"{chunk2.citation_header}\n{chunk2.text}"
        ),
        total_chunks_found=2,
        doc_id_filter="doc_1",
    )
    return retriever


@pytest.fixture
def mock_llm():
    return MockLLMClient()


@pytest.mark.asyncio
async def test_transform_single_output_type(mock_retriever, mock_llm):
    orchestrator = TransformationOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    request = TransformationRequest(
        document_id="doc_1",
        output_types=[OutputType.LINKEDIN],
        audience="executive",
        tone="professional",
        detail_level=DetailLevel.STANDARD.value,
    )

    response = await orchestrator.transform(request)

    assert isinstance(response, TransformationResponse)
    assert response.document_id == "doc_1"
    assert response.status == "completed"
    assert response.linkedin is not None
    assert response.summary is None
    assert response.advisory is None
    assert response.presentation is None
    assert response.video_package is None
    assert len(response.citations) == 2
    assert response.grounding_score > 0.0
    assert "elapsed_ms" in response.metadata


@pytest.mark.asyncio
async def test_transform_multi_output_types(mock_retriever, mock_llm):
    orchestrator = TransformationOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    request = TransformationRequest(
        document_id="doc_1",
        output_types=[
            OutputType.SUMMARY,
            OutputType.PRESENTATION,
            OutputType.VIDEO_PACKAGE,
        ],
        audience="general_public",
        tone="persuasive",
        detail_level=DetailLevel.COMPREHENSIVE.value,
        objective="Drive adoption of local edge AI architectures.",
    )

    response = await orchestrator.transform(request)

    assert response.linkedin is None
    assert response.summary is not None
    assert response.presentation is not None
    assert response.video_package is not None
    assert len(response.presentation.slides) >= 1
    assert len(response.video_package.scenes) >= 1

    # Verify retriever was called with comprehensive top_k=8
    mock_retriever.retrieve.assert_called_once()
    _, kwargs = mock_retriever.retrieve.call_args
    assert kwargs["top_k"] == 8
    assert kwargs["doc_id"] == "doc_1"


@pytest.mark.asyncio
async def test_transform_custom_query_and_top_k_override(mock_retriever, mock_llm):
    orchestrator = TransformationOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    request = TransformationRequest(
        document_id="doc_1",
        output_types=[OutputType.ADVISORY],
        custom_query="Specific vulnerability and patch recommendations",
        top_k=7,
    )

    response = await orchestrator.transform(request)
    assert response.advisory is not None

    _, kwargs = mock_retriever.retrieve.call_args
    assert kwargs["query"] == "Specific vulnerability and patch recommendations"
    assert kwargs["top_k"] == 7


@pytest.mark.asyncio
async def test_transform_document_not_indexed_error(mock_retriever, mock_llm):
    # Empty retrieval result
    mock_retriever.retrieve.return_value = RAGRetrievalResult(
        query="query",
        retrieved_chunks=[],
        formatted_context="",
        total_chunks_found=0,
        doc_id_filter="doc_missing",
    )
    # Vector store confirms no chunks exist for this doc_id
    mock_retriever.vector_store.get_chunks_by_doc_id.return_value = []

    orchestrator = TransformationOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    request = TransformationRequest(
        document_id="doc_missing",
        output_types=[OutputType.LINKEDIN],
    )

    with pytest.raises(AppException) as exc_info:
        await orchestrator.transform(request)

    assert exc_info.value.code == "DOCUMENT_NOT_INDEXED"
    assert exc_info.value.status_code == 404


def test_transformation_factory():
    orch1 = get_transformation_orchestrator()
    orch2 = get_transformation_orchestrator()
    assert orch1 is orch2
