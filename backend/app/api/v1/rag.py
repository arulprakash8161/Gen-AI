from typing import Any, Dict, List, Optional
from fastapi import APIRouter, status, Query
from pydantic import BaseModel, Field, field_validator

from backend.app.services.rag.retriever import (
    RAGRetriever,
    RetrievedChunk,
    RAGRetrievalResult,
    get_rag_retriever,
)
from backend.app.services.vector_store.factory import get_vector_store
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG & Retrieval"])


class RAGQueryRequest(BaseModel):
    """
    Query payload for semantic RAG retrieval.
    """
    query: str = Field(..., description="User search query string")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of relevant chunks to retrieve")
    score_threshold: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Minimum cosine similarity score required for inclusion"
    )
    doc_id: Optional[str] = Field(None, description="Optional document ID to restrict search scope to")
    deduplicate: bool = Field(default=True, description="Whether to deduplicate repeating text content")

    @field_validator("query")
    @classmethod
    def validate_non_empty_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string cannot be empty or whitespace.")
        return v.strip()


class StoredChunkMetadata(BaseModel):
    """
    Chunk metadata model exposing chunk attributes and content
    without transferring large raw embedding vectors over HTTP.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="0-indexed position within parent document")
    text: str = Field(..., description="Cleaned chunk text content")
    page_number: Optional[int] = Field(None, description="Source page number")
    character_count: int = Field(default=0, description="Character count")
    token_count: int = Field(default=0, description="Estimated token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom document and chunk metadata")


class DeleteDocumentVectorsResponse(BaseModel):
    status: str = Field("deleted", description="Operation status")
    doc_id: str = Field(..., description="Deleted document ID")
    deleted_chunks: int = Field(..., ge=0, description="Number of vector chunks deleted")


@router.post(
    "/query",
    response_model=RAGRetrievalResult,
    status_code=status.HTTP_200_OK,
    summary="Semantic RAG Query",
    description="Retrieves ranked relevant chunks and structured citations for a query. Does not call LLM.",
)
async def query_rag(payload: RAGQueryRequest) -> RAGRetrievalResult:
    retriever: RAGRetriever = get_rag_retriever()
    vector_store = get_vector_store()

    # If document filtering is requested, verify the document exists in vector store
    if payload.doc_id:
        existing_chunks = await vector_store.get_chunks_by_doc_id(payload.doc_id)
        if not existing_chunks:
            raise AppException(
                message=f"Document with ID '{payload.doc_id}' was not found in vector store.",
                status_code=404,
                code="DOCUMENT_NOT_FOUND",
            )

    try:
        result = await retriever.retrieve(
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            doc_id=payload.doc_id,
            deduplicate=payload.deduplicate,
        )
        return result
    except AppException:
        raise
    except Exception as exc:
        logger.error(f"RAG query failed: {exc}")
        raise AppException(
            message=f"RAG retrieval query failed: {str(exc)}",
            status_code=500,
            code="RAG_QUERY_ERROR",
            details=str(exc),
        )


@router.get(
    "/chunks/{doc_id}",
    response_model=List[StoredChunkMetadata],
    status_code=status.HTTP_200_OK,
    summary="Inspect Stored Chunks for Document",
    description="Lists all indexed chunks and metadata for a document without returning raw embedding vectors.",
)
async def get_document_stored_chunks(doc_id: str) -> List[StoredChunkMetadata]:
    clean_doc_id = doc_id.strip() if doc_id else ""
    if not clean_doc_id:
        raise AppException(
            message="Document ID cannot be empty.",
            status_code=400,
            code="INVALID_DOCUMENT_ID",
        )

    vector_store = get_vector_store()
    try:
        chunks = await vector_store.get_chunks_by_doc_id(clean_doc_id)
    except Exception as exc:
        logger.error(f"Failed to fetch chunks for doc_id='{clean_doc_id}': {exc}")
        raise AppException(
            message=f"Failed to retrieve chunks from storage: {str(exc)}",
            status_code=500,
            code="STORAGE_READ_ERROR",
            details=str(exc),
        )

    if not chunks:
        raise AppException(
            message=f"Document with ID '{clean_doc_id}' was not found in the vector store.",
            status_code=404,
            code="DOCUMENT_NOT_FOUND",
        )

    return [
        StoredChunkMetadata(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            chunk_index=c.chunk_index,
            text=c.text,
            page_number=c.page_number,
            character_count=c.character_count,
            token_count=c.token_count,
            metadata=c.metadata,
        )
        for c in chunks
    ]


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentVectorsResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Document Vectors",
    description="Deletes all indexed vectors for the specified document ID without touching unrelated documents.",
)
async def delete_document_vectors(doc_id: str) -> DeleteDocumentVectorsResponse:
    clean_doc_id = doc_id.strip() if doc_id else ""
    if not clean_doc_id:
        raise AppException(
            message="Document ID cannot be empty.",
            status_code=400,
            code="INVALID_DOCUMENT_ID",
        )

    vector_store = get_vector_store()
    try:
        deleted_count = await vector_store.delete_by_doc_id(clean_doc_id)
    except Exception as exc:
        logger.error(f"Failed to delete document vectors for doc_id='{clean_doc_id}': {exc}")
        raise AppException(
            message=f"Failed to delete document vectors from storage: {str(exc)}",
            status_code=500,
            code="STORAGE_DELETE_ERROR",
            details=str(exc),
        )

    if deleted_count == 0:
        raise AppException(
            message=f"Document with ID '{clean_doc_id}' was not found in the vector store.",
            status_code=404,
            code="DOCUMENT_NOT_FOUND",
        )

    return DeleteDocumentVectorsResponse(
        status="deleted",
        doc_id=clean_doc_id,
        deleted_chunks=deleted_count,
    )
