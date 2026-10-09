from typing import Dict, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, status
from pydantic import BaseModel, Field

from backend.app.models.document import ProcessedDocument, DocumentChunk, ChunkingConfig
from backend.app.services.document_parser.parser import parse_document
from backend.app.services.document_parser.text_parser import parse_text
from backend.app.services.chunking.text_chunker import chunk_document
from backend.app.services.vector_store import (
    index_document,
    IndexingResult,
    get_vector_store,
)
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# In-memory document registry for parsed documents
_DOCUMENTS_STORE: Dict[str, ProcessedDocument] = {}


class RawTextInput(BaseModel):
    text: str = Field(..., description="Raw text content to ingest")
    title: Optional[str] = Field("pasted_text.txt", description="Optional title or identifier")


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    total_pages: int
    character_count: int
    created_at: str


@router.post(
    "/upload",
    response_model=ProcessedDocument,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Extract Document",
    description="Upload a PDF, DOCX, or text file. Extracts, cleans, and normalizes its content.",
)
async def upload_document(
    file: UploadFile = File(..., description="File to upload (PDF, DOCX, TXT)"),
) -> ProcessedDocument:
    filename = file.filename or "uploaded_file"
    try:
        content = await file.read()
    except Exception as exc:
        raise AppException(
            message=f"Failed to read uploaded file '{filename}'",
            status_code=400,
            code="FILE_READ_ERROR",
            details=str(exc),
        )

    processed_doc = parse_document(file_bytes=content, filename=filename)
    if processed_doc.text and processed_doc.text.strip():
        await index_document(processed_doc)
    else:
        processed_doc.metadata["indexed"] = False
        processed_doc.metadata["chunks_count"] = 0
        processed_doc.metadata["indexing_note"] = "Skipped: document contains no extractable text"

    _DOCUMENTS_STORE[processed_doc.doc_id] = processed_doc
    logger.info(
        f"Ingested document '{filename}' with ID '{processed_doc.doc_id}' "
        f"({processed_doc.character_count} chars, {processed_doc.metadata.get('chunks_count', 0)} chunks)"
    )
    return processed_doc


@router.post(
    "/raw",
    response_model=ProcessedDocument,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Raw Text",
    description="Ingest plain text directly, extracting, chunking, and indexing into vector store.",
)
async def ingest_raw_text(payload: RawTextInput) -> ProcessedDocument:
    processed_doc = parse_text(content=payload.text, filename=payload.title or "pasted_text.txt")
    if processed_doc.text and processed_doc.text.strip():
        await index_document(processed_doc)
    else:
        processed_doc.metadata["indexed"] = False
        processed_doc.metadata["chunks_count"] = 0
        processed_doc.metadata["indexing_note"] = "Skipped: document contains no extractable text"

    _DOCUMENTS_STORE[processed_doc.doc_id] = processed_doc
    logger.info(
        f"Ingested raw text with ID '{processed_doc.doc_id}' "
        f"({processed_doc.character_count} chars, {processed_doc.metadata.get('chunks_count', 0)} chunks)"
    )
    return processed_doc


@router.get(
    "",
    response_model=List[DocumentSummary],
    status_code=status.HTTP_200_OK,
    summary="List Ingested Documents",
    description="Lists all currently ingested documents in memory.",
)
async def list_documents() -> List[DocumentSummary]:
    return [
        DocumentSummary(
            doc_id=doc.doc_id,
            filename=doc.filename,
            file_type=doc.file_type,
            total_pages=doc.total_pages,
            character_count=doc.character_count,
            created_at=doc.created_at,
        )
        for doc in _DOCUMENTS_STORE.values()
    ]


@router.get(
    "/{doc_id}",
    response_model=ProcessedDocument,
    status_code=status.HTTP_200_OK,
    summary="Get Document Details",
    description="Retrieves the full extracted content and metadata of a specific document.",
)
async def get_document(doc_id: str) -> ProcessedDocument:
    if doc_id not in _DOCUMENTS_STORE:
        raise AppException(
            message=f"Document with ID '{doc_id}' was not found.",
            status_code=404,
            code="DOCUMENT_NOT_FOUND",
        )
    return _DOCUMENTS_STORE[doc_id]


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Removes an ingested document from memory and its vector embeddings from vector store.",
)
async def delete_document(doc_id: str) -> Dict[str, str]:
    if doc_id not in _DOCUMENTS_STORE:
        raise AppException(
            message=f"Document with ID '{doc_id}' was not found.",
            status_code=404,
            code="DOCUMENT_NOT_FOUND",
        )
    del _DOCUMENTS_STORE[doc_id]
    vector_store = get_vector_store()
    await vector_store.delete_by_doc_id(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@router.post(
    "/{doc_id}/index",
    response_model=IndexingResult,
    status_code=status.HTTP_200_OK,
    summary="Index Document Chunks into Vector Store",
    description="Explicitly indexes or re-indexes an ingested document into the persistent vector store.",
)
async def index_existing_document(
    doc_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> IndexingResult:
    if doc_id not in _DOCUMENTS_STORE:
        raise AppException(
            message=f"Document with ID '{doc_id}' was not found.",
            status_code=404,
            code="DOCUMENT_NOT_FOUND",
        )
    doc = _DOCUMENTS_STORE[doc_id]
    config = None
    if chunk_size is not None or chunk_overlap is not None:
        config = ChunkingConfig(
            chunk_size=chunk_size if chunk_size is not None else 600,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else 100,
        )
    return await index_document(doc, chunking_config=config)


@router.get(
    "/{doc_id}/chunks",
    response_model=List[DocumentChunk],
    status_code=status.HTTP_200_OK,
    summary="Get Document Chunks",
    description="Chunks a previously ingested document into traceable chunks using configurable chunk_size and chunk_overlap.",
)
async def get_document_chunks(
    doc_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[DocumentChunk]:
    if doc_id not in _DOCUMENTS_STORE:
        raise AppException(
            message=f"Document with ID '{doc_id}' was not found.",
            status_code=404,
            code="DOCUMENT_NOT_FOUND",
        )

    doc = _DOCUMENTS_STORE[doc_id]
    config = None
    if chunk_size is not None or chunk_overlap is not None:
        config = ChunkingConfig(
            chunk_size=chunk_size if chunk_size is not None else 600,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else 100,
        )

    return chunk_document(doc, config=config)
