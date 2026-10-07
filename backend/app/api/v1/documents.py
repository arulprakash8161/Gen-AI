from typing import Dict, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, status
from pydantic import BaseModel, Field

from backend.app.models.document import ProcessedDocument
from backend.app.services.document_parser.parser import parse_document
from backend.app.services.document_parser.text_parser import parse_text
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
    _DOCUMENTS_STORE[processed_doc.doc_id] = processed_doc
    logger.info(f"Ingested document '{filename}' with ID '{processed_doc.doc_id}' ({processed_doc.character_count} chars)")
    return processed_doc


@router.post(
    "/raw",
    response_model=ProcessedDocument,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Raw Text",
    description="Ingest plain text directly without a file upload.",
)
async def ingest_raw_text(payload: RawTextInput) -> ProcessedDocument:
    processed_doc = parse_text(content=payload.text, filename=payload.title or "pasted_text.txt")
    _DOCUMENTS_STORE[processed_doc.doc_id] = processed_doc
    logger.info(f"Ingested raw text with ID '{processed_doc.doc_id}' ({processed_doc.character_count} chars)")
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
    description="Removes an ingested document from memory.",
)
async def delete_document(doc_id: str) -> Dict[str, str]:
    if doc_id not in _DOCUMENTS_STORE:
        raise AppException(
            message=f"Document with ID '{doc_id}' was not found.",
            status_code=404,
            code="DOCUMENT_NOT_FOUND",
        )
    del _DOCUMENTS_STORE[doc_id]
    return {"status": "deleted", "doc_id": doc_id}
