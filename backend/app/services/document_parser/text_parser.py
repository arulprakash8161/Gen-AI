import uuid
from typing import Optional
from backend.app.models.document import ProcessedDocument, PageContent
from backend.app.services.document_parser.cleaner import clean_text
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def parse_text(
    content: str,
    filename: str = "pasted_text.txt",
    doc_id: Optional[str] = None,
) -> ProcessedDocument:
    """
    Parses, cleans, and normalizes raw string content or plain text.
    """
    if not content or not content.strip():
        raise AppException(
            message=f"Text content for '{filename}' is empty.",
            status_code=400,
            code="EMPTY_CONTENT",
        )

    document_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
    cleaned = clean_text(content)

    if not cleaned:
        raise AppException(
            message=f"Text content for '{filename}' is empty after cleaning.",
            status_code=400,
            code="EMPTY_CONTENT",
        )

    pages = [
        PageContent(
            page_number=1,
            text=cleaned,
            character_count=len(cleaned),
        )
    ]

    return ProcessedDocument(
        doc_id=document_id,
        filename=filename,
        file_type="txt",
        total_pages=1,
        text=cleaned,
        character_count=len(cleaned),
        pages=pages,
        metadata={"source": "plain_text"},
    )


def parse_text_bytes(
    file_bytes: bytes,
    filename: str = "document.txt",
    doc_id: Optional[str] = None,
) -> ProcessedDocument:
    """
    Parses plain text from raw bytes, trying UTF-8 and falling back to Latin-1.
    """
    if not file_bytes:
        raise AppException(
            message=f"Text file '{filename}' is empty.",
            status_code=400,
            code="EMPTY_FILE",
        )

    try:
        decoded_text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded_text = file_bytes.decode("latin-1")
        except Exception as exc:
            raise AppException(
                message=f"Could not decode text file '{filename}'. Unsupported encoding.",
                status_code=400,
                code="ENCODING_ERROR",
                details=str(exc),
            )

    return parse_text(decoded_text, filename=filename, doc_id=doc_id)
