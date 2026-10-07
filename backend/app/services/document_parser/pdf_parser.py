import io
import uuid
from typing import Optional, Dict, Any
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from backend.app.models.document import ProcessedDocument, PageContent
from backend.app.services.document_parser.cleaner import clean_text
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def parse_pdf(
    file_bytes: bytes,
    filename: str = "document.pdf",
    doc_id: Optional[str] = None,
) -> ProcessedDocument:
    """
    Extracts text from a PDF byte stream, tracking page numbers and document metadata.
    """
    if not file_bytes:
        raise AppException(
            message=f"PDF file '{filename}' is empty.",
            status_code=400,
            code="EMPTY_FILE",
        )

    document_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except (PdfReadError, Exception) as exc:
        logger.error(f"Failed to read PDF '{filename}': {str(exc)}")
        raise AppException(
            message=f"Could not parse PDF file '{filename}'. The file may be corrupt or invalid.",
            status_code=400,
            code="PDF_PARSE_ERROR",
            details=str(exc),
        )

    if reader.is_encrypted:
        try:
            # Attempt blank password decrypt
            decrypt_result = reader.decrypt("")
            if not decrypt_result:
                raise AppException(
                    message=f"PDF file '{filename}' is password protected and cannot be opened.",
                    status_code=400,
                    code="PDF_ENCRYPTED",
                )
        except Exception:
            raise AppException(
                message=f"PDF file '{filename}' is password protected.",
                status_code=400,
                code="PDF_ENCRYPTED",
            )

    pages: list[PageContent] = []
    page_texts: list[str] = []

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise AppException(
            message=f"PDF file '{filename}' has no pages.",
            status_code=400,
            code="PDF_NO_PAGES",
        )

    for index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as page_err:
            logger.warning(f"Error extracting text from page {index} of '{filename}': {page_err}")
            raw_text = ""

        cleaned = clean_text(raw_text)
        pages.append(
            PageContent(
                page_number=index,
                text=cleaned,
                character_count=len(cleaned),
            )
        )
        if cleaned:
            page_texts.append(cleaned)

    full_text = "\n\n".join(page_texts)

    # Extract metadata safely
    raw_meta = reader.metadata or {}
    pdf_meta: Dict[str, Any] = {}
    for key, value in raw_meta.items():
        clean_key = str(key).lstrip("/").lower()
        if isinstance(value, (str, int, float, bool)):
            pdf_meta[clean_key] = value

    pdf_meta["total_pages"] = total_pages

    return ProcessedDocument(
        doc_id=document_id,
        filename=filename,
        file_type="pdf",
        total_pages=total_pages,
        text=full_text,
        character_count=len(full_text),
        pages=pages,
        metadata=pdf_meta,
    )
