import os
from typing import Optional
from backend.app.models.document import ProcessedDocument
from backend.app.services.document_parser.pdf_parser import parse_pdf
from backend.app.services.document_parser.docx_parser import parse_docx
from backend.app.services.document_parser.text_parser import parse_text_bytes
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "txt",
    ".markdown": "txt",
    ".json": "txt",
    ".csv": "txt",
}


def parse_document(
    file_bytes: bytes,
    filename: str,
    doc_id: Optional[str] = None,
) -> ProcessedDocument:
    """
    Unified entry point for document parsing.
    Dispatches to format-specific parser based on file extension.
    """
    _, ext = os.path.splitext(filename.lower())

    if not ext:
        raise AppException(
            message=f"Filename '{filename}' is missing a file extension.",
            status_code=400,
            code="MISSING_FILE_EXTENSION",
        )

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise AppException(
            message=f"Unsupported file format '{ext}'. Supported formats: {supported}",
            status_code=400,
            code="UNSUPPORTED_FILE_TYPE",
        )

    file_type = SUPPORTED_EXTENSIONS[ext]

    if file_type == "pdf":
        return parse_pdf(file_bytes=file_bytes, filename=filename, doc_id=doc_id)
    elif file_type == "docx":
        return parse_docx(file_bytes=file_bytes, filename=filename, doc_id=doc_id)
    elif file_type == "txt":
        return parse_text_bytes(file_bytes=file_bytes, filename=filename, doc_id=doc_id)
    else:
        raise AppException(
            message=f"No parser available for format '{ext}'",
            status_code=500,
            code="NO_PARSER_CONFIGURED",
        )
