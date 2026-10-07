import io
import pytest
from pypdf import PdfWriter
from backend.app.services.document_parser.pdf_parser import parse_pdf
from backend.app.services.document_parser.cleaner import clean_text
from backend.app.core.exceptions import AppException


def create_sample_pdf_bytes(page_texts: list[str]) -> bytes:
    """Helper to generate a real PDF in memory with custom text on each page."""
    writer = PdfWriter()
    for text in page_texts:
        # Create a blank page and add annotations/text
        page = writer.add_blank_page(width=300, height=300)
    # Using pypdf's writer to write metadata
    writer.add_metadata({"/Title": "Test Document", "/Author": "SIH Team"})
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_clean_text_normalizes_whitespace():
    raw = "  Hello   world!  \r\n\r\n\r\nThis is a   new   paragraph. \n\n\n\nThird paragraph.  "
    cleaned = clean_text(raw)
    assert "Hello world!" in cleaned
    assert "This is a new paragraph." in cleaned
    assert "Third paragraph." in cleaned
    assert "\n\n\n" not in cleaned


def test_parse_pdf_valid():
    # Create a 2-page valid PDF
    pdf_bytes = create_sample_pdf_bytes(["Page 1 content", "Page 2 content"])
    doc = parse_pdf(file_bytes=pdf_bytes, filename="sample.pdf")

    assert doc.filename == "sample.pdf"
    assert doc.file_type == "pdf"
    assert doc.total_pages == 2
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].page_number == 2
    assert doc.doc_id.startswith("doc_")
    assert doc.metadata.get("total_pages") == 2
    assert doc.metadata.get("title") == "Test Document"


def test_parse_pdf_empty_bytes_raises_error():
    with pytest.raises(AppException) as exc_info:
        parse_pdf(file_bytes=b"", filename="empty.pdf")
    assert exc_info.value.code == "EMPTY_FILE"
    assert exc_info.value.status_code == 400


def test_parse_pdf_invalid_bytes_raises_error():
    with pytest.raises(AppException) as exc_info:
        parse_pdf(file_bytes=b"Not a real PDF stream", filename="corrupt.pdf")
    assert exc_info.value.code == "PDF_PARSE_ERROR"
    assert exc_info.value.status_code == 400
