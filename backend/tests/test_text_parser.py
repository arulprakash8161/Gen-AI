import pytest
from backend.app.services.document_parser.text_parser import parse_text, parse_text_bytes
from backend.app.services.document_parser.parser import parse_document
from backend.app.core.exceptions import AppException
from backend.tests.test_pdf_parser import create_sample_pdf_bytes


def test_parse_text_valid():
    content = "This is a plain text announcement.\n\nKey impact: High priority."
    doc = parse_text(content, filename="notice.txt")

    assert doc.filename == "notice.txt"
    assert doc.file_type == "txt"
    assert doc.total_pages == 1
    assert "This is a plain text announcement." in doc.text
    assert "Key impact: High priority." in doc.text


def test_parse_text_empty_raises_error():
    with pytest.raises(AppException) as exc_info:
        parse_text("   \n\n  ")
    assert exc_info.value.code == "EMPTY_CONTENT"
    assert exc_info.value.status_code == 400


def test_parse_text_bytes_latin1_fallback():
    # Encoded with latin-1 special character
    latin1_bytes = "Café report with currency: £100".encode("latin-1")
    doc = parse_text_bytes(latin1_bytes, filename="latin1.txt")
    assert "Café report" in doc.text
    assert "£100" in doc.text


def test_unified_parser_dispatches_correctly():
    # PDF dispatch
    pdf_bytes = create_sample_pdf_bytes(["Test Page"])
    pdf_doc = parse_document(pdf_bytes, filename="report.pdf")
    assert pdf_doc.file_type == "pdf"

    # TXT dispatch
    txt_doc = parse_document(b"Sample plain text content", filename="report.txt")
    assert txt_doc.file_type == "txt"

    # Markdown dispatch
    md_doc = parse_document(b"# Header\n\nBody content", filename="notes.md")
    assert md_doc.file_type == "txt"


def test_unified_parser_unsupported_extension():
    with pytest.raises(AppException) as exc_info:
        parse_document(b"some content", filename="script.exe")
    assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"
    assert exc_info.value.status_code == 400


def test_unified_parser_missing_extension():
    with pytest.raises(AppException) as exc_info:
        parse_document(b"some content", filename="no_extension_file")
    assert exc_info.value.code == "MISSING_FILE_EXTENSION"
    assert exc_info.value.status_code == 400
