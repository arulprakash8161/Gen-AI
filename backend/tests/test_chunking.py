import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models.document import ProcessedDocument, PageContent, ChunkingConfig
from backend.app.services.chunking.text_chunker import chunk_document, recursive_split
from backend.app.core.exceptions import AppException

client = TestClient(app)


def test_chunk_size_enforcement():
    # Long text with paragraphs and sentences
    paragraphs = [
        "Paragraph 1: AI-powered Content Transformation Platform designed for SIH problem statement. " * 3,
        "Paragraph 2: Grounded in source documents using RAG and local Ollama inference models. " * 3,
        "Paragraph 3: Output formats include LinkedIn posts, executive summaries, advisories, and presentations. " * 3,
    ]
    full_text = "\n\n".join(paragraphs)

    doc = ProcessedDocument(
        doc_id="doc_test_100",
        filename="test_report.txt",
        file_type="txt",
        total_pages=1,
        text=full_text,
        character_count=len(full_text),
    )

    config = ChunkingConfig(chunk_size=300, chunk_overlap=50)
    chunks = chunk_document(doc, config=config)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.character_count <= 300
        assert chunk.doc_id == "doc_test_100"
        assert chunk.chunk_id.startswith("doc_test_100_c")
        assert chunk.token_count > 0


def test_chunk_overlap_preservation():
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta. "
        "Iota kappa lambda mu nu xi omicron pi rho. "
        "Sigma tau upsilon phi chi psi omega end."
    )
    doc = ProcessedDocument(
        doc_id="doc_overlap_test",
        filename="alphabet.txt",
        file_type="txt",
        total_pages=1,
        text=text,
        character_count=len(text),
    )

    config = ChunkingConfig(chunk_size=70, chunk_overlap=25)
    chunks = chunk_document(doc, config=config)

    assert len(chunks) >= 2
    # Check that at least some tokens from chunk N appear in chunk N+1
    for i in range(len(chunks) - 1):
        words_c1 = set(chunks[i].text.split())
        words_c2 = set(chunks[i + 1].text.split())
        overlap_words = words_c1.intersection(words_c2)
        assert len(overlap_words) > 0, f"Expected overlapping words between chunk {i} and {i+1}"


def test_metadata_and_doc_id_preservation():
    doc = ProcessedDocument(
        doc_id="doc_meta_55",
        filename="policy_advisory.pdf",
        file_type="pdf",
        total_pages=1,
        text="Critical infrastructure update advisory details.",
        character_count=48,
        metadata={"classification": "confidential", "agency": "CERT"},
    )

    chunks = chunk_document(doc)
    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.doc_id == "doc_meta_55"
    assert chunk.chunk_index == 0
    assert chunk.metadata["classification"] == "confidential"
    assert chunk.metadata["agency"] == "CERT"
    assert chunk.metadata["filename"] == "policy_advisory.pdf"
    assert chunk.metadata["file_type"] == "pdf"


def test_page_source_preservation_multipage():
    page1 = PageContent(page_number=1, text="Page 1: Overview of the problem and background information.")
    page2 = PageContent(page_number=2, text="Page 2: Proposed architecture and technical specifications.")

    doc = ProcessedDocument(
        doc_id="doc_multipage_99",
        filename="specification.pdf",
        file_type="pdf",
        total_pages=2,
        text=f"{page1.text}\n\n{page2.text}",
        character_count=len(page1.text) + len(page2.text),
        pages=[page1, page2],
    )

    chunks = chunk_document(doc)
    assert len(chunks) == 2

    assert chunks[0].page_number == 1
    assert chunks[0].metadata["source_page"] == 1
    assert "Page 1" in chunks[0].text

    assert chunks[1].page_number == 2
    assert chunks[1].metadata["source_page"] == 2
    assert "Page 2" in chunks[1].text


def test_short_document_single_chunk():
    doc = ProcessedDocument(
        doc_id="doc_short",
        filename="short.txt",
        file_type="txt",
        total_pages=1,
        text="Brief one-line statement.",
        character_count=25,
    )

    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Brief one-line statement."
    assert chunks[0].chunk_index == 0


def test_invalid_overlap_raises_error():
    doc = ProcessedDocument(
        doc_id="doc_err",
        filename="err.txt",
        file_type="txt",
        total_pages=1,
        text="Sample text content.",
        character_count=20,
    )
    invalid_config = ChunkingConfig(chunk_size=200, chunk_overlap=250)
    with pytest.raises(AppException) as exc_info:
        chunk_document(doc, config=invalid_config)
    assert exc_info.value.code == "INVALID_CHUNKING_CONFIG"
    assert exc_info.value.status_code == 400


def test_empty_document_raises_error():
    doc = ProcessedDocument(
        doc_id="doc_empty",
        filename="empty.txt",
        file_type="txt",
        total_pages=0,
        text="",
        character_count=0,
    )
    with pytest.raises(AppException) as exc_info:
        chunk_document(doc)
    assert exc_info.value.code == "EMPTY_DOCUMENT_CHUNK"


def test_api_get_document_chunks():
    # Ingest document first
    res = client.post("/api/v1/documents/raw", json={
        "text": "First chunk sentence. Second chunk sentence. Third chunk sentence.",
        "title": "chunk_api_test.txt"
    })
    doc_id = res.json()["doc_id"]

    # Request chunks via endpoint with custom parameters
    chunks_res = client.get(f"/api/v1/documents/{doc_id}/chunks?chunk_size=35&chunk_overlap=10")
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    assert len(chunks) > 1
    for c in chunks:
        assert c["doc_id"] == doc_id
        assert "chunk_id" in c
        assert "text" in c
        assert "token_count" in c
