import io
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.tests.test_pdf_parser import create_sample_pdf_bytes

client = TestClient(app)


def test_ingest_raw_text_api():
    payload = {
        "text": "Cybersecurity advisory for SIH 2024. Critical vulnerability discovered.",
        "title": "Advisory_01.txt"
    }
    response = client.post("/api/v1/documents/raw", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "Advisory_01.txt"
    assert data["file_type"] == "txt"
    assert "Cybersecurity advisory" in data["text"]
    assert "doc_id" in data


def test_upload_txt_file_api():
    file_content = b"Uploaded text report content."
    files = {"file": ("report.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "report.txt"
    assert data["file_type"] == "txt"
    assert "Uploaded text report content." in data["text"]


def test_upload_pdf_file_api():
    pdf_bytes = create_sample_pdf_bytes(["Page 1 of uploaded PDF", "Page 2"])
    files = {"file": ("manual.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "manual.pdf"
    assert data["file_type"] == "pdf"
    assert data["total_pages"] == 2


def test_upload_unsupported_file_api():
    files = {"file": ("archive.zip", io.BytesIO(b"PK...fakezip"), "application/zip")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_get_and_list_documents_api():
    # Ingest one document first
    res = client.post("/api/v1/documents/raw", json={"text": "Unique test doc for listing."})
    doc_id = res.json()["doc_id"]

    # List all documents
    list_res = client.get("/api/v1/documents")
    assert list_res.status_code == 200
    summaries = list_res.json()
    assert any(s["doc_id"] == doc_id for s in summaries)

    # Get specific document
    get_res = client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["doc_id"] == doc_id

    # Delete document
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200

    # Verify not found after delete
    get_after_del = client.get(f"/api/v1/documents/{doc_id}")
    assert get_after_del.status_code == 404
