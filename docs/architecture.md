# System Architecture — AI Content Transformation Platform

## 1. Executive Summary & Problem Context
The AI Content Transformation Platform is designed for the Smart India Hackathon (SIH) problem statement. The system ingests multi-format source data (raw text, PDF, DOCX, advisories, announcements) and accurately synthesizes and transforms it into tailored communication deliverables:
- **LinkedIn Posts** (with engaging hooks, structured body, CTAs, hashtags)
- **Executive Summaries** (concise overviews, findings, action items)
- **Official Advisories** (structured situation, impact, recommended actions)
- **Presentations** (slide-by-slide structure with speaker notes)
- **Video Packages** (full production scripts, scene descriptions, visual cues)

The system enforces strict **source grounding** via Retrieval-Augmented Generation (RAG) and local LLM inference (Ollama), ensuring verifiable facts and minimizing hallucinations.

---

## 2. High-Level System Architecture

```text
+-------------------------------------------------------------------------+
|                              React Frontend                             |
|  - Multi-file Ingestion  - Output Selector  - Persona & Tone Controls   |
|  - Live Status Tracker   - Deliverable Viewers  - Export Actions (PPTX) |
+------------------------------------+------------------------------------+
                                     | HTTP / JSON & Multipart
                                     v
+-------------------------------------------------------------------------+
|                          FastAPI Application Gateway                    |
|  - Request Validation    - Auth & Rate Limiting  - Error Handling       |
|  - Modular API Routers (/api/v1/health, /documents, /rag, /transform)   |
+----+-------------------+--------------------+-------------------+-------+
     |                   |                    |                   |
     v                   v                    v                   v
+------------+   +---------------+    +---------------+   +---------------+
| Document   |   | Semantic      |    | RAG Retrieval |   | Transformation|
| Ingestion  |   | Chunking      |    | & Context     |   | Engine        |
| - PDF      |-->| - Token/Char  |--->| - Vector DB   |-->| - Prompt Strat|
| - DOCX     |   | - Overlap     |    | - Metadata    |   | - LLM Client  |
| - Text     |   | - Source Meta |    |   Filtering   |   |   (Ollama)    |
+------------+   +---------------+    +---------------+   +-------+-------+
                                                                  |
                                                                  v
+-------------------------------------------------------------------------+
|                    Validation & Export Pipeline                         |
|  - Output Schema Validation  - Source Grounding & Claim Checker         |
|  - Export Generators (PDF, DOCX, PPTX, JSON)                            |
+-------------------------------------------------------------------------+
```

---

## 3. Folder Structure

```text
├── docs/
│   └── architecture.md            # System architecture document (this file)
├── backend/
│   ├── app/
│   │   ├── api/                   # API Route Layer
│   │   │   ├── v1/
│   │   │   │   ├── health.py      # Health & readiness probes
│   │   │   │   ├── documents.py   # Document upload, extraction & status
│   │   │   │   ├── rag.py         # RAG query, debug, and chunk inspection
│   │   │   │   ├── transform.py   # Content transformation endpoints
│   │   │   │   └── export.py      # Multi-format document export endpoints
│   │   │   └── router.py          # Central API v1 router definition
│   │   ├── core/                  # Infrastructure & Core Logic
│   │   │   ├── config.py          # Pydantic Settings & environment variables
│   │   │   ├── logging.py         # Structured logging configuration
│   │   │   └── exceptions.py      # Custom exceptions and exception handlers
│   │   ├── models/                # Pydantic Schemas & DTOs
│   │   │   ├── document.py        # Document, chunk, and metadata models
│   │   │   ├── request.py         # Transformation request schemas
│   │   │   └── deliverable.py     # Schemas for each deliverable type
│   │   ├── services/              # Core Business & AI Logic (Pluggable)
│   │   │   ├── document_parser/   # PDF, DOCX, and TXT extractors
│   │   │   ├── chunking/          # Recursive & semantic text chunkers
│   │   │   ├── embeddings/        # Pluggable embedding providers (local/Ollama)
│   │   │   ├── vector_store/      # Vector database abstraction (ChromaDB)
│   │   │   ├── rag/               # Context retrieval & prompt augmentation
│   │   │   ├── llm/               # Ollama client & structured completion service
│   │   │   ├── transformation/    # Central transformation orchestrator
│   │   │   ├── generators/        # Modular deliverable generators
│   │   │   │   ├── base.py
│   │   │   │   ├── linkedin.py
│   │   │   │   ├── summary.py
│   │   │   │   ├── advisory.py
│   │   │   │   ├── presentation.py
│   │   │   │   └── video_package.py
│   │   │   ├── validation/        # Output validation & source grounding checker
│   │   │   └── export/            # PPTX, PDF, and DOCX document exporters
│   │   └── main.py                # FastAPI ASGI application factory
│   ├── tests/                     # Unit, integration, and E2E tests
│   │   ├── unit/
│   │   ├── integration/
│   │   └── conftest.py
│   ├── requirements.txt           # Python backend dependencies
│   └── .env.example               # Template environment configuration
├── frontend/                      # React + Vite frontend dashboard
├── data/                          # Local persistent storage (vector store, temp files)
├── .gitignore
└── README.md
```

---

## 4. Component Responsibilities

| Component | Responsibility |
| :--- | :--- |
| **API Layer (`api/`)** | Handles HTTP requests, enforces request validation, error formatting, and maps domain responses to HTTP. |
| **Config & Core (`core/`)** | Centralizes environment variables, sets up logging, and enforces global exception handling. |
| **Document Parser (`services/document_parser/`)** | Ingests files (PDF via `pypdf`/`pdfplumber`, DOCX via `python-docx`, plain text), sanitizes whitespace, strips control characters, and returns normalized text with page numbers. |
| **Chunking (`services/chunking/`)** | Splits extracted text into semantic or recursive chunks with configurable size and overlap. Embeds chunk IDs, document IDs, and source page ranges. |
| **Embeddings Service (`services/embeddings/`)** | Generates vector representations using an abstract interface (default: Ollama local embeddings or HuggingFace sentence-transformers). |
| **Vector Database (`services/vector_store/`)** | Local persistent vector storage (ChromaDB) allowing collection creation, chunk insertion, metadata filtering, and k-NN similarity search. |
| **RAG Pipeline (`services/rag/`)** | Takes user transformation parameters, retrieves top-$k$ relevant chunks, deduplicates context, and constructs an augmented prompt payload. |
| **LLM Service (`services/llm/`)** | Communicates with the local Ollama instance (e.g. `llama3.2`, `mistral`, or `qwen2.5`). Supports streaming, timeouts, retries, and structured JSON output. |
| **Transformation Engine (`services/transformation/`)** | Coordinates context with specific deliverable generators based on user requirements (Audience, Tone, Detail Level, Objective). |
| **Output Generators (`services/generators/`)** | Specialized generators that enforce domain-specific prompt engineering and schemas for LinkedIn, Summaries, Advisories, Slides, and Video scripts. |
| **Validation Service (`services/validation/`)** | Validates generated deliverables against required Pydantic schemas and evaluates source grounding against retrieved source chunks to flag unsupported claims. |
| **Export Service (`services/export/`)** | Converts structured deliverable data into distributable formats (`.pptx` via `python-pptx`, `.pdf`, `.docx`). |

---

## 5. End-to-End Data Flow

```text
[Raw File: PDF/DOCX/TXT]
          │
          ▼
   1. Document Parser (extract clean text + page metadata)
          │
          ▼
   2. Chunking Engine (split text: 500-1000 tokens, 100 token overlap)
          │
          ▼
   3. Embedding Generator (vectorize chunks)
          │
          ▼
   4. Vector Database (persist chunks + embeddings + metadata)
          │
          ▼
[User Request: Target Deliverables + Audience + Tone + Objective]
          │
          ▼
   5. Vector Search (retrieve top-k relevant chunks based on query/document focus)
          │
          ▼
   6. RAG Context Builder (assemble grounding context + instructions)
          │
          ▼
   7. Ollama LLM Service (structured generation)
          │
          ▼
   8. Validation Layer (schema check + ground claim verification)
          │
          ▼
   9. Response Delivery (JSON to React Dashboard / Export to PPTX/PDF/DOCX)
```

---

## 6. API Architecture

All endpoints follow RESTful standards under `/api/v1`:

### Health
- `GET /api/v1/health` — Service health probe and version info
- `GET /api/v1/health/ready` — Verifies dependent service availability (Ollama, Vector DB)

### Document Management
- `POST /api/v1/documents/upload` — Ingests single or multi-part documents
- `GET /api/v1/documents` — Lists ingested documents
- `GET /api/v1/documents/{doc_id}` — Retrieves document extraction details
- `DELETE /api/v1/documents/{doc_id}` — Removes document and its vector embeddings

### RAG & Inspection
- `POST /api/v1/rag/query` — Test & inspect similarity search chunks with similarity scores
- `GET /api/v1/rag/chunks/{doc_id}` — Lists stored chunks and metadata for inspection

### Transformation
- `POST /api/v1/transform` — Core transformation endpoint. Accepts `document_id`, `output_types`, and customization parameters (`audience`, `tone`, `language`, `detail_level`, `objective`). Returns structured deliverables.

### Export
- `POST /api/v1/export/pptx` — Generates and downloads `.pptx` presentation file
- `POST /api/v1/export/docx` — Generates and downloads `.docx` report file
- `POST /api/v1/export/pdf` — Generates and downloads `.pdf` file

---

## 7. RAG Architecture

1. **Deterministic Document Slicing:**
   - Default chunk size: 600 characters / ~150 tokens (configurable)
   - Overlap: 100 characters / ~25 tokens
   - Metadata retained per chunk: `doc_id`, `filename`, `chunk_index`, `page_number`, `token_count`.
2. **Vector Space:**
   - Vector Store: **ChromaDB** in persistent directory mode (`data/chroma_db`).
   - Distance Metric: Cosine similarity (`cosine`).
3. **Retrieval Mechanics:**
   - Standard retrieval: Top-$k$ nearest neighbors ($k \in [3, 8]$ depending on detail level).
   - Document scoping: Filter by `doc_id` or query across a collection of documents.
4. **Context Construction:**
   - Clear context delimiters:
     ```text
     [SOURCE CHUNK 1 - Page 2]
     ... content ...
     [SOURCE CHUNK 2 - Page 3]
     ... content ...
     ```
   - System prompts explicitly instruct the model to ground assertions exclusively in the provided chunks.

---

## 8. Configuration & Environment Strategy

Configuration follows the 12-factor app methodology using `pydantic-settings`:
- Configuration is declared in `backend/app/core/config.py`.
- Defaults are safe for local development with zero external cloud dependencies.
- Config keys are loaded from `.env` with fallback to system environment variables.

### Key Environment Variables:
```ini
# Application
PROJECT_NAME="AI Content Transformation Platform"
API_V1_PREFIX="/api/v1"
DEBUG=True
ENVIRONMENT=development

# Storage Paths
DATA_DIR="data"
UPLOADS_DIR="data/uploads"
CHROMA_PERSIST_DIR="data/chroma_db"

# Ollama LLM Settings
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="llama3.2"
OLLAMA_TIMEOUT_SECONDS=120

# Embedding Settings
EMBEDDING_PROVIDER="ollama"  # "ollama" or "sentence-transformers"
EMBEDDING_MODEL="nomic-embed-text"

# Chunking Defaults
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K_CHUNKS=5
```

---

## 9. Database & Persistence Strategy

- **Vector Database:** Local **ChromaDB** stored in `data/chroma_db/`. Fully embedded, zero external daemon requirement, persistent across process restarts.
- **Document Metadata:** Embedded directly in ChromaDB chunk metadata (or a lightweight SQLite DB in `data/metadata.db` if complex relational queries are needed later).
- **Decoupled Interface:** All vector operations will go through a `BaseVectorStore` abstract class (`add_documents`, `similarity_search`, `delete_collection`), making swapping to Qdrant or Milvus seamless in future phases.

---

## 10. Testing Strategy

1. **Unit Testing (`backend/tests/unit/`):**
   - Document parsers tested with fixture files (.pdf, .docx, .txt).
   - Chunking algorithms tested for boundary preservation, metadata completeness, and overlap correctness.
   - Pydantic schema validation for request/response payloads.
2. **Integration Testing (`backend/tests/integration/`):**
   - FastAPI endpoint testing using `httpx.AsyncClient`.
   - Vector store insertion and retrieval fidelity.
   - Ollama client connectivity and error handling with mock responses.
3. **End-to-End Workflow Testing:**
   - Ingest document → Chunk → Store → Retrieve → Generate → Validate → Export.
   - Grounding validation scoring against synthetic hallucinations.
