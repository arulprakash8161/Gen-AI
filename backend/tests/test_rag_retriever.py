import pytest

from backend.app.services.rag.retriever import (
    RAGRetriever,
    format_citation_header,
    RAGRetrievalResult,
    get_rag_retriever,
)
from backend.app.services.vector_store.sqlite_store import SQLiteVectorStore
from backend.app.services.vector_store.base import VectorChunkRecord
from backend.app.services.embeddings.mock_provider import DeterministicLocalEmbeddingProvider
from backend.app.core.exceptions import AppException


def test_citation_header_formatting():
    # With valid page number
    cit_with_page = format_citation_header(
        rank_index=1,
        document_name="architecture.pdf",
        page_number=4,
        chunk_id="doc_arch_chunk_3",
    )
    assert cit_with_page == "[SOURCE CHUNK 1 — Document: architecture.pdf — Page: 4 — Chunk ID: doc_arch_chunk_3]"

    # Without page number (e.g. raw text or txt file)
    cit_no_page = format_citation_header(
        rank_index=2,
        document_name="advisory.txt",
        page_number=None,
        chunk_id="doc_adv_chunk_0",
    )
    assert cit_no_page == "[SOURCE CHUNK 2 — Document: advisory.txt — Page: N/A — Chunk ID: doc_adv_chunk_0]"

    # Empty document name fallback
    cit_fallback = format_citation_header(
        rank_index=3,
        document_name="",
        page_number=1,
        chunk_id="chunk_x",
    )
    assert cit_fallback == "[SOURCE CHUNK 3 — Document: Unknown Document — Page: 1 — Chunk ID: chunk_x]"


@pytest.mark.asyncio
async def test_rag_retrieval_ranking_and_formatted_context(tmp_path):
    test_db = str(tmp_path / "test_rag.db")
    store = SQLiteVectorStore(db_path=test_db)
    provider = DeterministicLocalEmbeddingProvider(dimension=64)

    # Pre-populate store with chunks
    emb_ai = await provider.embed_query("artificial intelligence RAG retrieval models")
    emb_sec = await provider.embed_query("cybersecurity vulnerability incident patch")
    emb_cook = await provider.embed_query("chocolate cake bakery recipe chocolate")

    records = [
        VectorChunkRecord(
            chunk_id="c_ai_1",
            doc_id="doc_ai",
            chunk_index=0,
            text="RAG systems combine vector retrieval with local neural language models.",
            embedding=emb_ai,
            page_number=2,
            metadata={"filename": "rag_overview.pdf"},
        ),
        VectorChunkRecord(
            chunk_id="c_sec_1",
            doc_id="doc_sec",
            chunk_index=0,
            text="Critical security vulnerability discovered in authentication gateway.",
            embedding=emb_sec,
            page_number=1,
            metadata={"filename": "security_advisory.docx"},
        ),
        VectorChunkRecord(
            chunk_id="c_cook_1",
            doc_id="doc_cook",
            chunk_index=0,
            text="Preheat the oven to 350 degrees before mixing dark chocolate and flour.",
            embedding=emb_cook,
            page_number=None,
            metadata={"filename": "recipes.txt"},
        ),
    ]
    await store.add_chunks(records)

    retriever = RAGRetriever(embedding_provider=provider, vector_store=store)

    # Query for AI and RAG
    result: RAGRetrievalResult = await retriever.retrieve(
        query="retrieval augmented generation AI models",
        top_k=2,
    )

    assert result.query == "retrieval augmented generation AI models"
    assert len(result.retrieved_chunks) <= 2
    assert result.total_chunks_found == len(result.retrieved_chunks)
    assert result.retrieved_chunks[0].chunk_id == "c_ai_1"
    assert result.retrieved_chunks[0].page_number == 2
    assert result.retrieved_chunks[0].document_name == "rag_overview.pdf"

    # Context formatting verification
    assert "[SOURCE CHUNK 1 — Document: rag_overview.pdf — Page: 2 — Chunk ID: c_ai_1]" in result.formatted_context
    assert "RAG systems combine vector retrieval" in result.formatted_context


@pytest.mark.asyncio
async def test_rag_retrieval_doc_id_filter(tmp_path):
    test_db = str(tmp_path / "test_rag_filter.db")
    store = SQLiteVectorStore(db_path=test_db)
    provider = DeterministicLocalEmbeddingProvider(dimension=64)

    emb = await provider.embed_query("Common information present across multiple documents")
    records = [
        VectorChunkRecord(
            chunk_id="doc1_c0",
            doc_id="doc_1",
            chunk_index=0,
            text="Document 1 exclusive chunk.",
            embedding=emb,
            page_number=1,
            metadata={"filename": "doc1.pdf"},
        ),
        VectorChunkRecord(
            chunk_id="doc2_c0",
            doc_id="doc_2",
            chunk_index=0,
            text="Document 2 exclusive chunk.",
            embedding=emb,
            page_number=1,
            metadata={"filename": "doc2.pdf"},
        ),
    ]
    await store.add_chunks(records)

    retriever = RAGRetriever(embedding_provider=provider, vector_store=store)

    res_doc1 = await retriever.retrieve(query="exclusive chunk", doc_id="doc_1")
    assert len(res_doc1.retrieved_chunks) == 1
    assert res_doc1.retrieved_chunks[0].doc_id == "doc_1"
    assert res_doc1.doc_id_filter == "doc_1"

    res_doc2 = await retriever.retrieve(query="exclusive chunk", doc_id="doc_2")
    assert len(res_doc2.retrieved_chunks) == 1
    assert res_doc2.retrieved_chunks[0].doc_id == "doc_2"
    assert res_doc2.doc_id_filter == "doc_2"


@pytest.mark.asyncio
async def test_rag_retrieval_score_threshold(tmp_path):
    test_db = str(tmp_path / "test_rag_threshold.db")
    store = SQLiteVectorStore(db_path=test_db)
    provider = DeterministicLocalEmbeddingProvider(dimension=64)

    emb_target = await provider.embed_query("quantum computing qubits superposition")
    emb_distant = await provider.embed_query("baking cookies oven flour chocolate")

    records = [
        VectorChunkRecord(
            chunk_id="q1",
            doc_id="doc_q",
            chunk_index=0,
            text="Quantum computing principles.",
            embedding=emb_target,
        ),
        VectorChunkRecord(
            chunk_id="b1",
            doc_id="doc_b",
            chunk_index=0,
            text="Baking recipes.",
            embedding=emb_distant,
        ),
    ]
    await store.add_chunks(records)

    retriever = RAGRetriever(embedding_provider=provider, vector_store=store)

    # Search with high similarity threshold
    res = await retriever.retrieve(
        query="quantum computing qubits",
        score_threshold=0.8,
    )
    assert len(res.retrieved_chunks) == 1
    assert res.retrieved_chunks[0].chunk_id == "q1"


@pytest.mark.asyncio
async def test_rag_retrieval_deduplication(tmp_path):
    test_db = str(tmp_path / "test_rag_dedup.db")
    store = SQLiteVectorStore(db_path=test_db)
    provider = DeterministicLocalEmbeddingProvider(dimension=64)

    emb = await provider.embed_query("identical chunk text repeated")
    records = [
        VectorChunkRecord(
            chunk_id="chunk_a",
            doc_id="doc_rep",
            chunk_index=0,
            text="Identical repeated chunk content.",
            embedding=emb,
        ),
        VectorChunkRecord(
            chunk_id="chunk_b",
            doc_id="doc_rep",
            chunk_index=1,
            text="Identical repeated chunk content.",  # Exactly duplicate text
            embedding=emb,
        ),
    ]
    await store.add_chunks(records)

    retriever = RAGRetriever(embedding_provider=provider, vector_store=store)

    # Deduplication enabled (default)
    res_dedup = await retriever.retrieve(query="identical repeated chunk", deduplicate=True)
    assert len(res_dedup.retrieved_chunks) == 1

    # Deduplication disabled
    res_no_dedup = await retriever.retrieve(query="identical repeated chunk", deduplicate=False)
    assert len(res_no_dedup.retrieved_chunks) == 2


@pytest.mark.asyncio
async def test_rag_retrieval_empty_query_raises_error():
    retriever = RAGRetriever(
        embedding_provider=DeterministicLocalEmbeddingProvider(),
        vector_store=SQLiteVectorStore(db_path=":memory:"),
    )

    with pytest.raises(AppException) as exc_info:
        await retriever.retrieve(query="   ")
    assert exc_info.value.code == "EMPTY_QUERY_ERROR"


@pytest.mark.asyncio
async def test_rag_retrieval_embedding_failure():
    class FailingProvider:
        async def embed_query(self, query):
            raise RuntimeError("Embedding network timeout")

    retriever = RAGRetriever(
        embedding_provider=FailingProvider(),  # type: ignore[arg-type]
        vector_store=SQLiteVectorStore(db_path=":memory:"),
    )

    with pytest.raises(AppException) as exc_info:
        await retriever.retrieve(query="valid query")
    assert exc_info.value.code == "QUERY_EMBEDDING_FAILED"


def test_get_rag_retriever_factory():
    retriever = get_rag_retriever(force_new=True)
    assert isinstance(retriever, RAGRetriever)
