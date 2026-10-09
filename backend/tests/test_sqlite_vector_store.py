import pytest
from backend.app.services.vector_store.sqlite_store import (
    SQLiteVectorStore,
    calculate_cosine_similarity,
)
from backend.app.services.vector_store.base import VectorChunkRecord
from backend.app.services.vector_store.factory import get_vector_store


def test_cosine_similarity_math():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v4 = [-1.0, 0.0, 0.0]

    assert pytest.approx(calculate_cosine_similarity(v1, v2), 0.001) == 1.0
    assert pytest.approx(calculate_cosine_similarity(v1, v3), 0.001) == 0.0
    assert pytest.approx(calculate_cosine_similarity(v1, v4), 0.001) == -1.0

    # Zero magnitude or dimension mismatch
    assert calculate_cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert calculate_cosine_similarity([], [1.0]) == 0.0
    assert calculate_cosine_similarity([1.0, 2.0], [1.0]) == 0.0


@pytest.mark.asyncio
async def test_sqlite_vector_store_persistence_across_restarts(tmp_path):
    db_file = str(tmp_path / "test_persistence.db")

    # Instance 1: Add records
    store1 = SQLiteVectorStore(db_path=db_file)
    records = [
        VectorChunkRecord(
            chunk_id="chunk_1",
            doc_id="doc_alpha",
            chunk_index=0,
            text="First chunk of alpha document.",
            embedding=[0.6, 0.8],
            page_number=1,
            character_count=30,
            token_count=6,
            metadata={"source": "alpha.pdf", "category": "engineering"},
        ),
        VectorChunkRecord(
            chunk_id="chunk_2",
            doc_id="doc_alpha",
            chunk_index=1,
            text="Second chunk of alpha document.",
            embedding=[0.8, 0.6],
            page_number=2,
            character_count=31,
            token_count=6,
            metadata={"source": "alpha.pdf", "category": "engineering"},
        ),
    ]
    added = await store1.add_chunks(records)
    assert added == 2

    stats1 = await store1.get_stats()
    assert stats1.total_chunks == 2
    assert stats1.total_documents == 1
    assert stats1.document_ids == ["doc_alpha"]
    assert stats1.dimension == 2

    # Simulate process restart by creating a new instance pointing to the same file
    store2 = SQLiteVectorStore(db_path=db_file)
    stats2 = await store2.get_stats()
    assert stats2.total_chunks == 2
    assert stats2.total_documents == 1
    assert stats2.document_ids == ["doc_alpha"]
    assert stats2.dimension == 2

    stored_chunks = await store2.get_chunks_by_doc_id("doc_alpha")
    assert len(stored_chunks) == 2
    assert stored_chunks[0].chunk_id == "chunk_1"
    assert stored_chunks[0].page_number == 1
    assert stored_chunks[0].metadata["category"] == "engineering"
    assert stored_chunks[0].embedding == [0.6, 0.8]


@pytest.mark.asyncio
async def test_sqlite_vector_store_cosine_ranking_and_threshold(tmp_path):
    db_file = str(tmp_path / "test_ranking.db")
    store = SQLiteVectorStore(db_path=db_file)

    records = [
        VectorChunkRecord(
            chunk_id="c_exact",
            doc_id="doc_rank",
            chunk_index=0,
            text="Exact match vector.",
            embedding=[1.0, 0.0, 0.0],
            page_number=1,
        ),
        VectorChunkRecord(
            chunk_id="c_close",
            doc_id="doc_rank",
            chunk_index=1,
            text="Close vector.",
            embedding=[0.8, 0.6, 0.0],
            page_number=2,
        ),
        VectorChunkRecord(
            chunk_id="c_orthogonal",
            doc_id="doc_rank",
            chunk_index=2,
            text="Orthogonal vector.",
            embedding=[0.0, 1.0, 0.0],
            page_number=3,
        ),
    ]
    await store.add_chunks(records)

    query = [1.0, 0.0, 0.0]
    hits = await store.similarity_search(query_vector=query, top_k=3)
    assert len(hits) == 3
    assert hits[0].chunk_id == "c_exact"
    assert pytest.approx(hits[0].similarity_score, 0.01) == 1.0
    assert hits[1].chunk_id == "c_close"
    assert pytest.approx(hits[1].similarity_score, 0.01) == 0.8
    assert hits[2].chunk_id == "c_orthogonal"
    assert pytest.approx(hits[2].similarity_score, 0.01) == 0.0

    # With score_threshold=0.5
    filtered_hits = await store.similarity_search(query_vector=query, score_threshold=0.5)
    assert len(filtered_hits) == 2
    assert [h.chunk_id for h in filtered_hits] == ["c_exact", "c_close"]


@pytest.mark.asyncio
async def test_sqlite_vector_store_doc_id_filtering(tmp_path):
    db_file = str(tmp_path / "test_filtering.db")
    store = SQLiteVectorStore(db_path=db_file)

    records = [
        VectorChunkRecord(
            chunk_id="c_doc1",
            doc_id="doc_1",
            chunk_index=0,
            text="Document 1 content.",
            embedding=[1.0, 0.0],
        ),
        VectorChunkRecord(
            chunk_id="c_doc2",
            doc_id="doc_2",
            chunk_index=0,
            text="Document 2 content.",
            embedding=[1.0, 0.0],
        ),
    ]
    await store.add_chunks(records)

    # Search filtered by doc_1
    hits_doc1 = await store.similarity_search(query_vector=[1.0, 0.0], doc_id="doc_1")
    assert len(hits_doc1) == 1
    assert hits_doc1[0].chunk_id == "c_doc1"
    assert hits_doc1[0].doc_id == "doc_1"

    # Search filtered by doc_2
    hits_doc2 = await store.similarity_search(query_vector=[1.0, 0.0], doc_id="doc_2")
    assert len(hits_doc2) == 1
    assert hits_doc2[0].chunk_id == "c_doc2"
    assert hits_doc2[0].doc_id == "doc_2"


@pytest.mark.asyncio
async def test_sqlite_vector_store_deduplication_on_reindexing(tmp_path):
    db_file = str(tmp_path / "test_dedup.db")
    store = SQLiteVectorStore(db_path=db_file)

    record = VectorChunkRecord(
        chunk_id="unique_chunk_id",
        doc_id="doc_x",
        chunk_index=0,
        text="Original text.",
        embedding=[0.5, 0.5],
        page_number=1,
    )
    await store.add_chunks([record])

    stats = await store.get_stats()
    assert stats.total_chunks == 1

    # Re-index with same chunk_id but updated content
    updated_record = VectorChunkRecord(
        chunk_id="unique_chunk_id",
        doc_id="doc_x",
        chunk_index=0,
        text="Updated text after re-extraction.",
        embedding=[0.9, 0.1],
        page_number=1,
    )
    await store.add_chunks([updated_record])

    # Total count must still be 1, not 2
    stats_after = await store.get_stats()
    assert stats_after.total_chunks == 1

    chunks = await store.get_chunks_by_doc_id("doc_x")
    assert len(chunks) == 1
    assert chunks[0].text == "Updated text after re-extraction."
    assert chunks[0].embedding == [0.9, 0.1]


@pytest.mark.asyncio
async def test_sqlite_vector_store_deletion_isolation(tmp_path):
    db_file = str(tmp_path / "test_deletion.db")
    store = SQLiteVectorStore(db_path=db_file)

    records = [
        VectorChunkRecord(
            chunk_id="c_keep_1",
            doc_id="doc_keep",
            chunk_index=0,
            text="Retained doc chunk 1.",
            embedding=[0.1, 0.2],
        ),
        VectorChunkRecord(
            chunk_id="c_keep_2",
            doc_id="doc_keep",
            chunk_index=1,
            text="Retained doc chunk 2.",
            embedding=[0.2, 0.3],
        ),
        VectorChunkRecord(
            chunk_id="c_del_1",
            doc_id="doc_delete",
            chunk_index=0,
            text="Target for deletion.",
            embedding=[0.4, 0.5],
        ),
    ]
    await store.add_chunks(records)

    assert (await store.get_stats()).total_chunks == 3

    # Delete only doc_delete
    deleted_count = await store.delete_by_doc_id("doc_delete")
    assert deleted_count == 1

    stats = await store.get_stats()
    assert stats.total_chunks == 2
    assert stats.total_documents == 1
    assert stats.document_ids == ["doc_keep"]

    # Verify doc_delete is gone
    deleted_chunks = await store.get_chunks_by_doc_id("doc_delete")
    assert len(deleted_chunks) == 0

    # Verify doc_keep is completely intact
    kept_chunks = await store.get_chunks_by_doc_id("doc_keep")
    assert len(kept_chunks) == 2
    assert [c.chunk_id for c in kept_chunks] == ["c_keep_1", "c_keep_2"]


@pytest.mark.asyncio
async def test_sqlite_vector_store_edge_cases_and_invalid_inputs(tmp_path):
    db_file = str(tmp_path / "test_edge_cases.db")
    store = SQLiteVectorStore(db_path=db_file)

    # Empty chunk list
    assert await store.add_chunks([]) == 0

    # Empty query vector
    assert await store.similarity_search([]) == []

    # All-zero query vector
    assert await store.similarity_search([0.0, 0.0, 0.0]) == []

    # Non-existent doc deletion
    assert await store.delete_by_doc_id("non_existent_doc") == 0
    assert await store.delete_by_doc_id("") == 0

    # Non-existent doc chunks
    assert await store.get_chunks_by_doc_id("unknown") == []


@pytest.mark.asyncio
async def test_vector_store_factory(tmp_path):
    custom_db = str(tmp_path / "factory_store.db")
    store = get_vector_store(store_type="sqlite", db_path=custom_db, force_new=True)
    assert isinstance(store, SQLiteVectorStore)
    assert store.db_path == custom_db
