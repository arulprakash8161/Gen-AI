import pytest
from typing import List, Optional
from pydantic import ValidationError

from backend.app.models.document import DocumentChunk
from backend.app.services.vector_store.base import (
    BaseVectorStore,
    VectorChunkRecord,
    SimilaritySearchResult,
    VectorStoreStats,
)


def test_vector_chunk_record_validation():
    # Valid record
    record = VectorChunkRecord(
        chunk_id="doc_123_chunk_0",
        doc_id="doc_123",
        chunk_index=0,
        text="Sample chunk content for testing.",
        embedding=[0.1, 0.2, 0.3, 0.4],
        page_number=2,
        character_count=34,
        token_count=8,
        metadata={"filename": "report.pdf", "topic": "AI"},
    )
    assert record.chunk_id == "doc_123_chunk_0"
    assert record.doc_id == "doc_123"
    assert record.embedding == [0.1, 0.2, 0.3, 0.4]
    assert record.page_number == 2
    assert record.metadata["topic"] == "AI"

    # Empty embedding should fail
    with pytest.raises(ValidationError):
        VectorChunkRecord(
            chunk_id="doc_123_chunk_0",
            doc_id="doc_123",
            chunk_index=0,
            text="Valid text",
            embedding=[],
        )

    # Empty chunk_id should fail
    with pytest.raises(ValidationError):
        VectorChunkRecord(
            chunk_id="   ",
            doc_id="doc_123",
            chunk_index=0,
            text="Valid text",
            embedding=[0.1, 0.2],
        )

    # Empty doc_id should fail
    with pytest.raises(ValidationError):
        VectorChunkRecord(
            chunk_id="doc_123_chunk_0",
            doc_id="",
            chunk_index=0,
            text="Valid text",
            embedding=[0.1, 0.2],
        )


def test_vector_chunk_record_from_and_to_chunk():
    source_chunk = DocumentChunk(
        chunk_id="doc_abc_chunk_1",
        doc_id="doc_abc",
        chunk_index=1,
        text="Content of the second chunk.",
        page_number=5,
        character_count=28,
        token_count=7,
        metadata={"source": "manual.docx"},
    )
    embedding = [0.05, 0.95, -0.2]

    # Convert from DocumentChunk
    record = VectorChunkRecord.from_chunk(source_chunk, embedding=embedding)
    assert record.chunk_id == source_chunk.chunk_id
    assert record.doc_id == source_chunk.doc_id
    assert record.chunk_index == source_chunk.chunk_index
    assert record.text == source_chunk.text
    assert record.page_number == 5
    assert record.character_count == 28
    assert record.token_count == 7
    assert record.metadata == {"source": "manual.docx"}
    assert record.embedding == embedding

    # Convert back to DocumentChunk
    reconstructed = record.to_chunk()
    assert reconstructed.chunk_id == source_chunk.chunk_id
    assert reconstructed.doc_id == source_chunk.doc_id
    assert reconstructed.chunk_index == source_chunk.chunk_index
    assert reconstructed.text == source_chunk.text
    assert reconstructed.page_number == 5
    assert reconstructed.character_count == 28
    assert reconstructed.token_count == 7
    assert reconstructed.metadata == {"source": "manual.docx"}


def test_similarity_search_result_model():
    result = SimilaritySearchResult(
        chunk_id="doc_123_chunk_0",
        doc_id="doc_123",
        chunk_index=0,
        text="Relevant context extracted.",
        similarity_score=0.875,
        page_number=3,
        character_count=27,
        token_count=6,
        metadata={"filename": "whitepaper.pdf"},
    )
    assert result.chunk_id == "doc_123_chunk_0"
    assert result.similarity_score == 0.875
    assert result.page_number == 3
    assert result.metadata["filename"] == "whitepaper.pdf"
    assert result.embedding is None


def test_vector_store_stats_model():
    stats = VectorStoreStats(
        total_chunks=15,
        total_documents=3,
        document_ids=["doc_1", "doc_2", "doc_3"],
        dimension=384,
    )
    assert stats.total_chunks == 15
    assert stats.total_documents == 3
    assert len(stats.document_ids) == 3
    assert stats.dimension == 384


def test_base_vector_store_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseVectorStore()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_base_vector_store_subclass_contract():
    class DummyVectorStore(BaseVectorStore):
        def __init__(self):
            self.records: List[VectorChunkRecord] = []

        async def add_chunks(self, records: List[VectorChunkRecord]) -> int:
            self.records.extend(records)
            return len(records)

        async def similarity_search(
            self,
            query_vector: List[float],
            top_k: int = 5,
            score_threshold: Optional[float] = None,
            doc_id: Optional[str] = None,
        ) -> List[SimilaritySearchResult]:
            results = []
            for r in self.records:
                if doc_id and r.doc_id != doc_id:
                    continue
                results.append(
                    SimilaritySearchResult(
                        chunk_id=r.chunk_id,
                        doc_id=r.doc_id,
                        chunk_index=r.chunk_index,
                        text=r.text,
                        similarity_score=1.0,
                        page_number=r.page_number,
                        character_count=r.character_count,
                        token_count=r.token_count,
                        metadata=r.metadata,
                    )
                )
            return results[:top_k]

        async def delete_by_doc_id(self, doc_id: str) -> int:
            initial = len(self.records)
            self.records = [r for r in self.records if r.doc_id != doc_id]
            return initial - len(self.records)

        async def get_chunks_by_doc_id(self, doc_id: str) -> List[VectorChunkRecord]:
            return [r for r in self.records if r.doc_id == doc_id]

        async def get_stats(self) -> VectorStoreStats:
            docs = list(set(r.doc_id for r in self.records))
            return VectorStoreStats(
                total_chunks=len(self.records),
                total_documents=len(docs),
                document_ids=docs,
                dimension=len(self.records[0].embedding) if self.records else None,
            )

    store = DummyVectorStore()
    record = VectorChunkRecord(
        chunk_id="c1",
        doc_id="d1",
        chunk_index=0,
        text="Test content",
        embedding=[0.1, 0.2, 0.3],
    )
    added = await store.add_chunks([record])
    assert added == 1

    stats = await store.get_stats()
    assert stats.total_chunks == 1
    assert stats.total_documents == 1
    assert stats.dimension == 3

    search = await store.similarity_search([0.1, 0.2, 0.3])
    assert len(search) == 1
    assert search[0].chunk_id == "c1"

    deleted = await store.delete_by_doc_id("d1")
    assert deleted == 1
    stats_after = await store.get_stats()
    assert stats_after.total_chunks == 0
