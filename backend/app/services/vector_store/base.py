from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from backend.app.models.document import DocumentChunk


class VectorChunkRecord(BaseModel):
    """
    Complete vector representation of a document chunk, combining the text,
    source metadata, and dense embedding vector.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. doc_xxx_chunk_0")
    doc_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., ge=0, description="0-indexed position within parent document")
    text: str = Field(..., min_length=1, description="Cleaned chunk text content")
    embedding: List[float] = Field(..., description="Dense numerical embedding vector")
    page_number: Optional[int] = Field(None, description="Source page or section number if applicable")
    character_count: int = Field(default=0, ge=0, description="Number of characters in the chunk")
    token_count: int = Field(default=0, ge=0, description="Estimated token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata preserved from parent document")

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        if not v or len(v) == 0:
            raise ValueError("Embedding vector cannot be empty")
        return v

    @field_validator("chunk_id", "doc_id")
    @classmethod
    def validate_non_empty_ids(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Identifier cannot be empty or whitespace")
        return v.strip()

    @classmethod
    def from_chunk(cls, chunk: DocumentChunk, embedding: List[float]) -> "VectorChunkRecord":
        """
        Builds a VectorChunkRecord from an existing DocumentChunk and its calculated embedding.
        """
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            embedding=embedding,
            page_number=chunk.page_number,
            character_count=chunk.character_count,
            token_count=chunk.token_count,
            metadata=dict(chunk.metadata),
        )

    def to_chunk(self) -> DocumentChunk:
        """
        Converts this record back to a standard DocumentChunk.
        """
        return DocumentChunk(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            chunk_index=self.chunk_index,
            text=self.text,
            page_number=self.page_number,
            character_count=self.character_count,
            token_count=self.token_count,
            metadata=dict(self.metadata),
        )


class SimilaritySearchResult(BaseModel):
    """
    Search hit returned by a vector similarity query, containing chunk content,
    traceability metadata, and calculated cosine similarity score.
    """
    chunk_id: str = Field(..., description="Chunk identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., ge=0, description="Position within parent document")
    text: str = Field(..., description="Chunk text content")
    similarity_score: float = Field(..., description="Cosine similarity score, range [-1.0, 1.0]")
    page_number: Optional[int] = Field(None, description="Source page number")
    character_count: int = Field(default=0, description="Character count")
    token_count: int = Field(default=0, description="Estimated token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk and document metadata")
    embedding: Optional[List[float]] = Field(None, description="Optional raw embedding vector")


class VectorStoreStats(BaseModel):
    """
    High-level telemetry and metrics for an indexed vector store collection.
    """
    total_chunks: int = Field(default=0, ge=0, description="Total indexed chunk vectors")
    total_documents: int = Field(default=0, ge=0, description="Distinct indexed document count")
    document_ids: List[str] = Field(default_factory=list, description="List of indexed document IDs")
    dimension: Optional[int] = Field(None, description="Expected dimensionality of vectors")


class BaseVectorStore(ABC):
    """
    Abstract interface for persistent vector stores.
    Decouples storage implementation details (SQLite, ChromaDB, Qdrant, Milvus)
    from the application and RAG pipeline.
    """

    @abstractmethod
    async def add_chunks(self, records: List[VectorChunkRecord]) -> int:
        """
        Adds or upserts chunk records and their vector embeddings into the store.
        If a chunk_id already exists, it should update/replace the existing entry.

        Returns:
            int: The number of records successfully indexed or updated.
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        doc_id: Optional[str] = None,
    ) -> List[SimilaritySearchResult]:
        """
        Performs a k-NN similarity search using cosine similarity against stored vectors.

        Args:
            query_vector: Query embedding vector to compare against.
            top_k: Maximum number of closest matches to return.
            score_threshold: Minimum cosine similarity score required for inclusion.
            doc_id: Optional document ID to filter chunks by.

        Returns:
            List[SimilaritySearchResult]: Ranked search results in descending order of similarity.
        """
        pass

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Deletes all vector records belonging to the specified document ID.

        Args:
            doc_id: Unique identifier of the document to delete.

        Returns:
            int: Number of chunk records deleted.
        """
        pass

    @abstractmethod
    async def get_chunks_by_doc_id(self, doc_id: str) -> List[VectorChunkRecord]:
        """
        Retrieves all stored chunk records and metadata for a specific document ID.

        Args:
            doc_id: Unique identifier of the target document.

        Returns:
            List[VectorChunkRecord]: List of chunk records sorted by chunk_index.
        """
        pass

    @abstractmethod
    async def get_stats(self) -> VectorStoreStats:
        """
        Retrieves overall collection statistics (total chunks, unique documents, dimension).

        Returns:
            VectorStoreStats: Current storage statistics.
        """
        pass
