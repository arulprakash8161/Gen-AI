from typing import Optional, List
from pydantic import BaseModel, Field

from backend.app.models.document import ProcessedDocument, ChunkingConfig
from backend.app.services.chunking.text_chunker import chunk_document
from backend.app.services.embeddings.base import BaseEmbeddingProvider
from backend.app.services.embeddings.factory import get_embedding_provider
from backend.app.services.vector_store.base import BaseVectorStore, VectorChunkRecord
from backend.app.services.vector_store.factory import get_vector_store
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class IndexingResult(BaseModel):
    """
    Metadata summary returned upon successful document indexing.
    """
    doc_id: str = Field(..., description="Indexed document ID")
    filename: str = Field(..., description="Document filename")
    chunks_indexed: int = Field(..., ge=0, description="Number of chunks stored in vector store")
    dimension: int = Field(..., description="Embedding vector dimension")
    embedding_model: str = Field(..., description="Embedding model name used")
    status: str = Field("indexed", description="Indexing status")


async def index_document(
    document: ProcessedDocument,
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    vector_store: Optional[BaseVectorStore] = None,
    chunking_config: Optional[ChunkingConfig] = None,
) -> IndexingResult:
    """
    Executes the full automated indexing flow:
    Document -> Chunking -> Embedding Generation -> Persistent Vector Indexing.

    Explicitly fails and raises AppException if chunking, embedding, or storage fails.
    Never reports success when indexing was not fully persisted.
    """
    if not document.text or not document.text.strip():
        raise AppException(
            message=f"Cannot index document '{document.filename}' because its text content is empty.",
            status_code=400,
            code="EMPTY_DOCUMENT_ERROR",
        )

    # 1. Resolve dependencies
    provider = embedding_provider or get_embedding_provider()
    store = vector_store or get_vector_store()

    # 2. Text Chunking
    try:
        chunks = chunk_document(document, config=chunking_config)
    except Exception as exc:
        logger.error(f"Chunking failed for document '{document.doc_id}': {exc}")
        raise AppException(
            message=f"Failed to chunk document '{document.filename}': {str(exc)}",
            status_code=400,
            code="CHUNKING_FAILED",
            details=str(exc),
        )

    if not chunks:
        raise AppException(
            message=f"No valid text chunks generated for document '{document.filename}'.",
            status_code=400,
            code="NO_CHUNKS_GENERATED",
        )

    # 3. Embedding Generation
    chunk_texts = [c.text for c in chunks]
    try:
        embeddings = await provider.embed_documents(chunk_texts)
    except Exception as exc:
        logger.error(f"Embedding generation failed for document '{document.doc_id}': {exc}")
        raise AppException(
            message=f"Embedding generation failed for document '{document.filename}': {str(exc)}",
            status_code=502,
            code="EMBEDDING_FAILED",
            details=str(exc),
        )

    if len(embeddings) != len(chunks):
        raise AppException(
            message=(
                f"Embedding count mismatch for document '{document.filename}': "
                f"expected {len(chunks)}, received {len(embeddings)}"
            ),
            status_code=500,
            code="EMBEDDING_COUNT_MISMATCH",
        )

    # 4. Vector Chunk Records Assembly
    records: List[VectorChunkRecord] = [
        VectorChunkRecord.from_chunk(chunk=chunk, embedding=emb)
        for chunk, emb in zip(chunks, embeddings)
    ]

    # 5. Persistent Vector Indexing
    try:
        indexed_count = await store.add_chunks(records)
    except Exception as exc:
        logger.error(f"Vector storage indexing failed for document '{document.doc_id}': {exc}")
        raise AppException(
            message=f"Vector storage indexing failed for document '{document.filename}': {str(exc)}",
            status_code=500,
            code="VECTOR_INDEXING_FAILED",
            details=str(exc),
        )

    # 6. Update parent document metadata
    document.metadata["indexed"] = True
    document.metadata["chunks_count"] = indexed_count
    document.metadata["embedding_model"] = provider.model_name
    document.metadata["embedding_dimension"] = provider.dimension

    logger.info(
        f"Document '{document.filename}' ({document.doc_id}) successfully indexed: "
        f"{indexed_count} chunks persisted with model '{provider.model_name}' (dim {provider.dimension})"
    )

    return IndexingResult(
        doc_id=document.doc_id,
        filename=document.filename,
        chunks_indexed=indexed_count,
        dimension=provider.dimension,
        embedding_model=provider.model_name,
        status="indexed",
    )
