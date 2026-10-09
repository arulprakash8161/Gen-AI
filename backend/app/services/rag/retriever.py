from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.services.embeddings.base import BaseEmbeddingProvider
from backend.app.services.embeddings.factory import get_embedding_provider
from backend.app.services.vector_store.base import BaseVectorStore, SimilaritySearchResult
from backend.app.services.vector_store.factory import get_vector_store
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class RetrievedChunk(BaseModel):
    """
    Individual chunk retrieved during a RAG search, enhanced with citation metadata.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., ge=0, description="0-indexed position within parent document")
    text: str = Field(..., description="Raw text of the chunk")
    similarity_score: float = Field(..., description="Cosine similarity score [-1.0, 1.0]")
    page_number: Optional[int] = Field(None, description="Source page number, or None if unavailable")
    document_name: str = Field(..., description="Human-readable document name or filename")
    citation_header: str = Field(..., description="Structured citation header tag")
    character_count: int = Field(default=0, description="Chunk character count")
    token_count: int = Field(default=0, description="Estimated token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom document/chunk metadata")


class RAGRetrievalResult(BaseModel):
    """
    Aggregated response from the RAG retrieval pipeline, containing ranked chunks
    and assembled grounding context ready for prompting.
    """
    query: str = Field(..., description="Original user search query")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Ranked retrieved chunks")
    formatted_context: str = Field(..., description="Aggregated context with standardized source citations")
    total_chunks_found: int = Field(default=0, description="Number of matching chunks retrieved")
    doc_id_filter: Optional[str] = Field(None, description="Document ID filter applied, if any")


def format_citation_header(
    rank_index: int,
    document_name: str,
    page_number: Optional[int],
    chunk_id: str,
) -> str:
    """
    Constructs a standardized, verifiable citation header:
    [SOURCE CHUNK 1 — Document: example.pdf — Page: 4 — Chunk ID: ...]
    Handles missing page info without fabricating numbers.
    """
    page_str = f"Page: {page_number}" if page_number is not None else "Page: N/A"
    clean_doc = document_name.strip() if document_name and document_name.strip() else "Unknown Document"
    return f"[SOURCE CHUNK {rank_index} — Document: {clean_doc} — {page_str} — Chunk ID: {chunk_id}]"


class RAGRetriever:
    """
    Retrieval-Augmented Generation (RAG) context engine.
    Transforms user queries into dense vectors, retrieves top-k chunks,
    applies similarity filters, deduplicates context, and builds structured citations.
    """

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 5,
        default_threshold: Optional[float] = None,
    ):
        self._provider = embedding_provider
        self._store = vector_store
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold

    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        if self._provider is None:
            self._provider = get_embedding_provider()
        return self._provider

    @property
    def vector_store(self) -> BaseVectorStore:
        if self._store is None:
            self._store = get_vector_store()
        return self._store

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        doc_id: Optional[str] = None,
        deduplicate: bool = True,
    ) -> RAGRetrievalResult:
        """
        Executes query retrieval, ranking, deduplication, and context assembly.
        """
        if not query or not query.strip():
            raise AppException(
                message="Retrieval query cannot be empty or whitespace.",
                status_code=400,
                code="EMPTY_QUERY_ERROR",
            )

        clean_query = query.strip()
        effective_top_k = max(1, top_k or self.default_top_k)
        effective_threshold = score_threshold if score_threshold is not None else self.default_threshold

        # 1. Embed user query
        try:
            query_vector = await self.embedding_provider.embed_query(clean_query)
        except Exception as exc:
            logger.error(f"Failed to generate query embedding for query '{clean_query}': {exc}")
            raise AppException(
                message=f"Failed to embed retrieval query: {str(exc)}",
                status_code=502,
                code="QUERY_EMBEDDING_FAILED",
                details=str(exc),
            )

        # 2. Vector similarity search (over-fetch slightly if deduplication is requested)
        fetch_k = effective_top_k * 2 if deduplicate else effective_top_k
        try:
            raw_hits: List[SimilaritySearchResult] = await self.vector_store.similarity_search(
                query_vector=query_vector,
                top_k=fetch_k,
                score_threshold=effective_threshold,
                doc_id=doc_id,
            )
        except Exception as exc:
            logger.error(f"Vector search failed for query '{clean_query}': {exc}")
            raise AppException(
                message=f"Vector similarity search failed: {str(exc)}",
                status_code=500,
                code="VECTOR_SEARCH_FAILED",
                details=str(exc),
            )

        # 3. Deduplicate repeating chunks/content
        selected_hits: List[SimilaritySearchResult] = []
        seen_chunk_ids = set()
        seen_texts = set()

        for hit in raw_hits:
            if deduplicate:
                if hit.chunk_id in seen_chunk_ids:
                    continue
                normalized_text = hit.text.strip().lower()
                if normalized_text in seen_texts:
                    continue
                seen_chunk_ids.add(hit.chunk_id)
                seen_texts.add(normalized_text)

            selected_hits.append(hit)
            if len(selected_hits) >= effective_top_k:
                break

        # 4. Build RetrievedChunk records and formatted citations
        retrieved_chunks: List[RetrievedChunk] = []
        context_blocks: List[str] = []

        for idx, hit in enumerate(selected_hits, start=1):
            doc_name = (
                hit.metadata.get("filename")
                or hit.metadata.get("title")
                or hit.doc_id
                or "Unknown Document"
            )
            citation = format_citation_header(
                rank_index=idx,
                document_name=str(doc_name),
                page_number=hit.page_number,
                chunk_id=hit.chunk_id,
            )

            retrieved_chunk = RetrievedChunk(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                chunk_index=hit.chunk_index,
                text=hit.text,
                similarity_score=hit.similarity_score,
                page_number=hit.page_number,
                document_name=str(doc_name),
                citation_header=citation,
                character_count=hit.character_count,
                token_count=hit.token_count,
                metadata=dict(hit.metadata),
            )
            retrieved_chunks.append(retrieved_chunk)
            context_blocks.append(f"{citation}\n{hit.text}")

        formatted_context = "\n\n".join(context_blocks)

        logger.info(
            f"RAG retrieved {len(retrieved_chunks)} chunks for query: '{clean_query[:50]}' "
            f"(doc_id filter: {doc_id})"
        )

        return RAGRetrievalResult(
            query=clean_query,
            retrieved_chunks=retrieved_chunks,
            formatted_context=formatted_context,
            total_chunks_found=len(retrieved_chunks),
            doc_id_filter=doc_id,
        )


_RAG_RETRIEVER_INSTANCE: Optional[RAGRetriever] = None


def get_rag_retriever(
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    vector_store: Optional[BaseVectorStore] = None,
    force_new: bool = False,
) -> RAGRetriever:
    """
    Factory creating or returning the configured RAG retriever.
    """
    global _RAG_RETRIEVER_INSTANCE
    if force_new or _RAG_RETRIEVER_INSTANCE is None or embedding_provider or vector_store:
        instance = RAGRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        if not embedding_provider and not vector_store and not force_new:
            _RAG_RETRIEVER_INSTANCE = instance
        return instance
    return _RAG_RETRIEVER_INSTANCE
