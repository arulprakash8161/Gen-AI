import time
from uuid import uuid4
from typing import Optional, List
from backend.app.core.config import settings
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger
from backend.app.models.request import (
    TransformationRequest,
    DetailLevel,
    OutputType,
)
from backend.app.models.deliverable import TransformationResponse
from backend.app.services.rag.retriever import RAGRetriever, get_rag_retriever
from backend.app.services.llm.base import BaseLLMClient
from backend.app.services.llm.factory import get_llm_client
from backend.app.services.generators.factory import get_generator

logger = get_logger(__name__)


class TransformationOrchestrator:
    """
    Central orchestration service for multi-channel document transformation.
    Coordinates RAG retrieval, context assembly, grounding calculation,
    and concurrent/sequential dispatch to domain deliverable generators.
    """

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self._retriever = retriever
        self._llm_client = llm_client

    @property
    def retriever(self) -> RAGRetriever:
        if self._retriever is None:
            self._retriever = get_rag_retriever()
        return self._retriever

    @property
    def llm_client(self) -> BaseLLMClient:
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def _determine_top_k(self, request: TransformationRequest) -> int:
        """
        Determines optimal top-k retrieval count based on detail level or client override.
        """
        if request.top_k is not None:
            return request.top_k

        if request.detail_level == DetailLevel.HIGH_LEVEL.value:
            return 3
        if request.detail_level == DetailLevel.COMPREHENSIVE.value:
            return 8
        return getattr(settings, "TOP_K_CHUNKS", 5)

    def _build_retrieval_query(self, request: TransformationRequest) -> str:
        """
        Synthesizes a dense retrieval query tailored to the user's objective and outputs.
        """
        if request.custom_query and request.custom_query.strip():
            return request.custom_query.strip()

        if request.objective and request.objective.strip():
            return f"{request.objective.strip()} - key findings, core facts, background, objectives, and directives"

        return "Executive summary, core findings, strategic implications, background, and actionable recommendations"

    async def transform(self, request: TransformationRequest) -> TransformationResponse:
        """
        Executes end-to-end transformation workflow:
        1. Retrieves and deduplicates grounded context from vector store.
        2. Computes grounding confidence score.
        3. Dispatches context to each requested deliverable generator.
        4. Packages and returns unified TransformationResponse.
        """
        start_time = time.time()
        logger.info(
            f"Starting transformation for doc_id='{request.document_id}' with "
            f"{len(request.output_types)} output types: {[ot.value for ot in request.output_types]}"
        )

        # 1. Determine retrieval parameters
        top_k = self._determine_top_k(request)
        retrieval_query = self._build_retrieval_query(request)

        # 2. Retrieve grounded RAG context
        retrieval_res = await self.retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            doc_id=request.document_id,
        )

        # If zero chunks matched, verify if the document even exists in the vector store
        if retrieval_res.total_chunks_found == 0:
            doc_chunks = await self.retriever.vector_store.get_chunks_by_doc_id(request.document_id)
            if not doc_chunks:
                raise AppException(
                    message=f"Document '{request.document_id}' is not indexed or contains no stored chunks.",
                    status_code=404,
                    code="DOCUMENT_NOT_INDEXED",
                )
            # Chunks exist but query missed; fallback to retrieving without score threshold
            retrieval_res = await self.retriever.retrieve(
                query=retrieval_query,
                top_k=top_k,
                doc_id=request.document_id,
                score_threshold=-1.0,
            )

        context = retrieval_res.formatted_context
        source_chunk_ids: List[str] = [c.chunk_id for c in retrieval_res.retrieved_chunks]
        citations: List[str] = [c.citation_header for c in retrieval_res.retrieved_chunks]

        # 3. Calculate grounding confidence score
        if retrieval_res.retrieved_chunks:
            avg_similarity = sum(c.similarity_score for c in retrieval_res.retrieved_chunks) / len(retrieval_res.retrieved_chunks)
            # Map cosine similarity [-1, 1] to normalized confidence [0, 1]
            normalized_score = max(0.0, min(1.0, (avg_similarity + 1.0) / 2.0))
            grounding_score = round(normalized_score, 3)
        else:
            grounding_score = 0.0

        # 4. Generate deliverables for each requested output type
        task_id = f"trans_{uuid4().hex[:12]}"
        response = TransformationResponse(
            task_id=task_id,
            document_id=request.document_id,
            status="completed",
            citations=citations,
            grounding_score=grounding_score,
            metadata={
                "retrieved_chunks_count": len(retrieval_res.retrieved_chunks),
                "output_types": [ot.value for ot in request.output_types],
                "detail_level": request.detail_level,
                "audience": request.audience,
                "tone": request.tone,
            },
        )

        for output_type in request.output_types:
            logger.info(f"Generating deliverable '{output_type.value}' for task '{task_id}'")
            generator = get_generator(output_type)
            deliverable = await generator.generate(
                context=context,
                source_chunk_ids=source_chunk_ids,
                request=request,
                llm_client=self.llm_client,
            )

            if output_type == OutputType.LINKEDIN:
                response.linkedin = deliverable
            elif output_type == OutputType.SUMMARY:
                response.summary = deliverable
            elif output_type == OutputType.ADVISORY:
                response.advisory = deliverable
            elif output_type == OutputType.PRESENTATION:
                response.presentation = deliverable
            elif output_type == OutputType.VIDEO_PACKAGE:
                response.video_package = deliverable

        elapsed_ms = int((time.time() - start_time) * 1000)
        response.metadata["elapsed_ms"] = elapsed_ms
        logger.info(f"Transformation task '{task_id}' completed in {elapsed_ms}ms")

        return response
