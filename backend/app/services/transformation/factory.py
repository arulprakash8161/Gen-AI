from typing import Optional
from backend.app.services.transformation.orchestrator import TransformationOrchestrator
from backend.app.services.rag.retriever import RAGRetriever
from backend.app.services.llm.base import BaseLLMClient

_orchestrator_instance: Optional[TransformationOrchestrator] = None


def get_transformation_orchestrator(
    retriever: Optional[RAGRetriever] = None,
    llm_client: Optional[BaseLLMClient] = None,
    force_new: bool = False,
) -> TransformationOrchestrator:
    """
    Factory providing a configured TransformationOrchestrator instance.
    """
    global _orchestrator_instance
    if force_new or retriever is not None or llm_client is not None:
        return TransformationOrchestrator(retriever=retriever, llm_client=llm_client)

    if _orchestrator_instance is None:
        _orchestrator_instance = TransformationOrchestrator()

    return _orchestrator_instance
