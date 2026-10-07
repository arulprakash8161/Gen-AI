from typing import Optional
from backend.app.services.embeddings.base import BaseEmbeddingProvider
from backend.app.services.embeddings.ollama_provider import OllamaEmbeddingProvider
from backend.app.services.embeddings.mock_provider import DeterministicLocalEmbeddingProvider
from backend.app.core.config import settings
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


def get_embedding_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
    """
    Factory creating an embedding provider based on configuration or explicit request.
    Supported types: 'ollama', 'mock', 'local'.
    """
    selected = (provider_type or settings.EMBEDDING_PROVIDER).lower().strip()

    if selected == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model_name=settings.EMBEDDING_MODEL,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
    elif selected in ("mock", "local", "deterministic"):
        return DeterministicLocalEmbeddingProvider()
    else:
        raise AppException(
            message=f"Unknown embedding provider '{selected}'. Supported providers: 'ollama', 'local'.",
            status_code=400,
            code="UNKNOWN_EMBEDDING_PROVIDER",
        )
