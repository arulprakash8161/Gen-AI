from typing import Optional

from backend.app.services.llm.base import BaseLLMClient
from backend.app.services.llm.ollama_client import OllamaLLMClient
from backend.app.services.llm.mock_client import MockLLMClient
from backend.app.core.config import settings
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_LLM_CLIENT: Optional[BaseLLMClient] = None


def get_llm_client(
    provider_type: Optional[str] = None,
    force_new: bool = False,
) -> BaseLLMClient:
    """
    Factory creating or returning the configured LLM client.
    Supported types: 'local'/'mock', 'ollama'.
    """
    global _DEFAULT_LLM_CLIENT

    selected = (provider_type or getattr(settings, "LLM_PROVIDER", "local")).lower().strip()

    if force_new or _DEFAULT_LLM_CLIENT is None or provider_type is not None:
        if selected in ("mock", "local", "deterministic"):
            client = MockLLMClient()
        elif selected == "ollama":
            client = OllamaLLMClient(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=settings.OLLAMA_MODEL,
                timeout=settings.OLLAMA_TIMEOUT_SECONDS,
            )
        else:
            raise AppException(
                message=f"Unknown LLM provider '{selected}'. Supported providers: 'local', 'ollama'.",
                status_code=400,
                code="UNKNOWN_LLM_PROVIDER",
            )

        if provider_type is None and not force_new:
            _DEFAULT_LLM_CLIENT = client
        return client

    return _DEFAULT_LLM_CLIENT
