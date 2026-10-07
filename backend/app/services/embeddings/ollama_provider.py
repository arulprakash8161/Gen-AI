from typing import List, Optional
import httpx

from backend.app.services.embeddings.base import BaseEmbeddingProvider
from backend.app.core.config import settings
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider utilizing a local or remote Ollama instance.
    Uses the Ollama /api/embed or /api/embeddings REST endpoint.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: int = 768,
        timeout: Optional[int] = None,
    ):
        self._base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._dimension = dimension
        self._timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_query(self, text: str) -> List[float]:
        """Generates an embedding for a single text query."""
        results = await self.embed_documents([text])
        if not results:
            raise AppException(
                message="Ollama returned empty embedding for query.",
                status_code=502,
                code="OLLAMA_EMBEDDING_EMPTY",
            )
        return results[0]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using Ollama.
        Tries batch /api/embed endpoint first, falling back to /api/embeddings per item.
        """
        if not texts:
            return []

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # 1. Try Ollama's modern batch /api/embed endpoint
            try:
                response = await client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model_name, "input": texts},
                )
                if response.status_code == 200:
                    data = response.json()
                    embeddings = data.get("embeddings") or []
                    if embeddings:
                        self._dimension = len(embeddings[0])
                        return embeddings
            except httpx.ConnectError:
                raise AppException(
                    message=f"Cannot connect to Ollama service at '{self._base_url}'. Is Ollama running?",
                    status_code=503,
                    code="OLLAMA_CONNECTION_ERROR",
                )
            except httpx.TimeoutException:
                raise AppException(
                    message=f"Ollama embedding request timed out after {self._timeout}s.",
                    status_code=504,
                    code="OLLAMA_TIMEOUT",
                )
            except Exception as e:
                logger.debug(f"Ollama /api/embed endpoint failed, trying fallback: {e}")

            # 2. Fallback to classic /api/embeddings endpoint per text
            embeddings: List[List[float]] = []
            for text in texts:
                try:
                    response = await client.post(
                        f"{self._base_url}/api/embeddings",
                        json={"model": self._model_name, "prompt": text},
                    )
                    if response.status_code != 200:
                        raise AppException(
                            message=f"Ollama embedding failed with status {response.status_code}: {response.text}",
                            status_code=502,
                            code="OLLAMA_EMBEDDING_FAILED",
                        )
                    data = response.json()
                    vec = data.get("embedding", [])
                    if not vec:
                        raise AppException(
                            message=f"Ollama returned empty embedding for model '{self._model_name}'.",
                            status_code=502,
                            code="OLLAMA_EMPTY_EMBEDDING",
                        )
                    self._dimension = len(vec)
                    embeddings.append(vec)
                except (httpx.ConnectError, httpx.TimeoutException) as net_err:
                    raise AppException(
                        message=f"Network error communicating with Ollama at '{self._base_url}': {str(net_err)}",
                        status_code=503,
                        code="OLLAMA_NETWORK_ERROR",
                    )

            return embeddings
