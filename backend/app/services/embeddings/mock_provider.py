import math
import hashlib
from typing import List
from backend.app.services.embeddings.base import BaseEmbeddingProvider


class DeterministicLocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Offline, deterministic embedding provider designed for rapid local testing
    and environments where external model servers are not actively running.
    Produces unit-normalized dense vectors with semantic term-overlap sensitivity.
    """

    def __init__(self, dimension: int = 384, model_name: str = "local-deterministic-384"):
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self._dimension

        vec = [0.0] * self._dimension
        words = text.lower().split()

        for word in words:
            # Deterministically hash each word into a bucket and sign
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            bucket = h % self._dimension
            sign = 1.0 if ((h >> 16) & 1) == 1 else -1.0
            vec[bucket] += sign

            # Spread to adjacent harmonic bucket for dense representation
            secondary_bucket = (h >> 8) % self._dimension
            vec[secondary_bucket] += (sign * 0.5)

        # L2 Normalize the vector to unit length
        norm = math.sqrt(sum(val * val for val in vec))
        if norm > 0.0:
            vec = [val / norm for val in vec]
        else:
            vec[0] = 1.0

        return [round(val, 6) for val in vec]

    async def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]
