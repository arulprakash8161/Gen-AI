from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """
    Abstract interface for generating vector embeddings from text.
    Decouples the system from any single model or embedding service.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vector."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single query or input text.
        """
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of text chunks/documents.
        """
        pass
