from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """
    Abstract interface for Language Model (LLM) inference clients.
    Decouples transformation engines and generators from underlying inference servers
    (e.g., local Ollama, vLLM, mock testing clients).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the active model name/tag."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generates a raw string completion from the model given a user prompt and optional system prompt.
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """
        Generates a structured completion strictly conforming to a given Pydantic schema model.
        """
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """
        Verifies connectivity and readiness of the underlying model inference service.
        """
        pass
