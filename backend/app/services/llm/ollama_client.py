import json
import re
from typing import Any, Dict, Optional, Type, TypeVar
import httpx
from pydantic import BaseModel, ValidationError

from backend.app.services.llm.base import BaseLLMClient
from backend.app.core.config import settings
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json_payload(raw_text: str) -> str:
    """
    Extracts JSON substring from raw model output, handling potential markdown fences (```json ... ```).
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Attempt to locate the outer JSON braces or brackets if model returned conversational preamble
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if match:
        return match.group(0).strip()

    return cleaned


class OllamaLLMClient(BaseLLMClient):
    """
    Inference client connecting to local or remote Ollama server instances.
    Supports free-form text completions and schema-validated JSON completions.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self._base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._model_name = model_name or settings.OLLAMA_MODEL
        self._timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def base_url(self) -> str:
        return self._base_url

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise AppException(
                message="Prompt cannot be empty.",
                status_code=400,
                code="EMPTY_PROMPT_ERROR",
            )

        payload: Dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt.strip(),
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system_prompt and system_prompt.strip():
            payload["system"] = system_prompt.strip()

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        url = f"{self._base_url}/api/generate"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)

            if response.status_code != 200:
                raise AppException(
                    message=f"Ollama API returned HTTP {response.status_code}: {response.text}",
                    status_code=502,
                    code="OLLAMA_API_ERROR",
                    details=response.text,
                )

            data = response.json()
            completion = data.get("response", "")
            return completion.strip()

        except httpx.ConnectError as exc:
            logger.error(f"Cannot connect to Ollama at {self._base_url}: {exc}")
            raise AppException(
                message=f"Cannot connect to Ollama server at {self._base_url}. Ensure Ollama is running.",
                status_code=503,
                code="OLLAMA_CONNECTION_ERROR",
                details=str(exc),
            )
        except httpx.TimeoutException as exc:
            logger.error(f"Ollama request timed out after {self._timeout}s: {exc}")
            raise AppException(
                message=f"Ollama inference timed out after {self._timeout} seconds.",
                status_code=504,
                code="OLLAMA_TIMEOUT",
                details=str(exc),
            )
        except AppException:
            raise
        except Exception as exc:
            logger.error(f"Unexpected error communicating with Ollama: {exc}")
            raise AppException(
                message=f"Error communicating with Ollama: {str(exc)}",
                status_code=500,
                code="OLLAMA_CLIENT_ERROR",
                details=str(exc),
            )

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """
        Forces Ollama to emit JSON conforming to the requested Pydantic schema.
        """
        schema_json_str = json.dumps(schema.model_json_schema(), indent=2)

        # Enhance system instructions with strict schema adherence directives
        guidance = (
            "You MUST output valid, parseable JSON strictly matching this schema. "
            "Do not include explanation, greetings, or conversational preambles outside the JSON.\n"
            f"Schema:\n{schema_json_str}"
        )

        full_system = f"{system_prompt.strip()}\n\n{guidance}" if system_prompt else guidance

        payload: Dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt.strip(),
            "system": full_system,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        url = f"{self._base_url}/api/generate"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)

            if response.status_code != 200:
                raise AppException(
                    message=f"Ollama API returned HTTP {response.status_code}: {response.text}",
                    status_code=502,
                    code="OLLAMA_API_ERROR",
                    details=response.text,
                )

            data = response.json()
            raw_response = data.get("response", "")
            json_str = extract_json_payload(raw_response)

            try:
                parsed_instance = schema.model_validate_json(json_str)
                return parsed_instance
            except ValidationError as val_err:
                logger.error(f"Structured validation failed for model {schema.__name__}: {val_err}. Raw: {raw_response}")
                raise AppException(
                    message=f"Model output did not match expected schema '{schema.__name__}': {str(val_err)}",
                    status_code=500,
                    code="STRUCTURED_OUTPUT_VALIDATION_ERROR",
                    details={"raw": raw_response, "errors": val_err.errors()},
                )

        except (httpx.ConnectError, httpx.TimeoutException, AppException):
            raise
        except Exception as exc:
            logger.error(f"Unexpected error in Ollama structured generation: {exc}")
            raise AppException(
                message=f"Failed to generate structured completion: {str(exc)}",
                status_code=500,
                code="STRUCTURED_GENERATION_FAILED",
                details=str(exc),
            )

    async def check_health(self) -> bool:
        url = f"{self._base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(url)
            return res.status_code == 200
        except Exception:
            return False
