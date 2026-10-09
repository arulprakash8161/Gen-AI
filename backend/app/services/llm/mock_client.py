from typing import Any, Dict, List, Optional, Type, TypeVar, get_args, get_origin
from pydantic import BaseModel

from backend.app.services.llm.base import BaseLLMClient
from backend.app.core.exceptions import AppException

T = TypeVar("T", bound=BaseModel)


def generate_mock_field_value(field_name: str, field_type: Any, depth: int = 0) -> Any:
    """
    Recursively synthesizes realistic, schema-valid mock values based on field name and type annotation.
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Optional types (Union[X, None])
    if origin is not None and type(None) in args:
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return generate_mock_field_value(field_name, non_none_args[0], depth=depth)

    # Handle Lists
    if origin is list or field_type is list:
        item_type = args[0] if args else str
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            return [generate_mock_instance(item_type, depth=depth + 1) for _ in range(3)]
        if field_name in ("hashtags", "tags"):
            return ["#GenAI", "#Innovation", "#SIH2024", "#Automation"]
        if "action" in field_name or "recommendation" in field_name:
            return [
                "Implement role-based access control across endpoints.",
                "Deploy automated regression monitoring alerts.",
                "Review audit logs weekly.",
            ]
        if "finding" in field_name or "point" in field_name or "bullet" in field_name:
            return [
                "Primary objective achieved with 99.8% precision.",
                "Zero data leakage detected across ingestion gates.",
                "Automated processing decreased latency by 65%.",
            ]
        return [f"Key point 1 for {field_name}", f"Key point 2 for {field_name}"]

    # Handle Nested Pydantic Models
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        return generate_mock_instance(field_type, depth=depth + 1)

    # Handle Primitive String Types with contextual values
    if field_type is str or origin is None and isinstance(field_type, type) and issubclass(field_type, str):
        if "hook" in field_name or "headline" in field_name or "title" in field_name:
            return "Transforming Multi-Channel Communications with Generative AI"
        if "call_to_action" in field_name or "cta" in field_name:
            return "How is your organization scaling intelligent document workflows? Share your thoughts below!"
        if "summary" in field_name or "overview" in field_name:
            return "This executive summary consolidates the architectural highlights and performance indicators of the system."
        if "speaker_note" in field_name or "notes" in field_name:
            return "Emphasize to stakeholders that all metrics are strictly grounded in source documents."
        if "script" in field_name or "narration" in field_name:
            return "Welcome everyone. Today we are walking through the revolutionary AI content pipeline."
        if "visual" in field_name or "scene" in field_name:
            return "Close-up cinematic shot of architecture diagram with glowing data pathways."
        if "severity" in field_name:
            return "HIGH"
        if "impact" in field_name:
            return "Potential authorization bypass if unpatched."
        return f"Synthesized content for {field_name.replace('_', ' ').capitalize()}."

    if field_type is int or origin is None and isinstance(field_type, type) and issubclass(field_type, int):
        if "slide" in field_name or "scene" in field_name:
            return 1
        return 10

    if field_type is float:
        return 0.95

    if field_type is bool:
        return True

    return None


def generate_mock_instance(schema: Type[T], depth: int = 0) -> T:
    """
    Constructs a valid Pydantic model populated with realistic placeholder values.
    """
    data: Dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        annotation = field.annotation
        data[name] = generate_mock_field_value(name, annotation, depth=depth)
    return schema.model_validate(data)


class MockLLMClient(BaseLLMClient):
    """
    Deterministic offline mock client designed for automated test suites and
    offline development environments where Ollama is not active.
    """

    def __init__(self, model_name: str = "mock-llama3.2"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

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
        return (
            "Mock completion: Synthesized multi-channel communication deliverable "
            "strictly grounded in the provided source chunks."
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        if not prompt or not prompt.strip():
            raise AppException(
                message="Prompt cannot be empty.",
                status_code=400,
                code="EMPTY_PROMPT_ERROR",
            )
        return generate_mock_instance(schema)

    async def check_health(self) -> bool:
        return True
