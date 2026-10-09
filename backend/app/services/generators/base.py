from abc import ABC, abstractmethod
from typing import List, Type, TypeVar
from pydantic import BaseModel
from backend.app.models.request import OutputType, TransformationRequest
from backend.app.services.llm.base import BaseLLMClient

T = TypeVar("T", bound=BaseModel)


class BaseDeliverableGenerator(ABC):
    """
    Abstract base class for domain-specific content generators.
    """

    @property
    @abstractmethod
    def output_type(self) -> OutputType:
        """The OutputType handled by this generator."""
        pass

    @property
    @abstractmethod
    def schema(self) -> Type[BaseModel]:
        """The Pydantic deliverable schema produced by this generator."""
        pass

    def build_system_prompt(self, request: TransformationRequest) -> str:
        """
        Constructs a strict grounding system prompt enforcing zero hallucinations.
        """
        return (
            "You are an enterprise AI Content Transformation Engine specializing in document synthesis.\n"
            "STRICT GROUNDING RULES:\n"
            "1. Ground every statement exclusively on facts, figures, and claims present in the provided source chunks.\n"
            "2. DO NOT speculate, invent facts, or extrapolate beyond the text.\n"
            "3. If a detail is missing from the source context, omit it or state that it is not specified.\n"
            "4. Respect the requested audience perspective, tone, and language without violating factual integrity.\n"
            "5. Return your answer ONLY as valid JSON conforming strictly to the requested schema."
        )

    def build_user_prompt(self, context: str, request: TransformationRequest) -> str:
        """
        Constructs the generation prompt with context delimiters and parameters.
        """
        prompt_parts = [
            f"Please generate a {self.output_type.value.replace('_', ' ').title()} deliverable based strictly on the source context below.",
            f"- Target Audience: {request.audience}",
            f"- Stylistic Tone: {request.tone}",
            f"- Output Language: {request.language}",
            f"- Detail Level: {request.detail_level}",
        ]

        if request.objective:
            prompt_parts.append(f"- Specific Focus / Objective: {request.objective}")

        prompt_parts.append("\n=== SOURCE CONTEXT ===")
        prompt_parts.append(context if context.strip() else "[No source chunks provided]")
        prompt_parts.append("=== END SOURCE CONTEXT ===\n")
        prompt_parts.append("Synthesize the material into the required structured format.")

        return "\n".join(prompt_parts)

    @abstractmethod
    async def generate(
        self,
        context: str,
        source_chunk_ids: List[str],
        request: TransformationRequest,
        llm_client: BaseLLMClient,
    ) -> BaseModel:
        """
        Executes structured generation using the provided LLM client and source context.
        """
        pass
