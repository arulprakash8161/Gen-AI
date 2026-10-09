from typing import List, Type
from pydantic import BaseModel
from backend.app.models.request import OutputType, TransformationRequest
from backend.app.models.deliverable import LinkedInPostDeliverable
from backend.app.services.generators.base import BaseDeliverableGenerator
from backend.app.services.llm.base import BaseLLMClient


class LinkedInGenerator(BaseDeliverableGenerator):
    """
    Generates high-engagement, professional LinkedIn posts strictly grounded in source documents.
    """

    @property
    def output_type(self) -> OutputType:
        return OutputType.LINKEDIN

    @property
    def schema(self) -> Type[BaseModel]:
        return LinkedInPostDeliverable

    def build_system_prompt(self, request: TransformationRequest) -> str:
        base = super().build_system_prompt(request)
        return (
            f"{base}\n\n"
            "ADDITIONAL LINKEDIN FORMATTING INSTRUCTIONS:\n"
            "- Hook the reader with a strong, insightful opening line.\n"
            "- Use clean formatting with line breaks and concise bullet points.\n"
            "- Extract 3-5 relevant industry hashtags.\n"
            "- End with an actionable question or call to action to spark discussion."
        )

    async def generate(
        self,
        context: str,
        source_chunk_ids: List[str],
        request: TransformationRequest,
        llm_client: BaseLLMClient,
    ) -> LinkedInPostDeliverable:
        prompt = self.build_user_prompt(context, request)
        system_prompt = self.build_system_prompt(request)

        result: LinkedInPostDeliverable = await llm_client.generate_structured(
            prompt=prompt,
            schema=LinkedInPostDeliverable,
            system_prompt=system_prompt,
        )

        # Ensure source chunks passed from RAG retrieval are recorded
        result.source_chunk_ids = list(source_chunk_ids)

        return result
