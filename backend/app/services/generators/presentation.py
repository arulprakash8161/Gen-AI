from typing import List, Type
from pydantic import BaseModel
from backend.app.models.request import OutputType, TransformationRequest
from backend.app.models.deliverable import PresentationDeliverable
from backend.app.services.generators.base import BaseDeliverableGenerator
from backend.app.services.llm.base import BaseLLMClient


class PresentationGenerator(BaseDeliverableGenerator):
    """
    Generates structured slide presentations ready for delivery or PPTX export.
    """

    @property
    def output_type(self) -> OutputType:
        return OutputType.PRESENTATION

    @property
    def schema(self) -> Type[BaseModel]:
        return PresentationDeliverable

    def build_system_prompt(self, request: TransformationRequest) -> str:
        base = super().build_system_prompt(request)
        return (
            f"{base}\n\n"
            "ADDITIONAL PRESENTATION SLIDE DECK INSTRUCTIONS:\n"
            "- Create a coherent, persuasive slide deck sequence (Title -> Context -> Findings -> Strategy -> Next Steps).\n"
            "- Each slide must contain a focused title, 3-5 concise bullet points, and detailed speaker notes.\n"
            "- Include visual cue recommendations (e.g. 'Bar chart of Q3 growth', 'Workflow diagram').\n"
            "- Tailor the number of slides to the detail level (3-5 for high_level, 5-8 for standard, 8+ for comprehensive)."
        )

    async def generate(
        self,
        context: str,
        source_chunk_ids: List[str],
        request: TransformationRequest,
        llm_client: BaseLLMClient,
    ) -> PresentationDeliverable:
        prompt = self.build_user_prompt(context, request)
        system_prompt = self.build_system_prompt(request)

        result: PresentationDeliverable = await llm_client.generate_structured(
            prompt=prompt,
            schema=PresentationDeliverable,
            system_prompt=system_prompt,
        )

        # Ensure source chunks passed from RAG retrieval are recorded
        result.source_chunk_ids = list(source_chunk_ids)

        return result
