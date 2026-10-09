from typing import List, Type
from pydantic import BaseModel
from backend.app.models.request import OutputType, TransformationRequest
from backend.app.models.deliverable import ExecutiveSummaryDeliverable
from backend.app.services.generators.base import BaseDeliverableGenerator
from backend.app.services.llm.base import BaseLLMClient


class ExecutiveSummaryGenerator(BaseDeliverableGenerator):
    """
    Generates high-density executive briefings and strategic syntheses.
    """

    @property
    def output_type(self) -> OutputType:
        return OutputType.SUMMARY

    @property
    def schema(self) -> Type[BaseModel]:
        return ExecutiveSummaryDeliverable

    def build_system_prompt(self, request: TransformationRequest) -> str:
        base = super().build_system_prompt(request)
        return (
            f"{base}\n\n"
            "ADDITIONAL EXECUTIVE SUMMARY INSTRUCTIONS:\n"
            "- Synthesize key takeaways into an executive brief suitable for C-level readers.\n"
            "- Highlight concrete facts, quantitative metrics, and outcomes in key findings.\n"
            "- Formulate strategic implications, immediate action items, and known risk factors."
        )

    async def generate(
        self,
        context: str,
        source_chunk_ids: List[str],
        request: TransformationRequest,
        llm_client: BaseLLMClient,
    ) -> ExecutiveSummaryDeliverable:
        prompt = self.build_user_prompt(context, request)
        system_prompt = self.build_system_prompt(request)

        result: ExecutiveSummaryDeliverable = await llm_client.generate_structured(
            prompt=prompt,
            schema=ExecutiveSummaryDeliverable,
            system_prompt=system_prompt,
        )

        # Ensure source chunks passed from RAG retrieval are recorded
        result.source_chunk_ids = list(source_chunk_ids)

        return result
