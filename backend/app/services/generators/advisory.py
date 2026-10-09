from typing import List, Type
from pydantic import BaseModel
from backend.app.models.request import OutputType, TransformationRequest
from backend.app.models.deliverable import OfficialAdvisoryDeliverable
from backend.app.services.generators.base import BaseDeliverableGenerator
from backend.app.services.llm.base import BaseLLMClient


class OfficialAdvisoryGenerator(BaseDeliverableGenerator):
    """
    Generates formal, compliance-oriented public or institutional advisories.
    """

    @property
    def output_type(self) -> OutputType:
        return OutputType.ADVISORY

    @property
    def schema(self) -> Type[BaseModel]:
        return OfficialAdvisoryDeliverable

    def build_system_prompt(self, request: TransformationRequest) -> str:
        base = super().build_system_prompt(request)
        return (
            f"{base}\n\n"
            "ADDITIONAL OFFICIAL ADVISORY INSTRUCTIONS:\n"
            "- Adopt an authoritative, formal, and precise tone.\n"
            "- Include reference code, urgency rating, and effective dates.\n"
            "- Articulate background rationale, specific mandate details, and compliance requirements.\n"
            "- Identify the issuing authority or department."
        )

    async def generate(
        self,
        context: str,
        source_chunk_ids: List[str],
        request: TransformationRequest,
        llm_client: BaseLLMClient,
    ) -> OfficialAdvisoryDeliverable:
        prompt = self.build_user_prompt(context, request)
        system_prompt = self.build_system_prompt(request)

        result: OfficialAdvisoryDeliverable = await llm_client.generate_structured(
            prompt=prompt,
            schema=OfficialAdvisoryDeliverable,
            system_prompt=system_prompt,
        )

        # Ensure source chunks passed from RAG retrieval are recorded
        result.source_chunk_ids = list(source_chunk_ids)

        return result
