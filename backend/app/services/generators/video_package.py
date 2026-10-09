from typing import List, Type
from pydantic import BaseModel
from backend.app.models.request import OutputType, TransformationRequest
from backend.app.models.deliverable import VideoPackageDeliverable
from backend.app.services.generators.base import BaseDeliverableGenerator
from backend.app.services.llm.base import BaseLLMClient


class VideoPackageGenerator(BaseDeliverableGenerator):
    """
    Generates video scripts, storyboard breakdowns, scene directions, and voiceovers.
    """

    @property
    def output_type(self) -> OutputType:
        return OutputType.VIDEO_PACKAGE

    @property
    def schema(self) -> Type[BaseModel]:
        return VideoPackageDeliverable

    def build_system_prompt(self, request: TransformationRequest) -> str:
        base = super().build_system_prompt(request)
        return (
            f"{base}\n\n"
            "ADDITIONAL VIDEO STORYBOARD & SCRIPT INSTRUCTIONS:\n"
            "- Begin with an attention-grabbing hook in the first 3-5 seconds.\n"
            "- Sequence chronological scenes with clear visual descriptions (B-roll, animation, graphics) and spoken voiceover lines.\n"
            "- Keep narration natural, dynamic, and synchronized with realistic timing (e.g. 10-20s per scene).\n"
            "- Conclude with an impactful outro call-to-action."
        )

    async def generate(
        self,
        context: str,
        source_chunk_ids: List[str],
        request: TransformationRequest,
        llm_client: BaseLLMClient,
    ) -> VideoPackageDeliverable:
        prompt = self.build_user_prompt(context, request)
        system_prompt = self.build_system_prompt(request)

        result: VideoPackageDeliverable = await llm_client.generate_structured(
            prompt=prompt,
            schema=VideoPackageDeliverable,
            system_prompt=system_prompt,
        )

        # Ensure source chunks passed from RAG retrieval are recorded
        result.source_chunk_ids = list(source_chunk_ids)

        return result
