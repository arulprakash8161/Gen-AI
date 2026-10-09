from typing import Dict
from backend.app.core.exceptions import AppException
from backend.app.models.request import OutputType
from backend.app.services.generators.base import BaseDeliverableGenerator
from backend.app.services.generators.linkedin import LinkedInGenerator
from backend.app.services.generators.summary import ExecutiveSummaryGenerator
from backend.app.services.generators.advisory import OfficialAdvisoryGenerator
from backend.app.services.generators.presentation import PresentationGenerator
from backend.app.services.generators.video_package import VideoPackageGenerator

_GENERATORS: Dict[OutputType, BaseDeliverableGenerator] = {
    OutputType.LINKEDIN: LinkedInGenerator(),
    OutputType.SUMMARY: ExecutiveSummaryGenerator(),
    OutputType.ADVISORY: OfficialAdvisoryGenerator(),
    OutputType.PRESENTATION: PresentationGenerator(),
    OutputType.VIDEO_PACKAGE: VideoPackageGenerator(),
}


def get_generator(output_type: OutputType) -> BaseDeliverableGenerator:
    """
    Returns the deliverable generator registered for the given output type.
    """
    generator = _GENERATORS.get(output_type)
    if not generator:
        raise AppException(
            code="UNSUPPORTED_OUTPUT_TYPE",
            message=f"No generator registered for output type: '{output_type}'",
            status_code=400,
        )
    return generator


def get_all_generators() -> Dict[OutputType, BaseDeliverableGenerator]:
    """
    Returns a copy of all registered deliverable generators.
    """
    return dict(_GENERATORS)
