import pytest
from backend.app.core.exceptions import AppException
from backend.app.models import (
    OutputType,
    TransformationRequest,
    LinkedInPostDeliverable,
    ExecutiveSummaryDeliverable,
    OfficialAdvisoryDeliverable,
    PresentationDeliverable,
    VideoPackageDeliverable,
)
from backend.app.services.llm.mock_client import MockLLMClient
from backend.app.services.generators import (
    LinkedInGenerator,
    ExecutiveSummaryGenerator,
    OfficialAdvisoryGenerator,
    PresentationGenerator,
    VideoPackageGenerator,
    get_generator,
    get_all_generators,
)


@pytest.fixture
def sample_context():
    return (
        "[SOURCE CHUNK 0 - Page 1 - Chunk ID: chunk_001]\n"
        "The automated transformation pipeline reduces manual authoring time by 75%.\n"
        "[SOURCE CHUNK 1 - Page 2 - Chunk ID: chunk_002]\n"
        "Security compliance requires all vector embeddings to be indexed locally."
    )


@pytest.fixture
def sample_request():
    return TransformationRequest(
        document_id="doc_test_123",
        output_types=[
            OutputType.LINKEDIN,
            OutputType.SUMMARY,
            OutputType.ADVISORY,
            OutputType.PRESENTATION,
            OutputType.VIDEO_PACKAGE,
        ],
        audience="enterprise executives",
        tone="authoritative",
        language="en",
        detail_level="standard",
        objective="Accelerate AI document processing adoption.",
    )


@pytest.fixture
def mock_llm():
    return MockLLMClient()


def test_generator_prompt_construction(sample_context, sample_request):
    gen = LinkedInGenerator()
    assert gen.output_type == OutputType.LINKEDIN
    assert gen.schema == LinkedInPostDeliverable

    sys_prompt = gen.build_system_prompt(sample_request)
    assert "STRICT GROUNDING RULES" in sys_prompt
    assert "ADDITIONAL LINKEDIN FORMATTING INSTRUCTIONS" in sys_prompt

    user_prompt = gen.build_user_prompt(sample_context, sample_request)
    assert "=== SOURCE CONTEXT ===" in user_prompt
    assert "Target Audience: enterprise executives" in user_prompt
    assert "Accelerate AI document processing adoption" in user_prompt


@pytest.mark.asyncio
async def test_linkedin_generator(sample_context, sample_request, mock_llm):
    gen = LinkedInGenerator()
    result = await gen.generate(
        context=sample_context,
        source_chunk_ids=["chunk_001", "chunk_002"],
        request=sample_request,
        llm_client=mock_llm,
    )
    assert isinstance(result, LinkedInPostDeliverable)
    assert result.headline
    assert result.content
    assert result.source_chunk_ids == ["chunk_001", "chunk_002"]


@pytest.mark.asyncio
async def test_executive_summary_generator(sample_context, sample_request, mock_llm):
    gen = ExecutiveSummaryGenerator()
    assert gen.output_type == OutputType.SUMMARY
    result = await gen.generate(
        context=sample_context,
        source_chunk_ids=["chunk_001"],
        request=sample_request,
        llm_client=mock_llm,
    )
    assert isinstance(result, ExecutiveSummaryDeliverable)
    assert result.title
    assert result.executive_brief
    assert result.source_chunk_ids == ["chunk_001"]


@pytest.mark.asyncio
async def test_official_advisory_generator(sample_context, sample_request, mock_llm):
    gen = OfficialAdvisoryGenerator()
    assert gen.output_type == OutputType.ADVISORY
    result = await gen.generate(
        context=sample_context,
        source_chunk_ids=["chunk_002"],
        request=sample_request,
        llm_client=mock_llm,
    )
    assert isinstance(result, OfficialAdvisoryDeliverable)
    assert result.reference_number
    assert result.subject
    assert result.source_chunk_ids == ["chunk_002"]


@pytest.mark.asyncio
async def test_presentation_generator(sample_context, sample_request, mock_llm):
    gen = PresentationGenerator()
    assert gen.output_type == OutputType.PRESENTATION
    result = await gen.generate(
        context=sample_context,
        source_chunk_ids=["chunk_001", "chunk_002"],
        request=sample_request,
        llm_client=mock_llm,
    )
    assert isinstance(result, PresentationDeliverable)
    assert result.presentation_title
    assert len(result.slides) >= 1
    assert result.source_chunk_ids == ["chunk_001", "chunk_002"]


@pytest.mark.asyncio
async def test_video_package_generator(sample_context, sample_request, mock_llm):
    gen = VideoPackageGenerator()
    assert gen.output_type == OutputType.VIDEO_PACKAGE
    result = await gen.generate(
        context=sample_context,
        source_chunk_ids=["chunk_001"],
        request=sample_request,
        llm_client=mock_llm,
    )
    assert isinstance(result, VideoPackageDeliverable)
    assert result.video_title
    assert result.hook
    assert len(result.scenes) >= 1
    assert result.source_chunk_ids == ["chunk_001"]


def test_generator_factory_and_registry():
    all_gens = get_all_generators()
    assert len(all_gens) == 5
    assert OutputType.LINKEDIN in all_gens
    assert OutputType.SUMMARY in all_gens
    assert OutputType.ADVISORY in all_gens
    assert OutputType.PRESENTATION in all_gens
    assert OutputType.VIDEO_PACKAGE in all_gens

    # Lookup by type
    assert isinstance(get_generator(OutputType.LINKEDIN), LinkedInGenerator)
    assert isinstance(get_generator(OutputType.SUMMARY), ExecutiveSummaryGenerator)
    assert isinstance(get_generator(OutputType.ADVISORY), OfficialAdvisoryGenerator)
    assert isinstance(get_generator(OutputType.PRESENTATION), PresentationGenerator)
    assert isinstance(get_generator(OutputType.VIDEO_PACKAGE), VideoPackageGenerator)

    # Invalid output type
    with pytest.raises(AppException) as exc_info:
        get_generator("nonexistent_type")  # type: ignore[arg-type]
    assert exc_info.value.code == "UNSUPPORTED_OUTPUT_TYPE"
