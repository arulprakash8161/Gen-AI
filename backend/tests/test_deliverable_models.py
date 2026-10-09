import pytest
from pydantic import ValidationError
from backend.app.models import (
    OutputType,
    ToneOption,
    AudienceOption,
    DetailLevel,
    TransformationRequest,
    LinkedInPostDeliverable,
    ExecutiveSummaryDeliverable,
    OfficialAdvisoryDeliverable,
    Slide,
    PresentationDeliverable,
    VideoScene,
    VideoPackageDeliverable,
    TransformationResponse,
)


def test_transformation_request_valid():
    req = TransformationRequest(
        document_id="doc_12345",
        output_types=[OutputType.LINKEDIN, OutputType.SUMMARY],
        audience=AudienceOption.EXECUTIVE.value,
        tone=ToneOption.PROFESSIONAL.value,
        language="en",
        detail_level=DetailLevel.STANDARD.value,
        objective="Focus on quarterly revenue growth and regional expansion.",
    )
    assert req.document_id == "doc_12345"
    assert len(req.output_types) == 2
    assert OutputType.LINKEDIN in req.output_types
    assert OutputType.SUMMARY in req.output_types
    assert req.audience == "executive"
    assert req.language == "en"
    assert req.objective is not None


def test_transformation_request_deduplicates_output_types():
    req = TransformationRequest(
        document_id="doc_xyz",
        output_types=[OutputType.LINKEDIN, OutputType.LINKEDIN, OutputType.ADVISORY],
    )
    assert len(req.output_types) == 2
    assert req.output_types == [OutputType.LINKEDIN, OutputType.ADVISORY]


def test_transformation_request_validation_failures():
    # Empty output types
    with pytest.raises(ValidationError):
        TransformationRequest(document_id="doc_1", output_types=[])

    # Blank document_id
    with pytest.raises(ValidationError):
        TransformationRequest(document_id="   ", output_types=[OutputType.SUMMARY])

    # Invalid top_k (< 1)
    with pytest.raises(ValidationError):
        TransformationRequest(
            document_id="doc_1",
            output_types=[OutputType.SUMMARY],
            top_k=0,
        )

    # Invalid top_k (> 20)
    with pytest.raises(ValidationError):
        TransformationRequest(
            document_id="doc_1",
            output_types=[OutputType.SUMMARY],
            top_k=25,
        )


def test_linkedin_deliverable_model():
    post = LinkedInPostDeliverable(
        headline="Major AI Breakthrough in Document Intelligence",
        content="We are thrilled to announce our new automated pipeline...",
        hashtags=["#GenAI", "#FastAPI", "#Innovation"],
        call_to_action="Read the full technical whitepaper below!",
        key_takeaways=["Zero-loss parsing", "Sub-second vector retrieval"],
        source_chunk_ids=["chunk_0", "chunk_1"],
    )
    assert post.headline.startswith("Major AI")
    assert len(post.hashtags) == 3
    assert len(post.source_chunk_ids) == 2


def test_executive_summary_deliverable_model():
    summary = ExecutiveSummaryDeliverable(
        title="Q3 Strategic Document Summary",
        executive_brief="The organization has achieved 34% productivity improvement...",
        key_findings=["Cost reduced by 18%", "Accuracy increased to 99.2%"],
        strategic_implications=["Deployment ready for enterprise rollout"],
        action_items=["Initiate Phase 2 pilot", "Allocate server budget"],
        risk_factors=["Requires local GPU or Ollama daemon running"],
        source_chunk_ids=["chunk_2"],
    )
    assert summary.title == "Q3 Strategic Document Summary"
    assert len(summary.key_findings) == 2
    assert len(summary.action_items) == 2


def test_official_advisory_deliverable_model():
    advisory = OfficialAdvisoryDeliverable(
        reference_number="ADV-2026-004",
        subject="Updated Security Directives for LLM Deployments",
        urgency_level="Critical",
        effective_date="2026-11-01",
        background="Recent audits require strict data isolation and on-premise inference.",
        advisory_details=["All sensitive records must run via local Ollama models."],
        compliance_requirements=["Submit compliance verification before effective date."],
        contact_or_authority="Chief Information Security Officer",
        source_chunk_ids=["chunk_3"],
    )
    assert advisory.reference_number == "ADV-2026-004"
    assert advisory.urgency_level == "Critical"
    assert advisory.contact_or_authority.startswith("Chief Information")


def test_presentation_deliverable_model():
    slide1 = Slide(
        slide_number=1,
        title="Executive Vision",
        bullet_points=["Key mandate", "Strategic objectives", "Target timeline"],
        speaker_notes="Welcome executive board members and introduce key themes.",
        visual_cue="High-contrast hero slide with organizational logo.",
        source_chunk_ids=["chunk_0"],
    )
    slide2 = Slide(
        slide_number=2,
        title="Key Performance Metrics",
        bullet_points=["Throughput: 120 docs/hr", "Latency: 250ms"],
        speaker_notes="Walk through performance benchmarks.",
        source_chunk_ids=["chunk_1"],
    )
    deck = PresentationDeliverable(
        presentation_title="Quarterly Review",
        subtitle="Department of Digital Transformation",
        target_duration_minutes=20,
        slides=[slide1, slide2],
        source_chunk_ids=["chunk_0", "chunk_1"],
    )
    assert deck.presentation_title == "Quarterly Review"
    assert len(deck.slides) == 2
    assert deck.slides[0].slide_number == 1
    assert deck.slides[1].title == "Key Performance Metrics"


def test_video_package_deliverable_model():
    scene1 = VideoScene(
        scene_number=1,
        visual_description="Dynamic drone shot of data center with animated overlay.",
        spoken_script="In an era where data doubles every year, intelligence is everything.",
        duration_seconds=10,
        on_screen_text="Next-Gen Document Intelligence",
    )
    scene2 = VideoScene(
        scene_number=2,
        visual_description="Split-screen showcasing real-time automated processing.",
        spoken_script="Meet the new platform transforming raw documents into high-impact deliverables.",
        duration_seconds=15,
    )
    pkg = VideoPackageDeliverable(
        video_title="Transforming Knowledge Into Action",
        target_duration_seconds=25,
        target_platform="YouTube",
        hook="Stop wasting hours reading 50-page reports manually.",
        scenes=[scene1, scene2],
        outro_cta="Visit our portal to start transforming your documents today.",
        source_chunk_ids=["chunk_0"],
    )
    assert pkg.video_title == "Transforming Knowledge Into Action"
    assert len(pkg.scenes) == 2
    assert pkg.scenes[0].duration_seconds == 10
    assert pkg.outro_cta.startswith("Visit our portal")


def test_transformation_response_model():
    response = TransformationResponse(
        task_id="trans_abc123",
        document_id="doc_789",
        status="completed",
        citations=["[SOURCE CHUNK 0 - Page 1 - Chunk ID: chunk_0]"],
        grounding_score=0.95,
        metadata={"processing_time_ms": 420},
    )
    assert response.task_id == "trans_abc123"
    assert response.status == "completed"
    assert response.linkedin is None
    assert response.grounding_score == 0.95
    assert len(response.citations) == 1

    # Verify JSON serialization works properly
    json_data = response.model_dump()
    assert json_data["task_id"] == "trans_abc123"
    assert "created_at" in json_data
