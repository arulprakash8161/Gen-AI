from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class LinkedInPostDeliverable(BaseModel):
    """
    Structured LinkedIn post designed for high professional engagement.
    """
    headline: str = Field(..., description="Compelling hook/headline opening the post")
    content: str = Field(..., description="Main body of the LinkedIn post with professional formatting")
    hashtags: List[str] = Field(default_factory=list, description="Relevant hashtags (e.g. #GovTech, #AI)")
    call_to_action: str = Field(..., description="Concluding engagement prompt or call to action")
    key_takeaways: List[str] = Field(default_factory=list, description="Key bullet takeaways highlighted")
    source_chunk_ids: List[str] = Field(default_factory=list, description="IDs of source chunks grounding this output")


class ExecutiveSummaryDeliverable(BaseModel):
    """
    High-density executive briefing for leadership and decision-makers.
    """
    title: str = Field(..., description="Executive briefing title")
    executive_brief: str = Field(..., description="High-level TL;DR summary synthesis")
    key_findings: List[str] = Field(default_factory=list, description="Core facts, findings, and figures extracted")
    strategic_implications: List[str] = Field(default_factory=list, description="Strategic organizational or policy impacts")
    action_items: List[str] = Field(default_factory=list, description="Recommended next steps or decisions required")
    risk_factors: List[str] = Field(default_factory=list, description="Key risks, blockers, or caveats identified")
    source_chunk_ids: List[str] = Field(default_factory=list, description="IDs of source chunks grounding this summary")


class OfficialAdvisoryDeliverable(BaseModel):
    """
    Formal, compliance-oriented public or institutional advisory.
    """
    reference_number: str = Field(..., description="Official bulletin or circular reference code")
    subject: str = Field(..., description="Formal subject line of the advisory")
    urgency_level: str = Field(default="Standard", description="Urgency: Informational, Important, or Critical")
    effective_date: str = Field(..., description="Date or timeframe the advisory takes effect")
    background: str = Field(..., description="Context and justification for the notification")
    advisory_details: List[str] = Field(default_factory=list, description="Specific instructions, mandates, or guidelines")
    compliance_requirements: List[str] = Field(default_factory=list, description="Actions required by affected parties")
    contact_or_authority: str = Field(..., description="Issuing entity, authority, or contact information")
    source_chunk_ids: List[str] = Field(default_factory=list, description="IDs of source chunks grounding this advisory")


class Slide(BaseModel):
    """
    Individual presentation slide representation.
    """
    slide_number: int = Field(..., description="1-indexed sequence position of the slide")
    title: str = Field(..., description="Slide title")
    bullet_points: List[str] = Field(default_factory=list, description="Concise bullet points for the slide body")
    speaker_notes: str = Field(default="", description="Talking points and narration for the presenter")
    visual_cue: Optional[str] = Field(None, description="Suggested diagram, chart, or visual asset description")
    source_chunk_ids: List[str] = Field(default_factory=list, description="Source chunks informing this slide")


class PresentationDeliverable(BaseModel):
    """
    Complete slide deck structure ready for presentation or PPTX export.
    """
    presentation_title: str = Field(..., description="Overall deck title")
    subtitle: Optional[str] = Field(None, description="Deck subtitle or presenter context")
    target_duration_minutes: int = Field(default=15, description="Estimated delivery duration in minutes")
    slides: List[Slide] = Field(default_factory=list, description="Ordered collection of slides")
    source_chunk_ids: List[str] = Field(default_factory=list, description="Aggregated source chunks grounding the deck")


class VideoScene(BaseModel):
    """
    A single scene in an explainer or promotional video package.
    """
    scene_number: int = Field(..., description="1-indexed scene number")
    visual_description: str = Field(..., description="Visual action, B-roll, camera movement, or animation")
    spoken_script: str = Field(..., description="Exact voiceover or dialogue spoken during this scene")
    duration_seconds: int = Field(default=15, description="Estimated scene duration in seconds")
    on_screen_text: Optional[str] = Field(None, description="Text overlays, lower thirds, or key title cards")


class VideoPackageDeliverable(BaseModel):
    """
    Full multimedia production storyboard and script package.
    """
    video_title: str = Field(..., description="Title of the video package")
    target_duration_seconds: int = Field(default=60, description="Total target duration in seconds")
    target_platform: str = Field(default="Corporate Briefing", description="Platform format: YouTube, Shorts, Webinar, etc.")
    hook: str = Field(..., description="First 3-5 seconds high-impact hook statement")
    scenes: List[VideoScene] = Field(default_factory=list, description="Sequenced scene storyboard")
    outro_cta: str = Field(..., description="Closing call to action and outro message")
    source_chunk_ids: List[str] = Field(default_factory=list, description="Source chunks grounding this script")


class TransformationResponse(BaseModel):
    """
    Unified response containing all requested deliverables, RAG citations, and grounding score.
    """
    task_id: str = Field(..., description="Unique ID for this transformation job")
    document_id: str = Field(..., description="Source document identifier")
    status: str = Field(default="completed", description="Status: completed, partial, or failed")
    linkedin: Optional[LinkedInPostDeliverable] = Field(None, description="Generated LinkedIn post deliverable")
    summary: Optional[ExecutiveSummaryDeliverable] = Field(None, description="Generated executive summary deliverable")
    advisory: Optional[OfficialAdvisoryDeliverable] = Field(None, description="Generated official advisory deliverable")
    presentation: Optional[PresentationDeliverable] = Field(None, description="Generated presentation slide deck")
    video_package: Optional[VideoPackageDeliverable] = Field(None, description="Generated video storyboard package")
    citations: List[str] = Field(default_factory=list, description="Formatted citation headers for all chunks utilized")
    grounding_score: float = Field(default=1.0, description="Confidence metric of source grounding (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata, timings, and token metrics")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of deliverable creation"
    )
