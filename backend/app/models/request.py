from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class OutputType(str, Enum):
    """Supported content transformation formats."""
    LINKEDIN = "linkedin"
    SUMMARY = "summary"
    ADVISORY = "advisory"
    PRESENTATION = "presentation"
    VIDEO_PACKAGE = "video_package"


class ToneOption(str, Enum):
    """Pre-set tone stylistic options."""
    PROFESSIONAL = "professional"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    PERSUASIVE = "persuasive"
    TECHNICAL = "technical"
    SIMPLIFIED = "simplified"


class AudienceOption(str, Enum):
    """Pre-set audience perspectives."""
    EXECUTIVE = "executive"
    GENERAL_PUBLIC = "general_public"
    TECHNICAL_EXPERTS = "technical_experts"
    DOMAIN_STAKEHOLDERS = "domain_stakeholders"
    STUDENTS = "students"


class DetailLevel(str, Enum):
    """Depth and granularity of generated content."""
    HIGH_LEVEL = "high_level"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class TransformationRequest(BaseModel):
    """
    Client request payload for multi-channel document transformation.
    """
    document_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the already ingested document"
    )
    output_types: List[OutputType] = Field(
        ...,
        min_length=1,
        description="One or more deliverable types to generate"
    )
    audience: str = Field(
        default=AudienceOption.EXECUTIVE.value,
        description="Target audience profile or custom audience string"
    )
    tone: str = Field(
        default=ToneOption.PROFESSIONAL.value,
        description="Stylistic tone or custom tone string"
    )
    language: str = Field(
        default="en",
        description="Target output language code (default 'en')"
    )
    detail_level: str = Field(
        default=DetailLevel.STANDARD.value,
        description="Content depth: high_level, standard, or comprehensive"
    )
    objective: Optional[str] = Field(
        default=None,
        description="Custom focus instructions or key points to highlight"
    )
    custom_query: Optional[str] = Field(
        default=None,
        description="Optional specific search query to guide RAG retrieval"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for number of source chunks retrieved"
    )

    @field_validator("document_id")
    @classmethod
    def validate_doc_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("document_id cannot be blank")
        return v

    @field_validator("output_types")
    @classmethod
    def validate_unique_outputs(cls, v: List[OutputType]) -> List[OutputType]:
        if not v:
            raise ValueError("At least one output_type must be specified")
        seen = set()
        unique = []
        for item in v:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique
