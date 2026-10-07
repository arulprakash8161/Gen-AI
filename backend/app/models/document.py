from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """
    Extracted text and metadata for a single document page or section.
    """
    page_number: int = Field(..., description="1-indexed page or section number")
    text: str = Field(..., description="Cleaned text content of the page")
    character_count: int = Field(default=0, description="Number of characters on this page")


class ProcessedDocument(BaseModel):
    """
    Canonical representation of a processed source document.
    """
    doc_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename or document title")
    file_type: str = Field(..., description="Source format: pdf, docx, or txt")
    total_pages: int = Field(..., description="Total pages/sections extracted")
    text: str = Field(..., description="Full aggregated and normalized document text")
    character_count: int = Field(default=0, description="Total characters in full text")
    pages: List[PageContent] = Field(default_factory=list, description="Per-page extracted text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom document metadata")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of extraction"
    )
