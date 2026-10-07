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


class ChunkingConfig(BaseModel):
    """
    Configuration parameters for text chunking.
    """
    chunk_size: int = Field(default=600, ge=20, le=4000, description="Target character length per chunk")
    chunk_overlap: int = Field(default=100, ge=0, le=1000, description="Overlapping characters between adjacent chunks")
    min_chunk_size: int = Field(default=20, ge=1, description="Minimum characters for a valid chunk")


class DocumentChunk(BaseModel):
    """
    A discrete chunk of text extracted from a document with full traceability metadata.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. doc_xxx_chunk_0")
    doc_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="0-indexed position within the document")
    text: str = Field(..., description="Cleaned chunk text content")
    page_number: Optional[int] = Field(None, description="Source page or section number if applicable")
    character_count: int = Field(..., description="Number of characters in the chunk")
    token_count: int = Field(..., description="Estimated token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata preserved from parent document")
