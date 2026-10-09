from typing import Any, Dict, List
from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from backend.app.models.request import (
    TransformationRequest,
    OutputType,
    ToneOption,
    AudienceOption,
    DetailLevel,
)
from backend.app.models.deliverable import TransformationResponse
from backend.app.services.transformation.factory import get_transformation_orchestrator
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/transform", tags=["Transformations"])


class OptionItem(BaseModel):
    value: str
    label: str


class TransformationOptionsResponse(BaseModel):
    """
    Returns pre-configured configuration options and valid enumeration values.
    """
    output_types: List[OptionItem]
    audiences: List[OptionItem]
    tones: List[OptionItem]
    detail_levels: List[OptionItem]


@router.post(
    "",
    response_model=TransformationResponse,
    status_code=status.HTTP_200_OK,
    summary="Transform document into multi-channel deliverables",
    description=(
        "Executes strict RAG context retrieval for the specified document and generates "
        "one or more grounded deliverables (LinkedIn post, Executive Summary, Official Advisory, "
        "Slide deck, Video script package) conforming to structured schemas."
    ),
)
async def transform_document(request: TransformationRequest) -> TransformationResponse:
    logger.info(
        f"Received transformation request for doc_id='{request.document_id}' "
        f"with output_types={[ot.value for ot in request.output_types]}"
    )
    orchestrator = get_transformation_orchestrator()
    response = await orchestrator.transform(request)
    return response


@router.get(
    "/options",
    response_model=TransformationOptionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get transformation options metadata",
    description="Returns list of available output types, audiences, tones, and detail levels for UI forms.",
)
async def get_transformation_options() -> TransformationOptionsResponse:
    output_types = [
        OptionItem(value=ot.value, label=ot.value.replace("_", " ").title())
        for ot in OutputType
    ]
    audiences = [
        OptionItem(value=ao.value, label=ao.value.replace("_", " ").title())
        for ao in AudienceOption
    ]
    tones = [
        OptionItem(value=to.value, label=to.value.replace("_", " ").title())
        for to in ToneOption
    ]
    detail_levels = [
        OptionItem(value=dl.value, label=dl.value.replace("_", " ").title())
        for dl in DetailLevel
    ]

    return TransformationOptionsResponse(
        output_types=output_types,
        audiences=audiences,
        tones=tones,
        detail_levels=detail_levels,
    )
