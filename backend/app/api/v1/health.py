from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, status
from pydantic import BaseModel
from backend.app.core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    environment: str
    timestamp: str


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns the health status, project metadata, and current timestamp.",
)
async def get_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Probe",
    description="Readiness check verifying application subsystems.",
)
async def get_readiness() -> Dict[str, Any]:
    return {
        "status": "ready",
        "checks": {
            "api": "ok",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
