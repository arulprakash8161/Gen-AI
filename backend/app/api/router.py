from fastapi import APIRouter
from backend.app.api.v1 import health

api_router = APIRouter()

# Include version 1 endpoints
api_router.include_router(health.router)
