from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging, get_logger
from backend.app.core.exceptions import (
    AppException,
    app_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)
from backend.app.api.router import api_router

# Initialize structured logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan events for application startup and shutdown.
    """
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


def create_app() -> FastAPI:
    """
    Application factory for the FastAPI backend.
    """
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="AI-powered Content Transformation Platform API for Smart India Hackathon",
        docs_url=f"{settings.API_V1_PREFIX}/docs" if settings.DEBUG else None,
        redoc_url=f"{settings.API_V1_PREFIX}/redoc" if settings.DEBUG else None,
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Configure CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Exception Handlers
    application.add_exception_handler(AppException, app_exception_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)

    # Include API Routers
    application.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Top-level convenience route for /health
    @application.get("/health", tags=["Health"], include_in_schema=False)
    async def root_health():
        return {
            "status": "ok",
            "project": settings.PROJECT_NAME,
            "version": settings.VERSION,
        }

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
