import os
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core App Info
    PROJECT_NAME: str = "AI Content Transformation Platform"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    # Storage Paths
    DATA_DIR: str = "data"
    UPLOADS_DIR: str = "data/uploads"
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    VECTOR_DB_PATH: str = "data/vector_store.db"

    # Ollama LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT_SECONDS: int = 120

    # Embedding Configuration
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Chunking Configuration
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100


settings = Settings()
