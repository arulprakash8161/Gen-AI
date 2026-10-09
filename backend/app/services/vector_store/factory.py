from typing import Optional
from backend.app.services.vector_store.base import BaseVectorStore
from backend.app.services.vector_store.sqlite_store import SQLiteVectorStore
from backend.app.core.config import settings
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_VECTOR_STORE: Optional[BaseVectorStore] = None


def get_vector_store(
    store_type: Optional[str] = None,
    db_path: Optional[str] = None,
    force_new: bool = False,
) -> BaseVectorStore:
    """
    Factory creating or returning the configured vector store instance.
    Defaults to SQLite-backed persistent vector store.
    """
    global _DEFAULT_VECTOR_STORE

    selected = (store_type or "sqlite").lower().strip()

    if selected in ("sqlite", "local", "persistent"):
        target_path = db_path or getattr(settings, "VECTOR_DB_PATH", "data/vector_store.db")
        if force_new or _DEFAULT_VECTOR_STORE is None or db_path is not None:
            instance = SQLiteVectorStore(db_path=target_path)
            if db_path is None and not force_new:
                _DEFAULT_VECTOR_STORE = instance
            return instance
        return _DEFAULT_VECTOR_STORE
    else:
        raise AppException(
            message=f"Unknown vector store type '{selected}'. Supported: 'sqlite'.",
            status_code=400,
            code="UNKNOWN_VECTOR_STORE_TYPE",
        )
