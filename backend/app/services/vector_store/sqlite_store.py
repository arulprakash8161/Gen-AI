import os
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.services.vector_store.base import (
    BaseVectorStore,
    VectorChunkRecord,
    SimilaritySearchResult,
    VectorStoreStats,
)

logger = get_logger(__name__)


def calculate_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Computes the cosine similarity between two numerical vectors.
    Returns 0.0 if either vector has zero magnitude or dimension mismatch.
    Clamps result within [-1.0, 1.0].
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    denominator = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denominator <= 0.0:
        return 0.0

    similarity = dot / denominator
    return max(-1.0, min(1.0, similarity))


class SQLiteVectorStore(BaseVectorStore):
    """
    Production-grade SQLite-backed persistent vector store.
    Persists embeddings, chunk content, and traceability metadata in an ACID-compliant
    local SQLite database file with zero external dependencies.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or getattr(settings, "VECTOR_DB_PATH", "data/vector_store.db")
        self._is_memory = self.db_path == ":memory:"
        self._memory_conn: Optional[sqlite3.Connection] = None

        if not self._is_memory:
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory:
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", timeout=30.0, check_same_thread=False)
            return self._memory_conn

        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        page_number INTEGER,
                        character_count INTEGER NOT NULL DEFAULT 0,
                        token_count INTEGER NOT NULL DEFAULT 0,
                        metadata TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_vector_chunks_doc_id ON vector_chunks (doc_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_vector_chunks_doc_index ON vector_chunks (doc_id, chunk_index);"
                )
        finally:
            if not self._is_memory:
                conn.close()

    async def add_chunks(self, records: List[VectorChunkRecord]) -> int:
        if not records:
            return 0

        conn = self._get_connection()
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with conn:
                payload = [
                    (
                        r.chunk_id,
                        r.doc_id,
                        r.chunk_index,
                        r.text,
                        json.dumps(r.embedding),
                        r.page_number,
                        r.character_count,
                        r.token_count,
                        json.dumps(r.metadata),
                        now_iso,
                    )
                    for r in records
                ]
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO vector_chunks (
                        chunk_id, doc_id, chunk_index, text, embedding,
                        page_number, character_count, token_count, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    payload,
                )
            logger.info(f"Successfully upserted {len(records)} chunks into SQLiteVectorStore ({self.db_path})")
            return len(records)
        finally:
            if not self._is_memory:
                conn.close()

    async def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        doc_id: Optional[str] = None,
    ) -> List[SimilaritySearchResult]:
        if not query_vector or len(query_vector) == 0:
            return []

        # Return empty list if query vector is all zeros (invalid magnitude)
        if all(x == 0.0 for x in query_vector):
            return []

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if doc_id and doc_id.strip():
                cursor.execute(
                    """
                    SELECT chunk_id, doc_id, chunk_index, text, embedding, page_number, character_count, token_count, metadata
                    FROM vector_chunks
                    WHERE doc_id = ?
                    ORDER BY chunk_index ASC;
                    """,
                    (doc_id.strip(),),
                )
            else:
                cursor.execute(
                    """
                    SELECT chunk_id, doc_id, chunk_index, text, embedding, page_number, character_count, token_count, metadata
                    FROM vector_chunks;
                    """
                )

            rows = cursor.fetchall()
            scored_results: List[SimilaritySearchResult] = []

            for row in rows:
                c_id, d_id, c_idx, text, emb_json, page_num, char_count, tok_count, meta_json = row
                try:
                    stored_embedding = json.loads(emb_json)
                except Exception:
                    continue

                sim_score = calculate_cosine_similarity(query_vector, stored_embedding)

                if score_threshold is not None and sim_score < score_threshold:
                    continue

                try:
                    metadata_dict = json.loads(meta_json)
                except Exception:
                    metadata_dict = {}

                scored_results.append(
                    SimilaritySearchResult(
                        chunk_id=c_id,
                        doc_id=d_id,
                        chunk_index=c_idx,
                        text=text,
                        similarity_score=round(sim_score, 6),
                        page_number=page_num,
                        character_count=char_count,
                        token_count=tok_count,
                        metadata=metadata_dict,
                    )
                )

            # Sort strictly descending by similarity_score
            scored_results.sort(key=lambda item: item.similarity_score, reverse=True)
            return scored_results[: max(1, top_k)]
        finally:
            if not self._is_memory:
                conn.close()

    async def delete_by_doc_id(self, doc_id: str) -> int:
        if not doc_id or not doc_id.strip():
            return 0

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM vector_chunks WHERE doc_id = ?;", (doc_id.strip(),))
                deleted_count = cursor.rowcount
            logger.info(f"Deleted {deleted_count} chunks for doc_id='{doc_id}' from SQLiteVectorStore")
            return deleted_count
        finally:
            if not self._is_memory:
                conn.close()

    async def get_chunks_by_doc_id(self, doc_id: str) -> List[VectorChunkRecord]:
        if not doc_id or not doc_id.strip():
            return []

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT chunk_id, doc_id, chunk_index, text, embedding, page_number, character_count, token_count, metadata
                FROM vector_chunks
                WHERE doc_id = ?
                ORDER BY chunk_index ASC;
                """,
                (doc_id.strip(),),
            )
            rows = cursor.fetchall()
            records: List[VectorChunkRecord] = []

            for row in rows:
                c_id, d_id, c_idx, text, emb_json, page_num, char_count, tok_count, meta_json = row
                records.append(
                    VectorChunkRecord(
                        chunk_id=c_id,
                        doc_id=d_id,
                        chunk_index=c_idx,
                        text=text,
                        embedding=json.loads(emb_json),
                        page_number=page_num,
                        character_count=char_count,
                        token_count=tok_count,
                        metadata=json.loads(meta_json) if meta_json else {},
                    )
                )
            return records
        finally:
            if not self._is_memory:
                conn.close()

    async def get_stats(self) -> VectorStoreStats:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT doc_id) FROM vector_chunks;")
            total_chunks, total_docs = cursor.fetchone()

            cursor.execute("SELECT DISTINCT doc_id FROM vector_chunks ORDER BY doc_id ASC;")
            doc_ids = [r[0] for r in cursor.fetchall()]

            cursor.execute("SELECT embedding FROM vector_chunks LIMIT 1;")
            dim_row = cursor.fetchone()
            dimension = None
            if dim_row and dim_row[0]:
                try:
                    dimension = len(json.loads(dim_row[0]))
                except Exception:
                    pass

            return VectorStoreStats(
                total_chunks=total_chunks or 0,
                total_documents=total_docs or 0,
                document_ids=doc_ids,
                dimension=dimension,
            )
        finally:
            if not self._is_memory:
                conn.close()

    async def clear(self) -> None:
        """
        Clears all records in the store. Useful for testing and resets.
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM vector_chunks;")
        finally:
            if not self._is_memory:
                conn.close()
