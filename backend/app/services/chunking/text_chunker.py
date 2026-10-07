from typing import List, Optional, Dict, Any
from backend.app.models.document import ProcessedDocument, DocumentChunk, ChunkingConfig
from backend.app.core.config import settings
from backend.app.core.exceptions import AppException
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]


def split_text_by_separator(text: str, separator: str) -> List[str]:
    """Splits text on a separator, keeping the separator with the preceding piece if appropriate."""
    if not separator:
        return list(text)
    splits = text.split(separator)
    results: List[str] = []
    for idx, s in enumerate(splits):
        if not s and idx != 0 and idx != len(splits) - 1:
            continue
        # Re-attach separator to the split element unless it's the last one or empty
        piece = s if idx == len(splits) - 1 else s + separator
        if piece:
            results.append(piece)
    return results


def recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List[str]:
    """
    Recursively splits text using a hierarchy of natural semantic boundaries:
    paragraphs -> lines -> sentences -> words -> characters.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the first separator present in the text
    chosen_separator = ""
    next_separators = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            chosen_separator = sep
            next_separators = separators[i + 1:]
            break

    splits = split_text_by_separator(text, chosen_separator)

    # Accumulate splits into chunks respecting chunk_size and chunk_overlap
    good_splits: List[str] = []
    for s in splits:
        if len(s) < chunk_size:
            good_splits.append(s)
        else:
            if next_separators:
                # Sub-split oversized segment with remaining finer-grained separators
                sub_splits = recursive_split(s, chunk_size, chunk_overlap, next_separators)
                good_splits.extend(sub_splits)
            else:
                # Hard character split fallback
                for char_idx in range(0, len(s), chunk_size - chunk_overlap):
                    sub = s[char_idx: char_idx + chunk_size]
                    if sub.strip():
                        good_splits.append(sub)

    # Combine good_splits with overlap
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for piece in good_splits:
        piece_len = len(piece)
        if current_length + piece_len <= chunk_size:
            current_chunk.append(piece)
            current_length += piece_len
        else:
            if current_chunk:
                joined = "".join(current_chunk).strip()
                if joined:
                    chunks.append(joined)

                # Rewind current_chunk for overlap
                while current_length > chunk_overlap and len(current_chunk) > 1:
                    popped = current_chunk.pop(0)
                    current_length -= len(popped)

                if len(current_chunk) == 1 and current_length > chunk_overlap:
                    words = current_chunk[0].split(" ")
                    tail_words = []
                    tail_len = 0
                    for w in reversed(words):
                        if not w:
                            continue
                        if tail_len + len(w) + 1 <= chunk_overlap:
                            tail_words.insert(0, w)
                            tail_len += len(w) + 1
                        else:
                            break
                    if tail_words:
                        current_chunk = [" ".join(tail_words) + " "]
                        current_length = len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_length = 0

                # Ensure that retained overlap plus incoming piece does not exceed chunk_size
                while current_length + piece_len > chunk_size and current_chunk:
                    popped = current_chunk.pop(0)
                    current_length -= len(popped)

            current_chunk.append(piece)
            current_length += piece_len

    if current_chunk:
        joined = "".join(current_chunk).strip()
        if joined and (not chunks or chunks[-1] != joined):
            chunks.append(joined)

    return chunks


def chunk_document(
    document: ProcessedDocument,
    config: Optional[ChunkingConfig] = None,
) -> List[DocumentChunk]:
    """
    Chunks a ProcessedDocument into semantic DocumentChunks with full traceability.
    Preserves page numbers, parent metadata, document IDs, and index ordering.
    """
    if not document.text or not document.text.strip():
        raise AppException(
            message=f"Cannot chunk empty document '{document.filename}'.",
            status_code=400,
            code="EMPTY_DOCUMENT_CHUNK",
        )

    active_config = config or ChunkingConfig(
        chunk_size=getattr(settings, "CHUNK_SIZE", 600),
        chunk_overlap=getattr(settings, "CHUNK_OVERLAP", 100),
    )

    if active_config.chunk_overlap >= active_config.chunk_size:
        raise AppException(
            message="chunk_overlap must be strictly less than chunk_size.",
            status_code=400,
            code="INVALID_CHUNKING_CONFIG",
        )

    all_chunks: List[DocumentChunk] = []
    global_index = 0

    # If document has multi-page breakdown, chunk per page to retain precise page citation
    if document.pages and len(document.pages) > 1:
        for page in document.pages:
            if not page.text.strip():
                continue

            page_raw_chunks = recursive_split(
                text=page.text,
                chunk_size=active_config.chunk_size,
                chunk_overlap=active_config.chunk_overlap,
                separators=DEFAULT_SEPARATORS,
            )

            for chunk_text in page_raw_chunks:
                if len(chunk_text) < active_config.min_chunk_size and page_raw_chunks.index(chunk_text) != 0:
                    continue

                token_estimate = max(1, len(chunk_text.split()))
                chunk_meta = {
                    **document.metadata,
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "source_page": page.page_number,
                }

                chunk = DocumentChunk(
                    chunk_id=f"{document.doc_id}_c{global_index}",
                    doc_id=document.doc_id,
                    chunk_index=global_index,
                    text=chunk_text,
                    page_number=page.page_number,
                    character_count=len(chunk_text),
                    token_count=token_estimate,
                    metadata=chunk_meta,
                )
                all_chunks.append(chunk)
                global_index += 1
    else:
        # Single page / flat document chunking
        raw_chunks = recursive_split(
            text=document.text,
            chunk_size=active_config.chunk_size,
            chunk_overlap=active_config.chunk_overlap,
            separators=DEFAULT_SEPARATORS,
        )

        for chunk_text in raw_chunks:
            if len(chunk_text) < active_config.min_chunk_size and raw_chunks.index(chunk_text) != 0:
                continue

            token_estimate = max(1, len(chunk_text.split()))
            chunk_meta = {
                **document.metadata,
                "filename": document.filename,
                "file_type": document.file_type,
                "source_page": 1,
            }

            chunk = DocumentChunk(
                chunk_id=f"{document.doc_id}_c{global_index}",
                doc_id=document.doc_id,
                chunk_index=global_index,
                text=chunk_text,
                page_number=1,
                character_count=len(chunk_text),
                token_count=token_estimate,
                metadata=chunk_meta,
            )
            all_chunks.append(chunk)
            global_index += 1

    logger.info(
        f"Document '{document.filename}' ({document.doc_id}) split into {len(all_chunks)} chunks "
        f"(size={active_config.chunk_size}, overlap={active_config.chunk_overlap})"
    )

    return all_chunks
