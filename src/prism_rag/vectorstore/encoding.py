from __future__ import annotations

from typing import Any

from prism_rag.chunking import Chunk
from prism_rag.vectorstore.schema import (
    F_CHUNK_HASH,
    F_CHUNK_INDEX,
    F_CONTENT_HASH,
    F_EMBEDDING,
    F_HEADING_PATH,
    F_ID,
    F_INGESTED_AT,
    F_PAGE,
    F_SOURCE_PATH,
    F_SOURCE_TYPE,
    F_TEXT,
)

# Milvus scalar fields don't support NULL; these sentinels stand in for None.
_PAGE_SENTINEL = -1
_HEADING_SENTINEL = ""


def chunk_to_row(chunk: Chunk, embedding: list[float]) -> dict[str, Any]:
    """Encode a Chunk and its embedding as a Milvus insert row."""
    return {
        F_ID: chunk.id,
        F_EMBEDDING: embedding,
        F_TEXT: chunk.text,
        F_SOURCE_PATH: chunk.source_path,
        F_SOURCE_TYPE: chunk.source_type,
        F_CONTENT_HASH: chunk.content_hash,
        F_CHUNK_INDEX: chunk.chunk_index,
        F_CHUNK_HASH: chunk.chunk_hash,
        F_HEADING_PATH: chunk.heading_path or _HEADING_SENTINEL,
        F_PAGE: chunk.page if chunk.page is not None else _PAGE_SENTINEL,
        F_INGESTED_AT: chunk.ingested_at,
    }


def decode_page(raw: Any) -> int | None:
    """Milvus stores page=-1 when absent. Convert back to None."""
    if raw is None or raw == _PAGE_SENTINEL:
        return None
    return int(raw)


def decode_heading(raw: Any) -> str | None:
    """Milvus stores heading_path='' when absent. Convert back to None."""
    if raw is None or raw == _HEADING_SENTINEL:
        return None
    return str(raw)
