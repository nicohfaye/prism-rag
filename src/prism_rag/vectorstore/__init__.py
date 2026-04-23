"""Milvus client wrapper, schema, and encoding helpers."""

from prism_rag.vectorstore.encoding import chunk_to_row, decode_heading, decode_page
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
    OUTPUT_FIELDS,
    build_schema,
)
from prism_rag.vectorstore.store import MilvusStore

__all__ = [
    "F_CHUNK_HASH",
    "F_CHUNK_INDEX",
    "F_CONTENT_HASH",
    "F_EMBEDDING",
    "F_HEADING_PATH",
    "F_ID",
    "F_INGESTED_AT",
    "F_PAGE",
    "F_SOURCE_PATH",
    "F_SOURCE_TYPE",
    "F_TEXT",
    "MilvusStore",
    "OUTPUT_FIELDS",
    "build_schema",
    "chunk_to_row",
    "decode_heading",
    "decode_page",
]
