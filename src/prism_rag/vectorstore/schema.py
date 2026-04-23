from __future__ import annotations

from pymilvus import CollectionSchema, DataType, FieldSchema

# Field names — the one source of truth. Imported by schema, encoding,
# retrieval, and CLI rendering.
F_ID = "id"
F_EMBEDDING = "embedding"
F_TEXT = "text"
F_SOURCE_PATH = "source_path"
F_SOURCE_TYPE = "source_type"
F_CONTENT_HASH = "content_hash"
F_CHUNK_INDEX = "chunk_index"
F_CHUNK_HASH = "chunk_hash"
F_HEADING_PATH = "heading_path"
F_PAGE = "page"
F_INGESTED_AT = "ingested_at"

# Scalar fields returned by search / query. Everything except the embedding
# vector — embeddings are big and never useful at the retrieval boundary.
OUTPUT_FIELDS = [
    F_ID,
    F_TEXT,
    F_SOURCE_PATH,
    F_SOURCE_TYPE,
    F_CONTENT_HASH,
    F_CHUNK_INDEX,
    F_HEADING_PATH,
    F_PAGE,
    F_INGESTED_AT,
]


def build_schema(dimension: int) -> CollectionSchema:
    """Build the CollectionSchema for a chunk collection with `dimension`-d vectors."""
    fields = [
        FieldSchema(F_ID, DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(F_EMBEDDING, DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(F_TEXT, DataType.VARCHAR, max_length=8192),
        FieldSchema(F_SOURCE_PATH, DataType.VARCHAR, max_length=1024),
        FieldSchema(F_SOURCE_TYPE, DataType.VARCHAR, max_length=32),
        FieldSchema(F_CONTENT_HASH, DataType.VARCHAR, max_length=128),
        FieldSchema(F_CHUNK_INDEX, DataType.INT64),
        FieldSchema(F_CHUNK_HASH, DataType.VARCHAR, max_length=128),
        FieldSchema(F_HEADING_PATH, DataType.VARCHAR, max_length=512),
        FieldSchema(F_PAGE, DataType.INT64),
        FieldSchema(F_INGESTED_AT, DataType.VARCHAR, max_length=64),
    ]
    return CollectionSchema(fields=fields, description="Prism-RAG chunk store")
