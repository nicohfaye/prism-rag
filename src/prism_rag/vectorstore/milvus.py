from __future__ import annotations

import logging
from typing import Any

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from prism_rag.chunking import Chunk

log = logging.getLogger(__name__)

# Field names — kept as constants so retrieval and schema agree.
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


def _build_schema(dimension: int) -> CollectionSchema:
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


class MilvusStore:
    """Thin wrapper over pymilvus MilvusClient (2.4+) with schema + helpers."""

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._client = MilvusClient(uri=uri)

    @property
    def uri(self) -> str:
        return self._uri

    def has_collection(self, name: str) -> bool:
        return self._client.has_collection(name)

    def list_collections(self) -> list[str]:
        return list(self._client.list_collections())

    def drop_collection(self, name: str) -> None:
        if self._client.has_collection(name):
            self._client.drop_collection(name)
            log.info("dropped collection '%s'", name)

    def ensure_collection(self, name: str, dimension: int) -> None:
        if self._client.has_collection(name):
            return
        schema = _build_schema(dimension)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=F_EMBEDDING,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        log.info("created collection '%s' (dim=%d)", name, dimension)

    def insert(
        self,
        collection: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")
        rows: list[dict[str, Any]] = []
        for chunk, vec in zip(chunks, embeddings, strict=True):
            rows.append(
                {
                    F_ID: chunk.id,
                    F_EMBEDDING: vec,
                    F_TEXT: chunk.text,
                    F_SOURCE_PATH: chunk.source_path,
                    F_SOURCE_TYPE: chunk.source_type,
                    F_CONTENT_HASH: chunk.content_hash,
                    F_CHUNK_INDEX: chunk.chunk_index,
                    F_CHUNK_HASH: chunk.chunk_hash,
                    F_HEADING_PATH: chunk.heading_path or "",
                    F_PAGE: chunk.page if chunk.page is not None else -1,
                    F_INGESTED_AT: chunk.ingested_at,
                }
            )
        self._client.insert(collection_name=collection, data=rows)

    def delete_ids(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        self._client.delete(collection_name=collection, ids=ids)

    def search(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self._client.search(
            collection_name=collection,
            data=[embedding],
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
            filter=filter_expr or "",
        )
        return list(results[0]) if results else []
