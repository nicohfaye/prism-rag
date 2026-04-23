from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from prism_rag.embeddings.base import Embedder
from prism_rag.vectorstore import (
    F_CHUNK_INDEX,
    F_HEADING_PATH,
    F_ID,
    F_PAGE,
    F_SOURCE_PATH,
    F_SOURCE_TYPE,
    F_TEXT,
    MilvusStore,
    decode_heading,
    decode_page,
)

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source_path: str
    source_type: str
    chunk_index: int
    heading_path: str | None
    page: int | None
    score: float


class DenseRetriever:
    """Dense vector retrieval over a Milvus collection."""

    def __init__(self, embedder: Embedder, store: MilvusStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, collection: str, top_k: int = 5) -> list[RetrievedChunk]:
        embedding = self._embedder.embed_query(question)
        raw = self._store.search(collection=collection, embedding=embedding, top_k=top_k)
        return [_to_chunk(r) for r in raw]


def _to_chunk(row: dict[str, Any]) -> RetrievedChunk:
    # MilvusClient returns {"id": ..., "distance": ..., "entity": {...}}.
    entity = row.get("entity") or row
    return RetrievedChunk(
        id=str(row.get("id") or entity.get(F_ID, "")),
        text=entity.get(F_TEXT, ""),
        source_path=entity.get(F_SOURCE_PATH, ""),
        source_type=entity.get(F_SOURCE_TYPE, ""),
        chunk_index=int(entity.get(F_CHUNK_INDEX, 0)),
        heading_path=decode_heading(entity.get(F_HEADING_PATH)),
        page=decode_page(entity.get(F_PAGE)),
        score=float(row.get("distance", 0.0)),
    )
