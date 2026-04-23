from __future__ import annotations

import logging
from typing import Any

from pymilvus import MilvusClient

from prism_rag.chunking import Chunk
from prism_rag.vectorstore.encoding import chunk_to_row
from prism_rag.vectorstore.schema import F_EMBEDDING, OUTPUT_FIELDS, build_schema

log = logging.getLogger(__name__)


class MilvusStore:
    """Thin wrapper over pymilvus MilvusClient (2.4+)."""

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._client = MilvusClient(uri=uri)

    @property
    def uri(self) -> str:
        return self._uri

    def has_collection(self, name: str) -> bool:
        return self._client.has_collection(name)  # type: ignore

    def list_collections(self) -> list[str]:
        return list(self._client.list_collections())  # type: ignore

    def drop_collection(self, name: str) -> None:
        if self._client.has_collection(name):
            self._client.drop_collection(name)
            log.info("dropped collection '%s'", name)

    def ensure_collection(self, name: str, dimension: int) -> bool:
        """Create the collection if missing. Returns True if newly created.

        If the collection already exists with a different embedding dimension,
        raises ValueError. Protects against silent corruption when swapping
        embedding providers (e.g., OpenAI 1536 → Ollama 768).
        """
        if self._client.has_collection(name):
            actual = self._collection_dim(name)
            if actual != dimension:
                raise ValueError(
                    f"Collection '{name}' has dim={actual} but the configured "
                    f"embedder produces dim={dimension}. "
                    f"Use a different collection name or drop '{name}' first."
                )
            return False

        schema = build_schema(dimension)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=F_EMBEDDING,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=name, schema=schema, index_params=index_params
        )
        log.info("created collection '%s' (dim=%d)", name, dimension)
        return True

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
        rows = [chunk_to_row(c, e) for c, e in zip(chunks, embeddings, strict=True)]
        self._client.insert(collection_name=collection, data=rows)

    def delete_ids(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        self._client.delete(collection_name=collection, ids=ids)

    def describe_collection(self, collection: str) -> dict[str, Any]:
        return dict(self._client.describe_collection(collection))  # type: ignore

    def _collection_dim(self, collection: str) -> int:
        """Return the embedding-field dimension of an existing collection."""
        description = self.describe_collection(collection)
        for field in description.get("fields", []):
            if field.get("name") == F_EMBEDDING:
                params = field.get("params", {}) or {}
                dim = params.get("dim") or field.get("dim")
                if isinstance(dim, int):
                    return dim
                if isinstance(dim, str) and dim.isdigit():
                    return int(dim)
                break
        raise RuntimeError(f"Could not read embedding dimension for collection '{collection}'.")

    def count(self, collection: str, filter_expr: str = "") -> int:
        # Milvus 2.4 lacks a dedicated count API; use count(*) via query.
        rows = self._client.query(
            collection_name=collection,
            filter=filter_expr,
            output_fields=["count(*)"],
        )
        if not rows:
            return 0
        return int(rows[0].get("count(*)", 0))

    def query(
        self,
        collection: str,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Scalar-only query (no vector search). For browsing stored chunks."""
        return list(
            self._client.query(
                collection_name=collection,
                filter=filter_expr,
                output_fields=output_fields or OUTPUT_FIELDS,
                limit=limit,
                offset=offset,
            )
        )

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
