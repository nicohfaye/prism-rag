from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from prism_rag.chunking import Chunker
from prism_rag.embeddings.base import Embedder
from prism_rag.loaders import compute_file_hash, iter_supported_files, load_file
from prism_rag.registry import IngestionRegistry
from prism_rag.vectorstore import MilvusStore

log = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    files_processed: int = 0
    files_skipped_unchanged: int = 0
    files_ingested: int = 0
    files_failed: int = 0
    chunks_inserted: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


class IngestionPipeline:
    """Orchestrates: load -> chunk -> embed -> insert, gated by the registry."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: MilvusStore,
        registry: IngestionRegistry,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._registry = registry

    def ingest_path(self, path: Path, collection: str) -> IngestionResult:
        self._store.ensure_collection(collection, dimension=self._embedder.dimension)
        result = IngestionResult()
        for file_path in iter_supported_files(path):
            result.files_processed += 1
            try:
                chunks_inserted = self._ingest_file(file_path, collection)
            except Exception as exc:  # noqa: BLE001 — ingestion must not abort on one file
                log.exception("failed to ingest %s", file_path)
                result.files_failed += 1
                result.errors.append((str(file_path), str(exc)))
                continue
            if chunks_inserted is None:
                result.files_skipped_unchanged += 1
            else:
                result.files_ingested += 1
                result.chunks_inserted += chunks_inserted
        return result

    def _ingest_file(self, file_path: Path, collection: str) -> int | None:
        """Return chunk count on ingest, or None when skipped as unchanged."""
        content_hash = compute_file_hash(file_path)
        existing = self._registry.get(str(file_path), collection)
        if existing and existing.content_hash == content_hash:
            log.info("skip (unchanged): %s", file_path)
            return None

        docs = load_file(file_path, content_hash=content_hash)
        chunks = self._chunker.chunk(docs)
        if not chunks:
            log.warning("no chunks produced for %s", file_path)
            return 0

        embeddings = self._embedder.embed_documents([c.text for c in chunks])

        if existing:
            self._store.delete_ids(collection, existing.chunk_ids)
        self._store.insert(collection, chunks, embeddings)
        self._registry.upsert(
            path=str(file_path),
            collection=collection,
            content_hash=content_hash,
            chunk_ids=[c.id for c in chunks],
        )
        log.info("ingested: %s (%d chunks)", file_path, len(chunks))
        return len(chunks)
