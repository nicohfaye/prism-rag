"""SQLite-backed ingestion registry for idempotency."""

from prism_rag.registry.sqlite import FileRecord, IngestionRegistry

__all__ = ["FileRecord", "IngestionRegistry"]
