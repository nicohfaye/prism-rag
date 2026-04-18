"""Ingestion and query pipeline assembly."""

from prism_rag.pipeline.factory import (
    DEFAULT_REGISTRY_PATH,
    build_ingestion_pipeline,
    build_query_pipeline,
)
from prism_rag.pipeline.ingestion import IngestionPipeline, IngestionResult
from prism_rag.pipeline.query import QueryPipeline

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "IngestionPipeline",
    "IngestionResult",
    "QueryPipeline",
    "build_ingestion_pipeline",
    "build_query_pipeline",
]
