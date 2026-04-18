from __future__ import annotations

from pathlib import Path

from prism_rag.chunking import Chunker
from prism_rag.config import Settings
from prism_rag.embeddings import build_embedder
from prism_rag.generation import build_generator
from prism_rag.pipeline.ingestion import IngestionPipeline
from prism_rag.pipeline.query import QueryPipeline
from prism_rag.registry import IngestionRegistry
from prism_rag.retrieval import DenseRetriever
from prism_rag.vectorstore import MilvusStore

DEFAULT_REGISTRY_PATH = Path("data/registry.sqlite")


def build_ingestion_pipeline(
    settings: Settings,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> IngestionPipeline:
    return IngestionPipeline(
        chunker=Chunker(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        ),
        embedder=build_embedder(settings.embedding),
        store=MilvusStore(uri=settings.milvus.uri),
        registry=IngestionRegistry(registry_path),
    )


def build_query_pipeline(settings: Settings) -> QueryPipeline:
    embedder = build_embedder(settings.embedding)
    store = MilvusStore(uri=settings.milvus.uri)
    return QueryPipeline(
        retriever=DenseRetriever(embedder=embedder, store=store),
        generator=build_generator(settings.generation),
    )
