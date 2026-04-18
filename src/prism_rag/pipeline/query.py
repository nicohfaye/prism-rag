from __future__ import annotations

import logging
from collections.abc import Iterator

from prism_rag.generation.base import Generator
from prism_rag.retrieval import DenseRetriever, RetrievedChunk

log = logging.getLogger(__name__)


class QueryPipeline:
    """Retrieve-then-generate pipeline."""

    def __init__(self, retriever: DenseRetriever, generator: Generator) -> None:
        self._retriever = retriever
        self._generator = generator

    def retrieve(self, question: str, collection: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self._retriever.retrieve(question, collection, top_k=top_k)

    def stream(
        self, question: str, collection: str, top_k: int = 5
    ) -> tuple[list[RetrievedChunk], Iterator[str]]:
        chunks = self.retrieve(question, collection, top_k)
        return chunks, self._generator.stream(question, chunks)
