from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from prism_rag.retrieval import RetrievedChunk


class Generator(Protocol):
    """Provider-agnostic streaming generation interface."""

    @property
    def model(self) -> str: ...

    def stream(self, question: str, context: list[RetrievedChunk]) -> Iterator[str]: ...
