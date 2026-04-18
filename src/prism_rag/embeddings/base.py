from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Provider-agnostic embedding interface."""

    @property
    def dimension(self) -> int: ...

    @property
    def model(self) -> str: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
