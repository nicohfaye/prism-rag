from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class LoadedDocument:
    """A single document (or page) loaded from disk, pre-chunking."""

    text: str
    source_path: str
    source_type: str  # "markdown" | "pdf"
    content_hash: str  # sha256 hex of the ORIGINAL file bytes; drives idempotency
    page: int | None = None  # 1-indexed; None for non-paginated types


class DocumentLoader(Protocol):
    source_type: str

    def load(self, path: Path, content_hash: str) -> list[LoadedDocument]: ...
