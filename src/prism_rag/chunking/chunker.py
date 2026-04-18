from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from prism_rag.loaders.base import LoadedDocument

MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


@dataclass
class Chunk:
    """The unit of storage and retrieval: a slice of text with full provenance."""

    id: str
    text: str
    source_path: str
    source_type: str
    content_hash: str
    chunk_index: int
    chunk_hash: str
    heading_path: str | None = None
    page: int | None = None
    ingested_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Chunker:
    """Structure-aware chunking with a recursive-character fallback.

    - Markdown: split on headers (preserves a ``heading_path`` per chunk),
      then recursively by token count.
    - PDF / other: recursive token-aware split only.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 52) -> None:
        self._recursive = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._md_header = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS,
            strip_headers=False,
        )

    def chunk(self, documents: list[LoadedDocument]) -> list[Chunk]:
        out: list[Chunk] = []
        idx = 0
        for doc in documents:
            for text, heading_path in self._split(doc):
                chunk_hash = _sha256(text)
                out.append(
                    Chunk(
                        id=f"{doc.content_hash[:16]}-{idx}",
                        text=text,
                        source_path=doc.source_path,
                        source_type=doc.source_type,
                        content_hash=doc.content_hash,
                        chunk_index=idx,
                        chunk_hash=chunk_hash,
                        heading_path=heading_path,
                        page=doc.page,
                    )
                )
                idx += 1
        return out

    def _split(self, doc: LoadedDocument) -> list[tuple[str, str | None]]:
        if doc.source_type == "markdown":
            header_docs = self._md_header.split_text(doc.text)
            results: list[tuple[str, str | None]] = []
            for hd in header_docs:
                heading_path = self._format_heading_path(hd.metadata)
                for piece in self._recursive.split_text(hd.page_content):
                    results.append((piece, heading_path))
            return results
        return [(piece, None) for piece in self._recursive.split_text(doc.text)]

    @staticmethod
    def _format_heading_path(metadata: dict) -> str | None:
        parts = [metadata[k] for k in ("h1", "h2", "h3", "h4") if k in metadata]
        return " > ".join(parts) if parts else None
