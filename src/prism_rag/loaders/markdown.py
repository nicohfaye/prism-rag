from __future__ import annotations

from pathlib import Path

from prism_rag.loaders.base import LoadedDocument


class MarkdownLoader:
    source_type = "markdown"

    def load(self, path: Path, content_hash: str) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8")
        return [
            LoadedDocument(
                text=text,
                source_path=str(path),
                source_type=self.source_type,
                content_hash=content_hash,
            )
        ]
