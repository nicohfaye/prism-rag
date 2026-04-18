from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from prism_rag.loaders.base import DocumentLoader, LoadedDocument
from prism_rag.loaders.markdown import MarkdownLoader
from prism_rag.loaders.pdf import PDFLoader

EXTENSION_LOADERS: dict[str, DocumentLoader] = {
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".pdf": PDFLoader(),
}


class UnsupportedFileTypeError(ValueError):
    pass


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_file(path: Path, content_hash: str | None = None) -> list[LoadedDocument]:
    ext = path.suffix.lower()
    loader = EXTENSION_LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileTypeError(f"No loader registered for extension '{ext}'")
    if content_hash is None:
        content_hash = compute_file_hash(path)
    return loader.load(path, content_hash)


def iter_supported_files(path: Path) -> Iterator[Path]:
    """Yield all supported files under `path`. Works for files or directories."""
    if path.is_file():
        if path.suffix.lower() in EXTENSION_LOADERS:
            yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSION_LOADERS:
            yield p
