"""Document loaders (markdown, PDF) behind a common interface."""

from prism_rag.loaders.base import DocumentLoader, LoadedDocument
from prism_rag.loaders.dispatcher import (
    EXTENSION_LOADERS,
    UnsupportedFileTypeError,
    compute_file_hash,
    iter_supported_files,
    load_file,
)
from prism_rag.loaders.markdown import MarkdownLoader
from prism_rag.loaders.pdf import PDFLoader

__all__ = [
    "DocumentLoader",
    "LoadedDocument",
    "MarkdownLoader",
    "PDFLoader",
    "EXTENSION_LOADERS",
    "UnsupportedFileTypeError",
    "compute_file_hash",
    "iter_supported_files",
    "load_file",
]
