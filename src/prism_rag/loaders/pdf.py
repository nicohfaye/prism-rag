from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader

from prism_rag.loaders.base import LoadedDocument


class PDFLoader:
    source_type = "pdf"

    def load(self, path: Path, content_hash: str) -> list[LoadedDocument]:
        raw_docs = PyMuPDFLoader(str(path)).load()
        out: list[LoadedDocument] = []
        for d in raw_docs:
            page = d.metadata.get("page")
            if isinstance(page, int):
                page = page + 1  # PyMuPDF is 0-indexed; humanize to 1-indexed
            out.append(
                LoadedDocument(
                    text=d.page_content,
                    source_path=str(path),
                    source_type=self.source_type,
                    content_hash=content_hash,
                    page=page if isinstance(page, int) else None,
                )
            )
        return out
