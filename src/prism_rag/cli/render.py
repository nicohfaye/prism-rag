from __future__ import annotations

# Presentation helpers shared across CLI commands. Purely about text rendering —
# storage-layer concerns like sentinel decoding live in prism_rag.vectorstore.

PREVIEW_CHARS = 160


def preview(text: str | None, limit: int = PREVIEW_CHARS) -> str:
    """Truncate with an ellipsis for table cells."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


def locator(source_path: str, page: int | None, chunk_index: int) -> str:
    """Human+agent-friendly locator: `path#pN` when paginated, else `path#cN`."""
    if page is not None:
        return f"{source_path}#p{page}"
    return f"{source_path}#c{chunk_index}"


def meta_cell(heading_path: str | None, page: int | None) -> str:
    """Compact 'heading · page' cell for table views."""
    heading = heading_path or ""
    page_str = f"p{page}" if page is not None else ""
    return " · ".join(x for x in (heading, page_str) if x) or "—"
