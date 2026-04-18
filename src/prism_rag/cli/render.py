from __future__ import annotations

# Rendering helpers shared across commands.

PREVIEW_CHARS = 160


def preview(text: str | None, limit: int = PREVIEW_CHARS) -> str:
    """Truncate with an ellipsis for table cells."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


def decode_page(page: int | None) -> int | None:
    """Milvus stores page=-1 as a sentinel for 'no page'. Convert back to None."""
    if page is None or page == -1:
        return None
    return page


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
