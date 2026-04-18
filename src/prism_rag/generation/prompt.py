from __future__ import annotations

from prism_rag.retrieval import RetrievedChunk

SYSTEM_PROMPT = """\
You are a helpful assistant answering questions using only the provided context.

Rules:
- Answer strictly from the context. If the answer is not present, say:
  "I don't know based on the provided context."
- Cite sources inline using the format [source_path#chunk_index] after each claim.
- Be concise.
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for c in chunks:
        header = [f"[{c.source_path}#{c.chunk_index}]"]
        if c.heading_path:
            header.append(f"section: {c.heading_path}")
        if c.page is not None:
            header.append(f"page: {c.page}")
        blocks.append(f"{' '.join(header)}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"Context:\n\n{format_context(chunks)}\n\nQuestion: {question}"
