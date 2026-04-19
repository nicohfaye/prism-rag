from __future__ import annotations

import json

import typer
from rich.table import Table

from prism_rag.cli.context import COLLECTION_HELP, console, get_settings, resolve_collection
from prism_rag.cli.render import locator, preview
from prism_rag.pipeline import build_query_pipeline
from prism_rag.retrieval import RetrievedChunk


def register(app: typer.Typer) -> None:
    app.command()(query)
    app.command()(retrieve)


def query(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Natural-language question."),
    collection: str | None = typer.Option(None, "--collection", "-c", help=COLLECTION_HELP),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
) -> None:
    """Retrieve + generate an answer with citations."""
    settings = get_settings(ctx)
    target = resolve_collection(ctx, collection)
    pipeline = build_query_pipeline(settings)
    chunks, tokens = pipeline.stream(question, target, top_k)
    if not chunks:
        console.print("[yellow]No context retrieved.[/]")
        return
    console.print(f"[dim]retrieved {len(chunks)} chunks[/]\n")
    if stream:
        for token in tokens:
            console.print(token, end="", soft_wrap=True, highlight=False, markup=False)
        console.print()
    else:
        console.print("".join(tokens))


def retrieve(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Natural-language question."),
    collection: str | None = typer.Option(None, "--collection", "-c", help=COLLECTION_HELP),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON (agent-friendly)."),
) -> None:
    """Retrieve-only. Precursor to the MCP tool surface."""
    settings = get_settings(ctx)
    target = resolve_collection(ctx, collection)
    pipeline = build_query_pipeline(settings)
    chunks = pipeline.retrieve(question, target, top_k)

    if json_out:
        console.print_json(json.dumps([_chunk_to_dict(c) for c in chunks]))
        return
    if not chunks:
        console.print("[yellow]No chunks retrieved.[/]")
        return
    console.print(_chunks_table(chunks))


def _chunk_to_dict(c: RetrievedChunk) -> dict:
    return {
        "id": c.id,
        "text": c.text,
        "source_path": c.source_path,
        "source_type": c.source_type,
        "chunk_index": c.chunk_index,
        "heading_path": c.heading_path,
        "page": c.page,
        "score": c.score,
    }


def _chunks_table(chunks: list[RetrievedChunk]) -> Table:
    table = Table(title=f"Top {len(chunks)} chunks")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("text")
    for i, c in enumerate(chunks, 1):
        table.add_row(
            str(i),
            f"{c.score:.3f}",
            locator(c.source_path, c.page, c.chunk_index),
            preview(c.text),
        )
    return table
