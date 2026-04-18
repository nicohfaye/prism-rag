from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from prism_rag.config import Settings, load_settings
from prism_rag.embeddings import build_embedder
from prism_rag.logging import configure_logging
from prism_rag.pipeline import (
    DEFAULT_REGISTRY_PATH,
    build_ingestion_pipeline,
    build_query_pipeline,
)
from prism_rag.registry import IngestionRegistry
from prism_rag.vectorstore import MilvusStore

app = typer.Typer(
    name="Prism",
    help="A modular RAG pipeline with pluggable providers.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root(
    ctx: typer.Context,
    profile: str = typer.Option(
        "openai",
        "--profile",
        "-p",
        help="Config profile (reads configs/<profile>.yaml).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    configure_logging(verbose=verbose)
    settings = load_settings(profile)
    ctx.obj = {"profile": profile, "settings": settings}


def _settings(ctx: typer.Context) -> Settings:
    return ctx.obj["settings"]


@app.command()
def ingest(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, help="File or directory to ingest."),  # noqa: B008
    collection: str = typer.Option("default", "--collection", "-c"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-ingest even if the registry says unchanged."
    ),
) -> None:
    """Ingest a file or directory into a Milvus collection (idempotent)."""
    settings = _settings(ctx)
    pipeline = build_ingestion_pipeline(settings, registry_path=DEFAULT_REGISTRY_PATH)
    console.print(f"[cyan]Ingesting[/] {path} → collection='{collection}'")
    result = pipeline.ingest_path(path, collection, force=force)
    console.print(
        f"[green]done[/] processed={result.files_processed} "
        f"ingested={result.files_ingested} "
        f"skipped={result.files_skipped_unchanged} "
        f"failed={result.files_failed} "
        f"chunks={result.chunks_inserted}"
    )
    for file, err in result.errors:
        console.print(f"[red]error[/] {file}: {err}")


@app.command()
def query(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Natural-language question."),
    collection: str = typer.Option("default", "--collection", "-c"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
) -> None:
    """Retrieve + generate an answer with citations."""
    settings = _settings(ctx)
    pipeline = build_query_pipeline(settings)
    chunks, tokens = pipeline.stream(question, collection, top_k)
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


@app.command()
def retrieve(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Natural-language question."),
    collection: str = typer.Option("default", "--collection", "-c"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON (agent-friendly)."),
) -> None:
    """Retrieve-only. Precursor to the MCP tool surface."""
    settings = _settings(ctx)
    pipeline = build_query_pipeline(settings)
    chunks = pipeline.retrieve(question, collection, top_k)
    if json_out:
        payload = [
            {
                "id": c.id,
                "text": c.text,
                "source_path": c.source_path,
                "source_type": c.source_type,
                "chunk_index": c.chunk_index,
                "heading_path": c.heading_path,
                "page": c.page,
                "score": c.score,
            }
            for c in chunks
        ]
        console.print_json(json.dumps(payload))
        return
    if not chunks:
        console.print("[yellow]No chunks retrieved.[/]")
        return
    table = Table(title=f"Top {len(chunks)} chunks")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("text")
    for i, c in enumerate(chunks, 1):
        locator = c.source_path + (f"#p{c.page}" if c.page else f"#c{c.chunk_index}")
        preview = c.text[:160] + ("…" if len(c.text) > 160 else "")
        table.add_row(str(i), f"{c.score:.3f}", locator, preview)
    console.print(table)


collections_app = typer.Typer(help="Manage Milvus collections.", no_args_is_help=True)
app.add_typer(collections_app, name="collections")


@collections_app.command("list")
def collections_list(ctx: typer.Context) -> None:
    """List all collections in the configured Milvus instance."""
    settings = _settings(ctx)
    store = MilvusStore(uri=settings.milvus.uri)
    names = store.list_collections()
    if not names:
        console.print("[dim]no collections[/]")
        return
    for n in names:
        console.print(f"• {n}")


@collections_app.command("create")
def collections_create(
    ctx: typer.Context,
    name: str = typer.Argument(...),
) -> None:
    """Create a collection sized to the configured embedder's dimension."""
    settings = _settings(ctx)
    store = MilvusStore(uri=settings.milvus.uri)
    embedder = build_embedder(settings.embedding)
    store.ensure_collection(name, dimension=embedder.dimension)
    console.print(
        f"[green]collection ready[/] name='{name}' dim={embedder.dimension} "
        f"model='{embedder.model}'"
    )


@collections_app.command("delete")
def collections_delete(
    ctx: typer.Context,
    name: str = typer.Argument(...),
) -> None:
    """Drop a collection and purge its registry entries."""
    settings = _settings(ctx)
    store = MilvusStore(uri=settings.milvus.uri)
    registry = IngestionRegistry(DEFAULT_REGISTRY_PATH)
    store.drop_collection(name)
    removed = registry.delete_collection(name)
    console.print(
        f"[green]deleted[/] '{name}' ({removed} registry record(s) purged)"
    )


eval_app = typer.Typer(help="Run retrieval metrics.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    dataset: str = typer.Argument(..., help="Path to Q&A dataset JSON."),
    collection: str = typer.Option("default", "--collection", "-c"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Run hit-rate@k and MRR (implemented in Phase 5)."""
    console.print(
        f"[yellow]eval run[/] not implemented yet — arrives in Phase 5. "
        f"(dataset={dataset}, collection={collection}, top_k={top_k})"
    )


if __name__ == "__main__":
    app()
