from __future__ import annotations

from pathlib import Path

import typer

from prism_rag.cli.context import COLLECTION_HELP, console, get_settings, resolve_collection
from prism_rag.pipeline import DEFAULT_REGISTRY_PATH, build_ingestion_pipeline


def register(app: typer.Typer) -> None:
    app.command()(ingest)


def ingest(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, help="File or directory to ingest."),  # noqa: B008
    collection: str | None = typer.Option(None, "--collection", "-c", help=COLLECTION_HELP),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-ingest even if the registry says unchanged."
    ),
) -> None:
    """Ingest a file or directory into a Milvus collection (idempotent)."""
    settings = get_settings(ctx)
    target = resolve_collection(ctx, collection)
    pipeline = build_ingestion_pipeline(settings, registry_path=DEFAULT_REGISTRY_PATH)
    console.print(f"[cyan]Ingesting[/] {path} → collection='{target}'")
    result = pipeline.ingest_path(path, target, force=force)
    console.print(
        f"[green]done[/] processed={result.files_processed} "
        f"ingested={result.files_ingested} "
        f"skipped={result.files_skipped_unchanged} "
        f"failed={result.files_failed} "
        f"chunks={result.chunks_inserted}"
    )
    for file, err in result.errors:
        console.print(f"[red]error[/] {file}: {err}")
