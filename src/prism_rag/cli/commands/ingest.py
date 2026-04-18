from __future__ import annotations

from pathlib import Path

import typer

from prism_rag.cli.context import console, get_settings
from prism_rag.pipeline import DEFAULT_REGISTRY_PATH, build_ingestion_pipeline


def register(app: typer.Typer) -> None:
    app.command()(ingest)


def ingest(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, help="File or directory to ingest."),  # noqa: B008
    collection: str = typer.Option("default", "--collection", "-c"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-ingest even if the registry says unchanged."
    ),
) -> None:
    """Ingest a file or directory into a Milvus collection (idempotent)."""
    settings = get_settings(ctx)
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
