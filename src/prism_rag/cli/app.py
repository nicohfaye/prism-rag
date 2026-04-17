from __future__ import annotations

import typer
from rich.console import Console

from prism_rag.logging import configure_logging

app = typer.Typer(
    name="prism",
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
    ctx.obj = {"profile": profile}


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest."),
    collection: str = typer.Option("default", "--collection", "-c"),
) -> None:
    """Ingest a file or directory into a Milvus collection."""
    console.print(f"[yellow]ingest[/] not implemented (path={path}, collection={collection})")


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural-language question."),
    collection: str = typer.Option("default", "--collection", "-c"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
) -> None:
    """Retrieve + generate an answer with citations."""
    console.print(
        f"[yellow]query[/] not implemented "
        f"(q={question!r}, collection={collection}, top_k={top_k}, stream={stream})"
    )


@app.command()
def retrieve(
    question: str = typer.Argument(..., help="Natural-language question."),
    collection: str = typer.Option("default", "--collection", "-c"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON (agent-friendly)."),
) -> None:
    """Retrieve-only. Precursor to the MCP tool surface."""
    console.print(
        f"[yellow]retrieve[/] not implemented "
        f"(q={question!r}, collection={collection}, top_k={top_k}, json={json_out})"
    )


collections_app = typer.Typer(help="Manage Milvus collections.", no_args_is_help=True)
app.add_typer(collections_app, name="collections")


@collections_app.command("list")
def collections_list() -> None:
    """List existing collections."""
    console.print("[yellow]collections list[/] not implemented")


@collections_app.command("create")
def collections_create(name: str = typer.Argument(...)) -> None:
    """Create a new collection."""
    console.print(f"[yellow]collections create[/] not implemented (name={name})")


@collections_app.command("delete")
def collections_delete(name: str = typer.Argument(...)) -> None:
    """Delete a collection."""
    console.print(f"[yellow]collections delete[/] not implemented (name={name})")


eval_app = typer.Typer(help="Run retrieval metrics.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    dataset: str = typer.Argument(..., help="Path to Q&A dataset JSON."),
    collection: str = typer.Option("default", "--collection", "-c"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Run hit-rate@k and MRR against a dataset."""
    console.print(
        f"[yellow]eval run[/] not implemented "
        f"(dataset={dataset}, collection={collection}, top_k={top_k})"
    )


if __name__ == "__main__":
    app()
