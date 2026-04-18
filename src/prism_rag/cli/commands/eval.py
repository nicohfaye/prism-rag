from __future__ import annotations

import typer

from prism_rag.cli.context import console


def register(app: typer.Typer) -> None:
    sub = typer.Typer(help="Run retrieval metrics.", no_args_is_help=True)
    sub.command("run")(eval_run)
    app.add_typer(sub, name="eval")


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
