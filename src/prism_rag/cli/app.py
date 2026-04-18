from __future__ import annotations

import typer

from prism_rag.cli.commands import collections, eval, ingest, query
from prism_rag.config import load_settings
from prism_rag.logging import configure_logging

app = typer.Typer(
    name="Prism",
    help="A modular RAG pipeline with pluggable providers.",
    no_args_is_help=True,
)


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


ingest.register(app)
query.register(app)
collections.register(app)
eval.register(app)


if __name__ == "__main__":
    app()
