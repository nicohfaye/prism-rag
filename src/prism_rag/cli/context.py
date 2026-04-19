from __future__ import annotations

import typer
from rich.console import Console

from prism_rag.config import Settings

# One shared Console instance so all commands render consistently.
console = Console()


def get_settings(ctx: typer.Context) -> Settings:
    """Pull the already-loaded Settings off the Typer context."""
    return ctx.obj["settings"]


def get_profile(ctx: typer.Context) -> str:
    return ctx.obj["profile"]


def resolve_collection(ctx: typer.Context, override: str | None) -> str:
    """CLI flag wins; otherwise fall back to milvus.default_collection in the active profile."""
    return override or get_settings(ctx).milvus.default_collection


COLLECTION_HELP = (
    "Target collection. Defaults to milvus.default_collection from the active profile."
)
