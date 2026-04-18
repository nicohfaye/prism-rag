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
