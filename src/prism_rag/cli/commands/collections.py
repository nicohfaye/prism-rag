from __future__ import annotations

import json
from typing import Any

import typer
from rich.table import Table

from prism_rag.cli.context import console, get_settings
from prism_rag.cli.render import decode_page, locator, meta_cell, preview
from prism_rag.embeddings import build_embedder
from prism_rag.pipeline import DEFAULT_REGISTRY_PATH
from prism_rag.registry import IngestionRegistry
from prism_rag.vectorstore import MilvusStore

# Fields read from Milvus for the `show` command.
_SHOW_FIELDS = [
    "id",
    "source_path",
    "source_type",
    "chunk_index",
    "heading_path",
    "page",
    "text",
]


def register(app: typer.Typer) -> None:
    sub = typer.Typer(help="Manage Milvus collections.", no_args_is_help=True)
    sub.command("list")(collections_list)
    sub.command("create")(collections_create)
    sub.command("delete")(collections_delete)
    sub.command("info")(collections_info)
    sub.command("show")(collections_show)
    app.add_typer(sub, name="collections")


def _store(ctx: typer.Context) -> MilvusStore:
    return MilvusStore(uri=get_settings(ctx).milvus.uri)


def _require_collection(store: MilvusStore, name: str) -> None:
    if not store.has_collection(name):
        console.print(f"[red]no such collection:[/] {name}")
        raise typer.Exit(code=1)


def collections_list(ctx: typer.Context) -> None:
    """List all collections in the configured Milvus instance."""
    names = _store(ctx).list_collections()
    if not names:
        console.print("[dim]no collections[/]")
        return
    for n in names:
        console.print(f"• {n}")


def collections_create(
    ctx: typer.Context,
    name: str = typer.Argument(...),
) -> None:
    """Create a collection sized to the configured embedder's dimension."""
    settings = get_settings(ctx)
    embedder = build_embedder(settings.embedding)
    _store(ctx).ensure_collection(name, dimension=embedder.dimension)
    console.print(
        f"[green]collection ready[/] name='{name}' dim={embedder.dimension} "
        f"model='{embedder.model}'"
    )


def collections_delete(
    ctx: typer.Context,
    name: str = typer.Argument(...),
) -> None:
    """Drop a collection and purge its registry entries."""
    store = _store(ctx)
    registry = IngestionRegistry(DEFAULT_REGISTRY_PATH)
    store.drop_collection(name)
    removed = registry.delete_collection(name)
    console.print(f"[green]deleted[/] '{name}' ({removed} registry record(s) purged)")


def collections_info(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Collection name."),
) -> None:
    """Summary stats for a collection: total chunks, per-source breakdown, schema."""
    store = _store(ctx)
    _require_collection(store, name)

    total = store.count(name)
    console.print(_info_header(name, total, _embedding_dim(store, name)))

    if total == 0:
        return
    console.print(_sources_table(name, _per_source_counts(store, name, total)))


def collections_show(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Collection name."),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Only show chunks from this source_path."
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max chunks to return."),
    offset: int = typer.Option(0, "--offset", help="Skip the first N chunks."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON (agent-friendly)."),
) -> None:
    """Browse stored chunks (scalar query — no vector search)."""
    store = _store(ctx)
    _require_collection(store, name)

    filter_expr = f'source_path == "{source}"' if source else ""
    rows = store.query(
        name,
        filter_expr=filter_expr,
        output_fields=_SHOW_FIELDS,
        limit=limit,
        offset=offset,
    )
    # Milvus doesn't sort scalar queries; sort client-side for a stable view.
    rows.sort(key=lambda r: (r.get("source_path", ""), int(r.get("chunk_index", 0))))

    if json_out:
        console.print_json(json.dumps([_row_to_dict(r) for r in rows]))
        return
    if not rows:
        console.print("[yellow]no chunks match[/]")
        return
    console.print(_chunks_table(name, rows))


# helpers:


def _embedding_dim(store: MilvusStore, collection: str) -> int | str:
    for field in store.describe_collection(collection).get("fields", []):
        if field.get("name") == "embedding":
            params = field.get("params", {}) or {}
            return params.get("dim", field.get("dim", "?"))
    return "?"


def _info_header(name: str, total: int, dim: int | str) -> Table:
    header = Table.grid(padding=(0, 2))
    header.add_row("[bold]collection[/]", name)
    header.add_row("[bold]total chunks[/]", str(total))
    header.add_row("[bold]embedding dim[/]", str(dim))
    return header


def _per_source_counts(
    store: MilvusStore, collection: str, total: int
) -> dict[tuple[str, str], int]:
    # No GROUP BY in Milvus 2.4 — pull source_path for every chunk and aggregate here.
    rows = store.query(
        collection,
        output_fields=["source_path", "source_type"],
        limit=max(total, 1),
    )
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r.get("source_path", ""), r.get("source_type", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _sources_table(collection: str, counts: dict[tuple[str, str], int]) -> Table:
    table = Table(title=f"Sources in '{collection}'")
    table.add_column("source_path")
    table.add_column("type")
    table.add_column("chunks", justify="right")
    for (path, stype), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        table.add_row(path, stype, str(n))
    return table


def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "source_path": row.get("source_path"),
        "source_type": row.get("source_type"),
        "chunk_index": row.get("chunk_index"),
        "heading_path": row.get("heading_path") or None,
        "page": decode_page(row.get("page")),
        "text": row.get("text"),
    }


def _chunks_table(collection: str, rows: list[dict[str, Any]]) -> Table:
    table = Table(title=f"{len(rows)} chunk(s) in '{collection}'")
    table.add_column("#", justify="right")
    table.add_column("source")
    table.add_column("heading / page")
    table.add_column("text")
    for r in rows:
        page = decode_page(r.get("page"))
        chunk_index = int(r.get("chunk_index", 0))
        table.add_row(
            str(chunk_index),
            locator(r.get("source_path", ""), page, chunk_index),
            meta_cell(r.get("heading_path") or None, page),
            preview(r.get("text")),
        )
    return table
