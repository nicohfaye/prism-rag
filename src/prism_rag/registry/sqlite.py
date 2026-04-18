from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS ingested_files (
    path          TEXT NOT NULL,
    collection    TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    chunk_ids     TEXT NOT NULL,
    ingested_at   TEXT NOT NULL,
    PRIMARY KEY (path, collection)
);
"""


@dataclass(frozen=True)
class FileRecord:
    path: str
    collection: str
    content_hash: str
    chunk_ids: list[str]
    ingested_at: str


class IngestionRegistry:
    """SQLite-backed idempotency store.

    Tracks which files have been ingested into which collection, with the
    hash of their content and the set of chunk ids produced. Used by the
    ingestion pipeline to skip unchanged files and to clean up old chunks
    when a file is re-ingested.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, path: str, collection: str) -> FileRecord | None:
        row = self._conn.execute(
            "SELECT path, collection, content_hash, chunk_ids, ingested_at "
            "FROM ingested_files WHERE path = ? AND collection = ?",
            (path, collection),
        ).fetchone()
        if row is None:
            return None
        return FileRecord(
            path=row[0],
            collection=row[1],
            content_hash=row[2],
            chunk_ids=json.loads(row[3]),
            ingested_at=row[4],
        )

    def upsert(
        self,
        path: str,
        collection: str,
        content_hash: str,
        chunk_ids: list[str],
    ) -> FileRecord:
        record = FileRecord(
            path=path,
            collection=collection,
            content_hash=content_hash,
            chunk_ids=list(chunk_ids),
            ingested_at=datetime.now(UTC).isoformat(),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO ingested_files "
            "(path, collection, content_hash, chunk_ids, ingested_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.path,
                record.collection,
                record.content_hash,
                json.dumps(record.chunk_ids),
                record.ingested_at,
            ),
        )
        self._conn.commit()
        return record

    def delete(self, path: str, collection: str) -> None:
        self._conn.execute(
            "DELETE FROM ingested_files WHERE path = ? AND collection = ?",
            (path, collection),
        )
        self._conn.commit()

    def delete_collection(self, collection: str) -> int:
        """Purge all records for a collection. Returns rows deleted."""
        cursor = self._conn.execute(
            "DELETE FROM ingested_files WHERE collection = ?",
            (collection,),
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()
