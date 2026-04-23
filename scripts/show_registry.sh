#!/usr/bin/env bash

set -euo pipefail

DB_PATH="./data/registry.sqlite"

if ! command -v sqlite3 >/dev/null 2>&1; then
	echo "Error: sqlite3 is not installed or not in PATH." >&2
	exit 1
fi

if [[ ! -f "$DB_PATH" ]]; then
	echo "Error: registry database not found at $DB_PATH" >&2
	exit 1
fi

sqlite3 "$DB_PATH" <<'SQL'
.mode box
.headers on
SELECT * FROM ingested_files;
SQL
