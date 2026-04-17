# Prism-RAG — Project Spec

A modular, containerized RAG pipeline built as a personal learning project, rooted in modern best practices. Each pipeline stage (load, chunk, embed, store, retrieve, generate) is independently usable and composable into a full pipeline.

## Goals & non-goals

**Goals**
- Production-quality patterns: SRP, provider-swappable components, clear module boundaries, typed config, idempotent ingestion, observable execution.
- CLI-first human interface; retrieval surface designed to later expose over MCP for agent use.
- Provider pluggability: embedding and generation models swappable between OpenAI and Ollama via config.

**Non-goals (v1)**
- Multi-tenancy, auth, or user management.
- Distributed deployment (single-host Docker Compose is enough).
- Neo4j / GraphRAG (deferred to a later phase).
- Cloud / remote ingestion sources (local filesystem only).
- Generation-quality evaluation (retrieval metrics only).

## Tech stack

| Concern | Choice |
|---|---|
| Language / runtime | Python 3.11+ |
| Package manager | `uv` (astral) |
| Orchestration framework | LangChain |
| Vector DB | Milvus (standalone, v2.4+ for hybrid search) |
| Embeddings | OpenAI `text-embedding-3-small/large` **or** Ollama (`nomic-embed-text` / `mxbai-embed-large`) |
| Generation | OpenAI (`gpt-4o` family) **or** Ollama `gpt-oss:20b` |
| Vision captioning | Same provider abstraction (OpenAI vision or Ollama vision model) |
| Reranker | `bge-reranker-v2-m3` (local) — alt: Cohere Rerank |
| Idempotency store | SQLite sidecar |
| CLI | `typer` + `rich` |
| Config | `pydantic-settings` + YAML profiles |
| Logging | stdlib structured logging + `rich`; optional LangSmith toggle |
| Containerization | Docker + Docker Compose |

## Architecture overview

```
                         ┌────────────────────────────┐
                         │            CLI             │
                         │  ingest / query / retrieve │
                         │   collections / eval       │
                         └─────────────┬──────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
  ┌───────────┐                ┌───────────────┐              ┌──────────────┐
  │ Ingestion │                │   Retrieval   │              │  Evaluation  │
  │ pipeline  │                │    pipeline   │              │  (retrieval  │
  │           │                │               │              │   metrics)   │
  └─────┬─────┘                └───────┬───────┘              └──────┬───────┘
        │                              │                             │
        ▼                              ▼                             │
┌───────────────┐              ┌────────────────┐                    │
│  Loader →     │              │ Dense + BM25   │                    │
│  Chunker →    │              │ → Rerank →     │                    │
│  Embedder →   │              │ Generate       │                    │
│  VectorStore  │              │ (streaming)    │                    │
└──────┬────────┘              └────────┬───────┘                    │
       │                                │                            │
       │                                │                            │
       ▼                                ▼                            ▼
   Milvus + SQLite registry        Milvus + LLM              Q&A dataset (JSON)
```

## Document support (v1)

| Type | Loader | Notes |
|---|---|---|
| Markdown | `UnstructuredMarkdownLoader` / markdown-aware splitter | Preserves headings as metadata |
| PDF | `PyMuPDFLoader` (layout-aware) | Page numbers tracked as metadata |
| HTML | `UnstructuredHTMLLoader` / `BeautifulSoup` cleanup | |
| Images | Vision LLM captioning → text | Single modality (text embeddings only) |

## Chunking

- Structure-aware where possible (markdown headers, PDF sections).
- Recursive character splitter fallback for unstructured text.
- Target chunk size: ~512 tokens, 10% overlap (tunable via config).
- Per-chunk metadata: `source_path`, `source_type`, `page` (PDF), `heading_path` (MD), `chunk_index`, `content_hash`, `ingested_at`.

## Retrieval

- **Hybrid**: dense (embedding ANN) + sparse (BM25) via Milvus 2.4 hybrid search.
- **Reranker**: `bge-reranker-v2-m3` (cross-encoder) reorders top-k candidates.
- Default flow: `retrieve top-50 hybrid → rerank → keep top-5 for generation`.
- Metadata filtering supported at query time (e.g., filter by `source_type` or `collection`).

## Generation

- Streaming tokens to CLI stdout.
- Prompt template injects retrieved chunks with inline citations (`[source_path#chunk_index]`).
- Provider swap via config — same `Generator` interface for OpenAI and Ollama.

## Ingestion idempotency

- SQLite sidecar DB tracks ingested files:
  - Table `ingested_files(path, content_hash, collection, chunk_ids JSON, ingested_at)`.
- Re-ingesting an unchanged file is a no-op.
- Changed file → delete old chunk_ids from Milvus, insert new ones.
- Removed file → CLI subcommand to purge.

## CLI surface

```
prism ingest <path> [--collection NAME] [--profile PROFILE]
    # Ingest a file or directory; idempotent via SQLite registry.

prism query "<question>" [--collection NAME] [--top-k N] [--stream]
    # Retrieve + generate with streaming output.

prism retrieve "<question>" [--collection NAME] [--top-k N] [--json]
    # Retrieval only. JSON output — precursor to MCP tool surface.

prism collections [list | create <name> | delete <name>]
    # Manage Milvus collections.

prism eval run <dataset.json> [--collection NAME] [--top-k N]
    # Run retrieval metrics against a Q&A dataset.
```

## Evaluation (retrieval only)

- Dataset format: JSON list of `{ "question": str, "relevant_sources": [str] }`.
- Metrics:
  - **Hit rate @ k** — was at least one relevant source in the top-k?
  - **MRR @ k** — mean reciprocal rank of first relevant hit.
- No judge-LLM required; fully deterministic and fast.
- Emits a per-question breakdown + aggregate table.

## Configuration

- Pydantic-settings root model loads from:
  1. YAML profile (`configs/openai.yaml`, `configs/local.yaml`, etc.)
  2. Environment variables (override YAML)
  3. CLI flags (override env)
- Profiles define: provider selection, model IDs, API keys via env, Milvus URI, chunker params, retrieval params.
- Secrets stay in env / `.env` — never in YAML checked to git.

## Module layout

```
src/prism_rag/
  __init__.py
  cli/                # typer app, subcommands
  config/             # pydantic-settings models + YAML loading
  loaders/            # markdown, pdf, html, image loaders (common interface)
  chunking/           # structure-aware + recursive splitters
  captioning/         # vision-LLM image → text (provider-abstracted)
  embeddings/         # openai / ollama providers behind one interface
  vectorstore/        # Milvus client wrapper, schema, hybrid search
  retrieval/          # dense, BM25, hybrid, reranker orchestration
  generation/         # streaming LLM generation, prompt templates
  pipeline/           # ingestion pipeline + query pipeline assembly
  registry/           # SQLite idempotency store
  eval/               # retrieval metric runners
  logging/            # structured logging setup, optional LangSmith hook
tests/
configs/
  openai.yaml
  local.yaml
docker/
  docker-compose.yml
  Dockerfile
```

## Docker topology

Single `docker-compose.yml` with profiles:

- `milvus-standalone` + its deps (`etcd`, `minio`) — always on.
- `ollama` — optional profile for local inference.
- `prism-rag` app container — CLI entrypoint, mounts a local `data/` dir for ingestion.

`.env` supplies API keys and selects the active config profile.

## Phased build plan

**Phase 0 — Scaffolding**
- `uv` project layout, module skeleton, Dockerfile + compose with Milvus, logging, Pydantic config, empty CLI.

**Phase 1 — Minimum ingest + query (OpenAI only)**
- Markdown + PDF loaders, structure-aware chunking, OpenAI embeddings, Milvus collection + insertion, dense-only retrieval, OpenAI generation, SQLite registry. `ingest` and `query` commands working end-to-end.

**Phase 2 — Provider swap**
- Ollama embeddings + generation behind the same interfaces. Config profiles for `openai` and `local`.

**Phase 3 — Modern retrieval**
- BM25 + dense hybrid search. `bge-reranker-v2-m3` reranker. Streaming generation. `retrieve` JSON command.

**Phase 4 — Expanded doc support**
- HTML loader. Image loader via vision-LLM captioning.

**Phase 5 — Evaluation**
- `eval run` command with hit-rate@k and MRR against a seed Q&A dataset.

**Phase 6 — Agent surface (deferred)**
- MCP server exposing `retrieve` as a tool.

**Phase 7 — GraphRAG (deferred)**
- Neo4j + entity/relation extraction layered onto ingestion; graph-aware retrieval fused with vector retrieval.

## Open questions for later phases

- GraphRAG: extract entities during ingestion (double-write) or as a separate pass? Which extraction model?
- MCP: single `retrieve` tool or also `list_collections`, `describe_source`?
- Eval: synthetic Q&A generation to bootstrap the dataset?
