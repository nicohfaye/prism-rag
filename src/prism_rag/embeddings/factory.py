from __future__ import annotations

from prism_rag.config import EmbeddingConfig
from prism_rag.embeddings.base import Embedder
from prism_rag.embeddings.openai import OpenAIEmbedder


def build_embedder(config: EmbeddingConfig) -> Embedder:
    if config.provider == "openai":
        return OpenAIEmbedder(model=config.model, api_key=config.api_key)
    raise NotImplementedError(
        f"Embedding provider '{config.provider}' not supported yet (Phase 1 supports OpenAI only)."
    )
