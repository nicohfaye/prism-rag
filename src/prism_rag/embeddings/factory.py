from __future__ import annotations

from prism_rag.config import EmbeddingConfig
from prism_rag.embeddings.base import Embedder
from prism_rag.embeddings.ollama import OllamaEmbedder
from prism_rag.embeddings.openai import OpenAIEmbedder


def build_embedder(config: EmbeddingConfig) -> Embedder:
    if config.provider == "openai":
        return OpenAIEmbedder(model=config.model, api_key=config.api_key)
    if config.provider == "ollama":
        return OllamaEmbedder(model=config.model, host=config.base_url)
    raise ValueError(f"Unknown embedding provider: {config.provider!r}")
