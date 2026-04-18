"""Embedding providers (OpenAI, Ollama) behind a common interface."""

from prism_rag.embeddings.base import Embedder
from prism_rag.embeddings.factory import build_embedder
from prism_rag.embeddings.openai import OpenAIEmbedder

__all__ = ["Embedder", "build_embedder", "OpenAIEmbedder"]
