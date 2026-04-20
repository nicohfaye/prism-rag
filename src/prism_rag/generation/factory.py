from __future__ import annotations

from prism_rag.config import GenerationConfig
from prism_rag.generation.base import Generator
from prism_rag.generation.ollama import OllamaGenerator
from prism_rag.generation.openai import OpenAIGenerator


def build_generator(config: GenerationConfig) -> Generator:
    if config.provider == "openai":
        return OpenAIGenerator(model=config.model, api_key=config.api_key)
    if config.provider == "ollama":
        return OllamaGenerator(model=config.model, host=config.base_url)
    raise ValueError(f"Unknown generation provider: {config.provider!r}")
