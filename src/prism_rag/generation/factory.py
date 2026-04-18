from __future__ import annotations

from prism_rag.config import GenerationConfig
from prism_rag.generation.base import Generator
from prism_rag.generation.openai import OpenAIGenerator


def build_generator(config: GenerationConfig) -> Generator:
    if config.provider == "openai":
        return OpenAIGenerator(model=config.model, api_key=config.api_key)
    raise NotImplementedError(
        f"Generation provider '{config.provider}' not supported yet (Phase 1 supports OpenAI only)."
    )
