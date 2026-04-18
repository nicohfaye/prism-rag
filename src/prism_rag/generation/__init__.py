"""LLM generation with streaming and citation-aware prompts."""

from prism_rag.generation.base import Generator
from prism_rag.generation.factory import build_generator
from prism_rag.generation.openai import OpenAIGenerator
from prism_rag.generation.prompt import SYSTEM_PROMPT, build_user_message, format_context

__all__ = [
    "Generator",
    "OpenAIGenerator",
    "SYSTEM_PROMPT",
    "build_generator",
    "build_user_message",
    "format_context",
]
