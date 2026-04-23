from __future__ import annotations

import logging
from collections.abc import Iterator

import ollama

from prism_rag.generation.prompt import SYSTEM_PROMPT, build_user_message
from prism_rag.retrieval import RetrievedChunk

log = logging.getLogger(__name__)


class OllamaGenerator:
    def __init__(self, model: str, host: str | None = None) -> None:
        self._model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()

    @property
    def model(self) -> str:
        return self._model

    def stream(self, question: str, context: list[RetrievedChunk]) -> Iterator[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, context)},
        ]
        try:
            stream = self._client.chat(model=self._model, messages=messages, stream=True)
            for event in stream:
                # Each event is a dict-like; content lives at event["message"]["content"].
                content = event.get("message", {}).get("content")
                if content:
                    yield content
        except ollama.ResponseError as exc:
            msg = str(exc).lower()
            if "not found" in msg or "no such model" in msg:
                raise RuntimeError(
                    f"Model '{self._model}' is not available on the Ollama server. "
                    f"Run: ollama pull {self._model}"
                ) from exc
            raise
