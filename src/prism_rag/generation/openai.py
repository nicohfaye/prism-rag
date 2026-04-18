from __future__ import annotations

import logging
from collections.abc import Iterator

from openai import OpenAI

from prism_rag.generation.prompt import SYSTEM_PROMPT, build_user_message
from prism_rag.retrieval import RetrievedChunk

log = logging.getLogger(__name__)


class OpenAIGenerator:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def stream(self, question: str, context: list[RetrievedChunk]) -> Iterator[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, context)},
        ]
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            stream=True,
        )  # type: ignore
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta
