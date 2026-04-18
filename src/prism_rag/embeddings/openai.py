from __future__ import annotations

import logging

from openai import OpenAI

log = logging.getLogger(__name__)

_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """OpenAI embeddings with client-side batching."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
    ) -> None:
        if model not in _MODEL_DIMENSIONS:
            raise ValueError(
                f"Unknown OpenAI embedding model '{model}'. Known: {sorted(_MODEL_DIMENSIONS)}"
            )
        self._model = model
        self._client = OpenAI(api_key=api_key)
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return _MODEL_DIMENSIONS[self._model]

    @property
    def model(self) -> str:
        return self._model

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            log.debug(
                "embedding batch %d/%d (%d items)",
                i // self._batch_size + 1,
                total_batches,
                len(batch),
            )
            response = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(d.embedding for d in response.data)
        return out
