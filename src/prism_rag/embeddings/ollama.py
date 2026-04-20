from __future__ import annotations

import logging

import ollama

log = logging.getLogger(__name__)


class ModelNotPulledError(RuntimeError):
    """Raised when the requested Ollama model is not available locally."""


class OllamaEmbedder:
    """Ollama-backed embeddings. Dimension is probed at construction time."""

    def __init__(self, model: str, host: str | None = None) -> None:
        self._model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()
        self._dimension = self._probe_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Ollama's /api/embeddings is single-input per call. Sequential is fine
        # for Phase 2; concurrency is a later optimization if ingest gets slow.
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        try:
            response = self._client.embeddings(model=self._model, prompt=text)
        except ollama.ResponseError as exc:
            self._raise_if_missing_model(exc)
            raise
        return list(response["embedding"])

    def _probe_dimension(self) -> int:
        log.debug("probing embedding dimension for '%s'", self._model)
        return len(self._embed_one("dimension probe"))

    def _raise_if_missing_model(self, exc: "ollama.ResponseError") -> None:
        msg = str(exc).lower()
        if "not found" in msg or "no such model" in msg:
            raise ModelNotPulledError(
                f"Model '{self._model}' is not available on the Ollama server. "
                f"Run: ollama pull {self._model}"
            ) from exc
