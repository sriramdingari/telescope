"""Ollama-backed embedding provider.

Uses the ``/api/embed`` (singular) endpoint which accepts ``input`` as a
list and returns all vectors in one HTTP round-trip. The older
``/api/embeddings`` endpoint only accepts one ``prompt`` at a time, which
would force one HTTP request per text — orders of magnitude slower for
batch queries.

Ollama responses for this endpoint look like::

    {"embeddings": [[...], [...], ...]}
"""

from __future__ import annotations

import httpx

from telescope.embeddings.base import BaseEmbeddingProvider

# Give Ollama generous headroom — embedding a large batch with a cold
# model can take tens of seconds on CPU-only hosts.
_DEFAULT_TIMEOUT = httpx.Timeout(300.0)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Generate embeddings via a local (or remote) Ollama server."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimensions: int = 768,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
        return data["embeddings"]
