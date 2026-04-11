"""OpenAI-backed embedding provider.

Thin wrapper around :class:`openai.AsyncOpenAI` that matches the
:class:`telescope.embeddings.base.BaseEmbeddingProvider` contract.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from telescope.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Generate embeddings via the OpenAI (or OpenAI-compatible) API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client_kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            self._client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**self._client_kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]
