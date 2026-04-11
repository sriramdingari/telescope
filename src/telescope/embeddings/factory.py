"""Embedding provider factory.

Selects and instantiates the concrete provider based on the active
``Config.embedding_provider`` setting. Providers are imported lazily so
that environments without one of the underlying client libraries
(unlikely in practice, but possible) don't crash on import just to use
the other provider.
"""

from __future__ import annotations

from telescope.config import Config
from telescope.embeddings.base import BaseEmbeddingProvider


def create_embedding_provider(config: Config) -> BaseEmbeddingProvider:
    """Return the concrete embedding provider for the active config."""
    provider = config.embedding_provider
    if provider == "openai":
        from telescope.embeddings.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
    if provider == "ollama":
        from telescope.embeddings.ollama_provider import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            base_url=config.ollama_base_url,
            model=config.ollama_embedding_model,
            dimensions=config.ollama_embedding_dimensions,
        )
    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        f"Expected 'openai' or 'ollama'."
    )
