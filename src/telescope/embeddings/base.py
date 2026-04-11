"""Abstract base class for embedding providers.

All query-side vector generation flows through an instance of a concrete
subclass. Implementations live as sibling modules (``openai_provider``,
``ollama_provider``) and are selected via :mod:`telescope.embeddings.factory`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Interface for query-side embedding generators.

    Subclasses wrap a specific backend (OpenAI API, Ollama HTTP API, etc.)
    and expose a uniform ``embed_batch`` coroutine plus ``model_name`` and
    ``dimensions`` introspection so telescope's read backends can build
    pgvector / Cypher-compatible query vectors without knowing which
    provider produced them.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the provider-specific model identifier in use."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding vector length this provider produces."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text, in order.

        An empty ``texts`` list MUST return an empty list without issuing
        any network calls.
        """
