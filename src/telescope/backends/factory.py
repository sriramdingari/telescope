"""Factory for creating ReadBackend instances."""

from __future__ import annotations

from telescope.config import Config
from telescope.backends.base import ReadBackend
from telescope.embeddings.factory import create_embedding_provider


def create_read_backend(config: Config) -> ReadBackend:
    """Return the configured read backend.

    Reads ``config.storage_backend`` to select the implementation and
    ``config.embedding_provider`` to pick the embedding provider that
    the backend uses for query-side vector generation.
    """
    embedder = create_embedding_provider(config)

    if config.storage_backend == "postgres":
        from telescope.backends.postgres import PostgresReadBackend
        return PostgresReadBackend(
            dsn=config.postgres_dsn,
            embedder=embedder,
        )
    elif config.storage_backend == "neo4j":
        from telescope.backends.neo4j import Neo4jReadBackend
        return Neo4jReadBackend(embedder=embedder)
    else:
        raise ValueError(
            f"Unknown storage backend: {config.storage_backend!r}. "
            f"Expected 'neo4j' or 'postgres'."
        )
