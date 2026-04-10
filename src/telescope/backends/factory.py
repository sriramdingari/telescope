"""Factory for creating ReadBackend instances."""

from __future__ import annotations

from telescope.config import Config
from telescope.backends.base import ReadBackend


def create_read_backend(config: Config) -> ReadBackend:
    """Return the configured read backend.

    Reads config.storage_backend to select the implementation.
    """
    if config.storage_backend == "postgres":
        from telescope.backends.postgres import PostgresReadBackend
        return PostgresReadBackend(
            dsn=config.postgres_dsn,
            openai_api_key=config.openai_api_key,
            openai_base_url=config.openai_base_url,
            embedding_model=config.embedding_model,
            embedding_dimensions=config.embedding_dimensions,
        )
    elif config.storage_backend == "neo4j":
        from telescope.backends.neo4j import Neo4jReadBackend
        return Neo4jReadBackend()
    else:
        raise ValueError(
            f"Unknown storage backend: {config.storage_backend!r}. "
            f"Expected 'neo4j' or 'postgres'."
        )
