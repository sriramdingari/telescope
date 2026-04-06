"""Configuration for Telescope MCP server."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Server configuration loaded from environment variables."""
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    openai_api_key: str
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    storage_backend: str = "neo4j"
    postgres_dsn: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "constellation"),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS", "1536")),
            storage_backend=os.environ.get("STORAGE_BACKEND", "neo4j"),
            postgres_dsn=os.environ.get("POSTGRES_DSN", ""),
        )


_config: Config | None = None


def get_config() -> Config:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
