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
    # Embedding provider selection -----------------------------------
    # ``embedding_provider`` defaults to "openai" so existing deployments
    # that set only OPENAI_* env vars keep working. Setting it to
    # "ollama" activates the Ollama-specific settings below.
    embedding_provider: str = "openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_embedding_dimensions: int = 768

    def __post_init__(self) -> None:
        """Validate that postgres_dsn is set when storage_backend is postgres.

        Fails fast at config construction instead of deferring to an opaque
        asyncpg connection error when the factory tries to dial an empty DSN.
        """
        if self.storage_backend == "postgres" and not self.postgres_dsn:
            raise ValueError(
                "storage_backend='postgres' requires postgres_dsn to be set. "
                "Example: postgresql://user:pass@host:5432/dbname"
            )

    def resolved_embedding_model(self) -> str:
        """Return the model name for the active ``embedding_provider``."""
        if self.embedding_provider == "openai":
            return self.embedding_model
        if self.embedding_provider == "ollama":
            return self.ollama_embedding_model
        raise ValueError(f"Unknown embedding provider: {self.embedding_provider!r}")

    def resolved_embedding_dimensions(self) -> int:
        """Return the vector length for the active ``embedding_provider``."""
        if self.embedding_provider == "openai":
            return self.embedding_dimensions
        if self.embedding_provider == "ollama":
            return self.ollama_embedding_dimensions
        raise ValueError(f"Unknown embedding provider: {self.embedding_provider!r}")

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
            embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "openai"),
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            ollama_embedding_model=os.environ.get(
                "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
            ),
            ollama_embedding_dimensions=int(
                os.environ.get("OLLAMA_EMBEDDING_DIMENSIONS", "768")
            ),
        )


_config: Config | None = None


def get_config() -> Config:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
