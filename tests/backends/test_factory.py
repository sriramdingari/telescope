"""Tests for create_read_backend factory."""
import pytest
from unittest.mock import MagicMock, patch

from telescope.config import Config
from telescope.backends.factory import create_read_backend
from telescope.backends.base import ReadBackend
from telescope.embeddings.openai_provider import OpenAIEmbeddingProvider
from telescope.embeddings.ollama_provider import OllamaEmbeddingProvider


def _make_config(
    storage_backend: str = "neo4j",
    postgres_dsn: str = "",
    embedding_provider: str = "openai",
    **kw,
) -> Config:
    return Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
        openai_api_key="test-key",
        storage_backend=storage_backend,
        postgres_dsn=postgres_dsn,
        embedding_provider=embedding_provider,
        **kw,
    )


def test_factory_returns_neo4j_by_default():
    from telescope.backends.neo4j import Neo4jReadBackend
    with patch("telescope.embeddings.openai_provider.AsyncOpenAI"):
        config = _make_config()
        backend = create_read_backend(config)
    assert isinstance(backend, Neo4jReadBackend)
    assert isinstance(backend, ReadBackend)


def test_factory_raises_on_unknown_backend():
    with patch("telescope.embeddings.openai_provider.AsyncOpenAI"):
        config = _make_config(storage_backend="sqlite")
        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_read_backend(config)


def test_factory_returns_postgres_when_configured():
    # Patch the class directly on the already-imported postgres module.
    # This avoids the brittle `sys.modules` replacement pattern — by the
    # time this test runs, `telescope.backends.postgres` is almost
    # certainly already in `sys.modules` via other test collection paths,
    # which would defeat a module-level stub. An attribute-level patch
    # targets the bound name the factory resolves at call time.
    with patch(
        "telescope.backends.postgres.PostgresReadBackend"
    ) as mock_postgres_cls, patch(
        "telescope.embeddings.openai_provider.AsyncOpenAI"
    ):
        config = _make_config(
            storage_backend="postgres", postgres_dsn="postgresql://test@localhost/db"
        )
        create_read_backend(config)
        mock_postgres_cls.assert_called_once()
        kwargs = mock_postgres_cls.call_args.kwargs
        assert kwargs["dsn"] == "postgresql://test@localhost/db"
        assert "embedder" in kwargs
        assert isinstance(kwargs["embedder"], OpenAIEmbeddingProvider)


def test_factory_passes_ollama_embedder_when_provider_is_ollama():
    from telescope.backends.neo4j import Neo4jReadBackend

    config = _make_config(embedding_provider="ollama")
    backend = create_read_backend(config)
    assert isinstance(backend, Neo4jReadBackend)
    assert isinstance(backend._embedder, OllamaEmbeddingProvider)


def test_factory_passes_openai_embedder_when_provider_is_openai():
    from telescope.backends.neo4j import Neo4jReadBackend

    with patch("telescope.embeddings.openai_provider.AsyncOpenAI"):
        config = _make_config(embedding_provider="openai")
        backend = create_read_backend(config)
    assert isinstance(backend, Neo4jReadBackend)
    assert isinstance(backend._embedder, OpenAIEmbeddingProvider)
