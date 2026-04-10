"""Tests for create_read_backend factory."""
import pytest
from unittest.mock import MagicMock, patch

from telescope.config import Config
from telescope.backends.factory import create_read_backend
from telescope.backends.base import ReadBackend


def _make_config(storage_backend: str = "neo4j", postgres_dsn: str = "") -> Config:
    return Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
        openai_api_key="test-key",
        storage_backend=storage_backend,
        postgres_dsn=postgres_dsn,
    )


def test_factory_returns_neo4j_by_default():
    from telescope.backends.neo4j import Neo4jReadBackend
    config = _make_config()
    backend = create_read_backend(config)
    assert isinstance(backend, Neo4jReadBackend)
    assert isinstance(backend, ReadBackend)


def test_factory_raises_on_unknown_backend():
    config = _make_config(storage_backend="sqlite")
    with pytest.raises(ValueError, match="Unknown storage backend"):
        create_read_backend(config)


def test_factory_returns_postgres_when_configured():
    mock_postgres_cls = MagicMock()
    with patch.dict(
        "sys.modules",
        {"telescope.backends.postgres": MagicMock(PostgresReadBackend=mock_postgres_cls)},
    ):
        config = _make_config(storage_backend="postgres", postgres_dsn="postgresql://test@localhost/db")
        create_read_backend(config)
        mock_postgres_cls.assert_called_once()
        kwargs = mock_postgres_cls.call_args.kwargs
        assert kwargs["dsn"] == "postgresql://test@localhost/db"
        assert kwargs["openai_api_key"] == "test-key"
