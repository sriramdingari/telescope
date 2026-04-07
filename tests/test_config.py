"""Tests for the Telescope config module."""

import os
from unittest.mock import patch

import pytest

import telescope.config as config_module
from telescope.config import Config, get_config


@pytest.fixture(autouse=True)
def reset_config():
    """Reset the global config singleton before each test."""
    config_module._config = None
    yield
    config_module._config = None


class TestConfigDefaults:
    """Tests for Config.from_env() default values."""

    def test_default_neo4j_uri(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.neo4j_uri == "bolt://localhost:7687"

    def test_default_neo4j_user(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.neo4j_user == "neo4j"

    def test_default_neo4j_password(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.neo4j_password == "constellation"

    def test_default_openai_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.openai_api_key == ""

    def test_default_openai_base_url_is_none(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.openai_base_url is None

    def test_default_embedding_model(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_default_embedding_dimensions(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.embedding_dimensions == 1536


class TestConfigEnvOverrides:
    """Tests that env vars correctly override defaults."""

    def test_neo4j_uri_override(self):
        with patch.dict(os.environ, {"NEO4J_URI": "bolt://remotehost:7687"}):
            cfg = Config.from_env()
        assert cfg.neo4j_uri == "bolt://remotehost:7687"

    def test_neo4j_user_override(self):
        with patch.dict(os.environ, {"NEO4J_USER": "admin"}):
            cfg = Config.from_env()
        assert cfg.neo4j_user == "admin"

    def test_neo4j_password_override(self):
        with patch.dict(os.environ, {"NEO4J_PASSWORD": "secretpass"}):
            cfg = Config.from_env()
        assert cfg.neo4j_password == "secretpass"

    def test_openai_api_key_override(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            cfg = Config.from_env()
        assert cfg.openai_api_key == "sk-test-key"

    def test_openai_base_url_override(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://custom.openai.example.com"}):
            cfg = Config.from_env()
        assert cfg.openai_base_url == "https://custom.openai.example.com"

    def test_embedding_model_override(self):
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "text-embedding-ada-002"}):
            cfg = Config.from_env()
        assert cfg.embedding_model == "text-embedding-ada-002"

    def test_embedding_dimensions_override(self):
        with patch.dict(os.environ, {"EMBEDDING_DIMENSIONS": "512"}):
            cfg = Config.from_env()
        assert cfg.embedding_dimensions == 512


class TestConfigTypeCoercion:
    """Tests for type parsing from string env vars."""

    def test_embedding_dimensions_is_int(self):
        with patch.dict(os.environ, {"EMBEDDING_DIMENSIONS": "768"}):
            cfg = Config.from_env()
        assert isinstance(cfg.embedding_dimensions, int)
        assert cfg.embedding_dimensions == 768

    def test_embedding_dimensions_default_is_int(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert isinstance(cfg.embedding_dimensions, int)

    def test_openai_base_url_empty_string_becomes_none(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": ""}):
            cfg = Config.from_env()
        assert cfg.openai_base_url is None


class TestGetConfig:
    """Tests for the get_config() singleton function."""

    def test_get_config_returns_config_instance(self):
        cfg = get_config()
        assert isinstance(cfg, Config)

    def test_get_config_caches_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_get_config_uses_fresh_instance_after_reset(self):
        cfg1 = get_config()
        config_module._config = None
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_reset_config_to_none_enables_isolation(self):
        """Verify that resetting _config allows a new env-based config to be created."""
        with patch.dict(os.environ, {"NEO4J_USER": "user_a"}):
            cfg1 = get_config()
        assert cfg1.neo4j_user == "user_a"

        config_module._config = None

        with patch.dict(os.environ, {"NEO4J_USER": "user_b"}):
            cfg2 = get_config()
        assert cfg2.neo4j_user == "user_b"

        assert cfg1 is not cfg2


def test_config_postgres_backend_requires_non_empty_dsn():
    """storage_backend=postgres with empty postgres_dsn must raise."""
    with pytest.raises(ValueError, match="postgres_dsn"):
        Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="pw",
            openai_api_key="sk-test",
            storage_backend="postgres",
            postgres_dsn="",
        )


def test_config_postgres_backend_accepts_non_empty_dsn():
    """storage_backend=postgres with a real DSN must construct cleanly."""
    c = Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="pw",
        openai_api_key="sk-test",
        storage_backend="postgres",
        postgres_dsn="postgresql://u:p@host:5432/db",
    )
    assert c.storage_backend == "postgres"
    assert c.postgres_dsn == "postgresql://u:p@host:5432/db"


def test_config_neo4j_backend_does_not_require_postgres_dsn():
    """storage_backend=neo4j must not require postgres_dsn."""
    c = Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="pw",
        openai_api_key="sk-test",
        storage_backend="neo4j",
        postgres_dsn="",
    )
    assert c.storage_backend == "neo4j"
