"""Tests for create_embedding_provider factory."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from telescope.config import Config
from telescope.embeddings.factory import create_embedding_provider
from telescope.embeddings.openai_provider import OpenAIEmbeddingProvider
from telescope.embeddings.ollama_provider import OllamaEmbeddingProvider


def _cfg(**kw) -> Config:
    return Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="pw",
        openai_api_key="sk-test",
        **kw,
    )


class TestCreateEmbeddingProvider:
    def test_returns_openai_when_provider_is_openai(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI"):
            cfg = _cfg(embedding_provider="openai")
            provider = create_embedding_provider(cfg)
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_openai_provider_uses_configured_model(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI"):
            cfg = _cfg(
                embedding_provider="openai",
                embedding_model="text-embedding-3-large",
                embedding_dimensions=3072,
            )
            provider = create_embedding_provider(cfg)
        assert provider.model_name == "text-embedding-3-large"
        assert provider.dimensions == 3072

    def test_returns_ollama_when_provider_is_ollama(self):
        cfg = _cfg(embedding_provider="ollama")
        provider = create_embedding_provider(cfg)
        assert isinstance(provider, OllamaEmbeddingProvider)

    def test_ollama_provider_uses_configured_model(self):
        cfg = _cfg(
            embedding_provider="ollama",
            ollama_embedding_model="mxbai-embed-large",
            ollama_embedding_dimensions=1024,
            ollama_base_url="http://host.docker.internal:11434",
        )
        provider = create_embedding_provider(cfg)
        assert provider.model_name == "mxbai-embed-large"
        assert provider.dimensions == 1024

    def test_raises_on_unknown_provider(self):
        cfg = _cfg(embedding_provider="nonsense")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(cfg)
