"""Tests for OpenAIEmbeddingProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telescope.embeddings.openai_provider import OpenAIEmbeddingProvider


class TestOpenAIEmbeddingProviderProperties:
    def test_returns_configured_model(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI"):
            p = OpenAIEmbeddingProvider(
                api_key="sk-test",
                model="text-embedding-3-large",
            )
        assert p.model_name == "text-embedding-3-large"

    def test_returns_configured_dimensions(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI"):
            p = OpenAIEmbeddingProvider(
                api_key="sk-test",
                dimensions=2048,
            )
        assert p.dimensions == 2048

    def test_default_model(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI"):
            p = OpenAIEmbeddingProvider(api_key="sk-test")
        assert p.model_name == "text-embedding-3-small"

    def test_default_dimensions(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI"):
            p = OpenAIEmbeddingProvider(api_key="sk-test")
        assert p.dimensions == 1536


class TestOpenAIEmbedBatch:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_without_network_call(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            p = OpenAIEmbeddingProvider(api_key="sk-test")
            result = await p.embed_batch([])
        assert result == []
        mock_cls.return_value.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_single(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            mock_item = MagicMock()
            mock_item.embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [mock_item]
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            p = OpenAIEmbeddingProvider(api_key="sk-test", dimensions=3)
            result = await p.embed_batch(["hello"])

        assert result == [[0.1, 0.2, 0.3]]
        mock_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input=["hello"],
            dimensions=3,
        )

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            items = []
            for vec in [[0.1], [0.2], [0.3]]:
                it = MagicMock()
                it.embedding = vec
                items.append(it)
            mock_response = MagicMock()
            mock_response.data = items
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            p = OpenAIEmbeddingProvider(api_key="sk-test", dimensions=1)
            result = await p.embed_batch(["a", "b", "c"])

        assert result == [[0.1], [0.2], [0.3]]
        mock_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input=["a", "b", "c"],
            dimensions=1,
        )

    @pytest.mark.asyncio
    async def test_passes_base_url_when_provided(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            OpenAIEmbeddingProvider(
                api_key="sk-test",
                base_url="https://proxy.example.com",
            )
        mock_cls.assert_called_once_with(
            api_key="sk-test",
            base_url="https://proxy.example.com",
        )

    @pytest.mark.asyncio
    async def test_omits_base_url_when_none(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            OpenAIEmbeddingProvider(api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test")
