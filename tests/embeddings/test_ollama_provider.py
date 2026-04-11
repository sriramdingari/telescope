"""Tests for OllamaEmbeddingProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telescope.embeddings.ollama_provider import OllamaEmbeddingProvider


class TestOllamaEmbeddingProviderProperties:
    def test_returns_configured_model(self):
        p = OllamaEmbeddingProvider(model="mxbai-embed-large")
        assert p.model_name == "mxbai-embed-large"

    def test_default_model(self):
        p = OllamaEmbeddingProvider()
        assert p.model_name == "nomic-embed-text"

    def test_returns_configured_dimensions(self):
        p = OllamaEmbeddingProvider(dimensions=1024)
        assert p.dimensions == 1024

    def test_default_dimensions(self):
        p = OllamaEmbeddingProvider()
        assert p.dimensions == 768

    def test_strips_trailing_slash_from_base_url(self):
        p = OllamaEmbeddingProvider(base_url="http://localhost:11434/")
        assert p._base_url == "http://localhost:11434"


class TestOllamaEmbedBatch:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_without_network_call(self):
        with patch("telescope.embeddings.ollama_provider.httpx.AsyncClient") as mock_cls:
            p = OllamaEmbeddingProvider()
            result = await p.embed_batch([])
        assert result == []
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_single_batch_request(self):
        with patch("telescope.embeddings.ollama_provider.httpx.AsyncClient") as mock_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__.return_value = mock_client

            p = OllamaEmbeddingProvider(
                base_url="http://myhost:11434",
                model="nomic-embed-text",
            )
            result = await p.embed_batch(["alpha", "beta"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.post.assert_awaited_once_with(
            "http://myhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": ["alpha", "beta"]},
        )

    @pytest.mark.asyncio
    async def test_return_shape_for_three_inputs(self):
        with patch("telescope.embeddings.ollama_provider.httpx.AsyncClient") as mock_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "embeddings": [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2],
                ],
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__.return_value = mock_client

            p = OllamaEmbeddingProvider(dimensions=4)
            result = await p.embed_batch(["a", "b", "c"])

        assert len(result) == 3
        assert all(len(vec) == 4 for vec in result)
        assert mock_client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        with patch("telescope.embeddings.ollama_provider.httpx.AsyncClient") as mock_cls:
            import httpx as _httpx

            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock(
                side_effect=_httpx.HTTPStatusError(
                    "500", request=MagicMock(), response=MagicMock()
                )
            )
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__.return_value = mock_client

            p = OllamaEmbeddingProvider()
            with pytest.raises(_httpx.HTTPStatusError):
                await p.embed_batch(["x"])
