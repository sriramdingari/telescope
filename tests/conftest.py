"""Shared test fixtures for Telescope tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def mock_neo4j_result():
    """Create a mock Neo4j result that returns configurable data."""
    def _make_result(data):
        result = AsyncMock()
        records = [MagicMock(data=MagicMock(return_value=d)) for d in data]
        result.fetch = AsyncMock(return_value=records)
        result.data = AsyncMock(return_value=data)
        return result
    return _make_result


@pytest.fixture()
def mock_neo4j_session(mock_neo4j_result):
    """Create a mock Neo4j async session."""
    session = AsyncMock()
    session.run = AsyncMock(return_value=mock_neo4j_result([]))
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture()
def mock_neo4j_driver(mock_neo4j_session):
    """Create a mock Neo4j async driver."""
    driver = AsyncMock()
    driver.session = MagicMock(return_value=mock_neo4j_session)
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    return driver


@pytest.fixture()
def mock_openai_response():
    """Create a mock OpenAI embedding response."""
    def _make_response(embedding=None):
        if embedding is None:
            embedding = [0.1] * 1536
        response = MagicMock()
        data_item = MagicMock()
        data_item.embedding = embedding
        response.data = [data_item]
        return response
    return _make_response


@pytest.fixture()
def mock_openai_client(mock_openai_response):
    """Create a mock AsyncOpenAI client."""
    client = AsyncMock()
    client.embeddings = AsyncMock()
    client.embeddings.create = AsyncMock(return_value=mock_openai_response())
    client.close = AsyncMock()
    return client
