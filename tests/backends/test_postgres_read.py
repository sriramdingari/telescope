"""Unit tests for PostgresReadBackend — mocked asyncpg pool."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from telescope.backends.postgres import PostgresReadBackend
from telescope.backends.base import ReadBackend


@pytest.fixture
def mock_pool():
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=0)
    return pool


@pytest.fixture
def backend(mock_pool):
    b = PostgresReadBackend(
        dsn="postgresql://test:test@localhost/test",
        openai_api_key="test-key",
    )
    b._pool = mock_pool
    return b


def test_implements_read_backend():
    assert issubclass(PostgresReadBackend, ReadBackend)


@pytest.mark.asyncio
async def test_list_repositories_returns_empty(backend, mock_pool):
    mock_pool.fetch.return_value = []
    result = await backend.list_repositories()
    assert result == []


@pytest.mark.asyncio
async def test_find_symbols_returns_empty(backend, mock_pool):
    mock_pool.fetch.return_value = []
    result = await backend.find_symbols("myMethod")
    assert result == []


@pytest.mark.asyncio
async def test_get_callers_returns_empty_when_symbol_not_found(backend, mock_pool):
    mock_pool.fetch.return_value = []
    mock_pool.fetchrow.return_value = None
    result = await backend.get_callers("nonexistent")
    assert result == []


@pytest.mark.asyncio
async def test_get_impact_returns_none_when_symbol_not_found(backend, mock_pool):
    mock_pool.fetch.return_value = []
    mock_pool.fetchrow.return_value = None
    result = await backend.get_impact("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_resolve_symbol_returns_none_when_not_found(backend, mock_pool):
    mock_pool.fetch.return_value = []
    result = await backend._resolve_symbol(
        "missing", repository=None, file_path=None,
        entity_id=None, types=["Method"]
    )
    assert result is None


@pytest.mark.asyncio
async def test_resolve_symbol_raises_on_ambiguity(backend, mock_pool):
    row1 = {"id": "id-1", "symbol_name": "duplicate", "file_path": "a.py"}
    row2 = {"id": "id-2", "symbol_name": "duplicate", "file_path": "b.py"}
    mock_pool.fetch.return_value = [row1, row2]
    with pytest.raises(ValueError, match="Ambiguous symbol"):
        await backend._resolve_symbol(
            "duplicate", repository=None, file_path=None,
            entity_id=None, types=["Method"]
        )


@pytest.mark.asyncio
async def test_get_function_context_returns_none_when_missing(backend, mock_pool):
    mock_pool.fetch.return_value = []
    result = await backend.get_function_context("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_list_repositories_normalizes_datetime(backend, mock_pool):
    """Verify get_repository_context handles datetime → str conversion."""
    from datetime import datetime
    mock_pool.fetch.return_value = [
        {"name": "repo1", "source": "/tmp/repo1", "entity_count": 10,
         "last_indexed_at": datetime(2026, 1, 1), "commit_sha": "abc"},
    ]
    result = await backend.list_repositories()
    assert len(result) == 1
    assert result[0]["name"] == "repo1"


@pytest.mark.asyncio
async def test_get_codebase_overview_returns_empty(backend, mock_pool):
    mock_pool.fetch.return_value = []
    result = await backend.get_codebase_overview()
    assert result.total_files == 0
    assert result.total_classes == 0


@pytest.mark.asyncio
async def test_get_file_context_populates_exports(backend, mock_pool):
    """get_file_context must actually return exports, not hardcode [] ."""
    # Mock pool.fetch to return different things for different queries in order:
    # 1. Initial file lookup (LIMIT 2)
    # 2. Contained entities
    # 3. Exports (full entity rows)
    # 4. Packages (via IN_PACKAGE)
    # 5. Hook usages
    # 6. References (USES_TYPE)
    file_row = {
        "id": "repo::src/foo.py",
        "symbol_name": "foo.py",
        "file_path": "src/foo.py",
        "repository": "repo",
        "language": "python",
        "content_hash": "h1",
    }
    fetch_results = [
        [file_row],  # initial file lookup
        [{"symbol_name": "MyClass", "symbol_type": "Class", "line_start": 1}],  # contained
        [{"id": "repo::src/foo.py::helper",  # exports (full entity rows)
          "symbol_name": "helper", "symbol_type": "Method",
          "file_path": "src/foo.py", "repository": "repo",
          "line_start": 5, "line_end": 10, "signature": "def helper()",
          "code": "def helper(): pass", "docstring": None,
          "language": "python", "return_type": None,
          "modifiers": [], "stereotypes": [], "content_hash": "h1",
          "properties": {}}],
        [{"symbol_name": "com.example"}],  # packages (IN_PACKAGE)
        [],  # hook_usages
        [{"symbol_name": "SomeType"}, {"symbol_name": "AnotherType"}],  # references
    ]
    call_count = {"n": 0}
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_results[idx] if idx < len(fetch_results) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_file_context("src/foo.py")

    assert result is not None
    # exports must have the entity we returned — not []
    assert len(result.exports) == 1
    assert result.exports[0].name == "helper"
    # packages must include the package we returned — not []
    assert "com.example" in result.packages
    # references must contain the Reference names we returned — not []
    assert "SomeType" in result.references
    assert "AnotherType" in result.references
    assert len(result.references) == 2
