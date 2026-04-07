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
async def test_resolve_method_family_returns_overrides(backend, mock_pool):
    """_resolve_method_family must return the original method plus siblings
    on classes related via IMPLEMENTS/EXTENDS."""
    mock_pool.fetch = AsyncMock(return_value=[
        {"id": "repo::IFoo.doIt"},
        {"id": "repo::FooImpl.doIt"},
        {"id": "repo::FooSub.doIt"},
    ])

    symbol = {"id": "repo::FooImpl.doIt", "symbol_name": "doIt"}
    family_ids = await backend._resolve_method_family(symbol)

    assert "repo::IFoo.doIt" in family_ids
    assert "repo::FooImpl.doIt" in family_ids
    assert "repo::FooSub.doIt" in family_ids
    assert len(family_ids) == 3

    # Verify the SQL was called with symbol["id"] as $1 and symbol["symbol_name"] as $2
    call_args = mock_pool.fetch.call_args[0]
    assert call_args[1] == "repo::FooImpl.doIt", \
        f"Expected symbol id as $1, got {call_args[1]}"
    assert call_args[2] == "doIt", \
        f"Expected symbol_name as $2, got {call_args[2]}"


@pytest.mark.asyncio
async def test_get_callers_uses_method_family_for_polymorphism(backend, mock_pool):
    """get_callers must pass the full method family to the recursive CTE
    so callers of a polymorphic override are found regardless of which
    concrete method was resolved."""
    # Patch _resolve_symbol and _resolve_method_family to focus on get_callers logic
    async def fake_resolve(*args, **kwargs):
        return {"id": "repo::FooImpl.doIt", "symbol_name": "doIt",
                "file_path": "Foo.java", "repository": "repo"}
    async def fake_family(symbol):
        return ["repo::IFoo.doIt", "repo::FooImpl.doIt", "repo::FooSub.doIt"]

    backend._resolve_symbol = fake_resolve
    backend._resolve_method_family = fake_family
    mock_pool.fetch = AsyncMock(return_value=[])  # no callers for simplicity

    await backend.get_callers("doIt")

    # Verify the recursive CTE was called with an array parameter (family IDs)
    assert mock_pool.fetch.call_count >= 1
    # The first positional arg after the SQL should be a list
    sql = mock_pool.fetch.call_args[0][0]
    args = mock_pool.fetch.call_args[0][1:]
    assert isinstance(args[0], list), f"Expected list of family IDs, got {type(args[0])}"
    assert len(args[0]) == 3
    # The SQL should use ANY($1), not $1 alone
    assert "ANY($1)" in sql
    # Verify exact family IDs are passed, not just the count
    assert set(args[0]) == {
        "repo::IFoo.doIt",
        "repo::FooImpl.doIt",
        "repo::FooSub.doIt",
    }


@pytest.mark.asyncio
async def test_get_callees_uses_method_family_for_polymorphism(backend, mock_pool):
    """get_callees must pass the full method family to the recursive CTE
    AND to the USES_HOOK query, symmetric to get_callers."""
    async def fake_resolve(*args, **kwargs):
        return {"id": "repo::FooImpl.doIt", "symbol_name": "doIt",
                "file_path": "Foo.java", "repository": "repo"}
    async def fake_family(symbol):
        return ["repo::IFoo.doIt", "repo::FooImpl.doIt", "repo::FooSub.doIt"]

    backend._resolve_symbol = fake_resolve
    backend._resolve_method_family = fake_family
    mock_pool.fetch = AsyncMock(return_value=[])

    await backend.get_callees("doIt")

    # get_callees makes 2 fetch calls: CALLS recursive CTE + USES_HOOK query
    assert mock_pool.fetch.call_count == 2

    # First call (CALLS): expects family_ids as $1
    first_sql, *first_args = mock_pool.fetch.call_args_list[0][0]
    assert "ANY($1)" in first_sql
    assert isinstance(first_args[0], list)
    assert set(first_args[0]) == {
        "repo::IFoo.doIt",
        "repo::FooImpl.doIt",
        "repo::FooSub.doIt",
    }

    # Second call (USES_HOOK): also uses family_ids
    second_sql, *second_args = mock_pool.fetch.call_args_list[1][0]
    assert "USES_HOOK" in second_sql
    assert "ANY($1)" in second_sql
    assert isinstance(second_args[0], list)
    assert set(second_args[0]) == {
        "repo::IFoo.doIt",
        "repo::FooImpl.doIt",
        "repo::FooSub.doIt",
    }


@pytest.mark.asyncio
async def test_get_impact_counts_before_limit_truncation(backend, mock_pool):
    """get_impact must compute total_callers/test_count/endpoint_count from
    the full caller set, then truncate each category independently by limit."""
    async def fake_resolve(*args, **kwargs):
        return {"id": "repo::svc.doIt", "symbol_name": "doIt",
                "file_path": "svc.py", "repository": "repo"}
    backend._resolve_symbol = fake_resolve
    backend._resolve_method_family = AsyncMock(return_value=["repo::svc.doIt"])

    # Return 20 tests, 5 endpoints, 10 others = 35 total callers
    all_rows = (
        [{"id": f"t{i}", "symbol_name": f"test_{i}", "file_path": "test.py",
          "repository": "repo", "signature": None, "line_start": i,
          "symbol_type": "Method", "is_test": True, "is_endpoint": False,
          "depth": 1} for i in range(20)]
        + [{"id": f"e{i}", "symbol_name": f"endpoint_{i}", "file_path": "api.py",
            "repository": "repo", "signature": None, "line_start": i,
            "symbol_type": "Method", "is_test": False, "is_endpoint": True,
            "depth": 1} for i in range(5)]
        + [{"id": f"o{i}", "symbol_name": f"other_{i}", "file_path": "other.py",
            "repository": "repo", "signature": None, "line_start": i,
            "symbol_type": "Method", "is_test": False, "is_endpoint": False,
            "depth": 1} for i in range(10)]
    )
    mock_pool.fetch = AsyncMock(return_value=all_rows)

    result = await backend.get_impact("doIt", limit=3)

    # Total counts reflect the FULL caller set, not the truncated ones
    assert result.total_callers == 35
    assert result.test_count == 20
    assert result.endpoint_count == 5
    # Each category truncated independently to limit=3
    assert len(result.affected_tests) == 3
    assert len(result.affected_endpoints) == 3
    assert len(result.other_callers) == 3
    # truncated is True because at least one category exceeded the limit
    assert result.truncated is True


@pytest.mark.asyncio
async def test_get_impact_excludes_family_methods_from_callers(backend, mock_pool):
    """A method calling a sibling override shouldn't count as impact on itself.
    The recursive CTE must exclude family_ids from the caller set."""
    async def fake_resolve(*args, **kwargs):
        return {"id": "repo::FooImpl.doIt", "symbol_name": "doIt",
                "file_path": "Foo.java", "repository": "repo"}
    async def fake_family(symbol):
        return ["repo::IFoo.doIt", "repo::FooImpl.doIt", "repo::FooSub.doIt"]

    backend._resolve_symbol = fake_resolve
    backend._resolve_method_family = fake_family
    mock_pool.fetch = AsyncMock(return_value=[])

    await backend.get_impact("doIt")

    # Verify the SQL includes the s.id != ALL($1) exclusion
    sql = mock_pool.fetch.call_args[0][0]
    assert "ANY($1)" in sql, f"Expected ANY($1) for family matching; got SQL: {sql}"
    assert "!= ALL($1)" in sql, \
        f"Expected s.id != ALL($1) to exclude family from callers; got SQL: {sql}"


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
