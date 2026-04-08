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


@pytest.mark.asyncio
async def test_get_codebase_overview_counts_exports(backend, mock_pool):
    """get_codebase_overview must include an exports count via the EXPORTS
    ref_type (matching Neo4j)."""
    counts = [{"symbol_type": "File", "cnt": 5},
              {"symbol_type": "Class", "cnt": 10}]
    langs = [{"language": "python"}]
    entry_points = []
    top_classes = []
    export_count = [{"cnt": 7}]

    call_count = {"n": 0}
    fetch_sequence = [counts, langs, entry_points, top_classes, export_count]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_codebase_overview(repository="repo")

    assert result.total_exports == 7


@pytest.mark.asyncio
async def test_get_codebase_overview_include_packages_returns_names(backend, mock_pool):
    """When include_packages=True, the overview must return package names."""
    counts = []
    langs = []
    entry_points = []
    top_classes = []
    export_count = [{"cnt": 0}]
    packages = [{"symbol_name": "com.example.app"}, {"symbol_name": "com.example.util"}]

    call_count = {"n": 0}
    fetch_sequence = [counts, langs, entry_points, top_classes, export_count, packages]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_codebase_overview(repository="repo", include_packages=True)

    assert "com.example.app" in result.packages
    assert "com.example.util" in result.packages


@pytest.mark.asyncio
async def test_get_package_context_populates_hooks_and_references(backend, mock_pool):
    """get_package_context must populate hooks and references as direct
    IN_PACKAGE members, not via a two-hop traversal. Matches Neo4j semantics."""
    async def fake_resolve_package(*args, **kwargs):
        # symbol_name is the LEAF only — _full_name_from_id reconstructs
        # the full dotted name "com.example" from the id.
        return {"id": "repo::com.example", "symbol_name": "example",
                "repository": "repo"}
    backend._resolve_package = fake_resolve_package

    # members query returns ALL direct IN_PACKAGE members including hooks and refs
    members = [
        {"symbol_name": "Foo", "symbol_type": "Class", "file_path": "Foo.java"},
        {"symbol_name": "useAuth", "symbol_type": "Hook", "file_path": "useAuth.ts"},
        {"symbol_name": "Logger", "symbol_type": "Reference", "file_path": "Logger.java"},
    ]
    child_pkgs: list = []

    call_count = {"n": 0}
    fetch_sequence = [members, child_pkgs]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_package_context("com.example")

    # Direct IN_PACKAGE members are categorized by symbol_type
    assert "useAuth" in result.hooks
    assert "Logger" in result.references
    assert "Foo" in result.classes
    # Full dotted name reconstructed from id, not the leaf "example"
    assert result.name == "com.example"

    # Verify only 2 fetch calls were made (members + child_pkgs)
    # No separate hooks or references queries — they come from the members row set
    assert mock_pool.fetch.call_count == 2


@pytest.mark.asyncio
async def test_get_package_context_resolves_nested_namespace_by_full_name(backend, mock_pool):
    """A nested .NET namespace like 'Company.Product.Services' must be
    resolvable by its full dotted name even though the parser stores only
    'Services' as symbol_name."""
    # _resolve_package queries the code_symbols table for packages whose
    # id ends with "::Company.Product.Services"
    resolved_row = {
        "id": "repo::Company.Product.Services",
        "symbol_name": "Services",  # leaf only — this is how .NET stores it
        "repository": "repo",
        "symbol_type": "Package",
    }
    members = [
        {"symbol_name": "AuthService", "symbol_type": "Class", "file_path": "Services/AuthService.cs"},
    ]
    child_pkgs: list = []  # no children

    call_count = {"n": 0}
    fetch_sequence = [[resolved_row], members, child_pkgs]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_package_context("Company.Product.Services")

    assert result is not None
    # name should be the full dotted name (not the leaf)
    assert result.name == "Company.Product.Services"
    assert result.package_id == "repo::Company.Product.Services"
    assert "AuthService" in result.classes

    # Verify the resolver query uses ID-suffix matching, not symbol_name=
    first_sql = mock_pool.fetch.call_args_list[0][0][0]
    assert "id LIKE" in first_sql or "id ~" in first_sql or "ENDS" in first_sql, \
        f"Expected id-suffix matching in resolver SQL; got: {first_sql}"


@pytest.mark.asyncio
async def test_get_package_context_enumerates_child_packages_via_id_prefix(backend, mock_pool):
    """Child packages must be discovered via id prefix matching, not via
    package-to-package CONTAINS edges (which .NET parser doesn't create)."""
    resolved_row = {
        "id": "repo::Company.Product",
        "symbol_name": "Product",
        "repository": "repo",
        "symbol_type": "Package",
    }
    members: list = []
    # Child packages: direct children only (Services and API, not Services.Auth)
    child_pkgs = [
        {"id": "repo::Company.Product.Services", "symbol_name": "Services"},
        {"id": "repo::Company.Product.API", "symbol_name": "API"},
    ]

    call_count = {"n": 0}
    fetch_sequence = [[resolved_row], members, child_pkgs]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_package_context("Company.Product")

    assert result is not None
    # Child packages reconstructed to their full dotted names
    assert "Company.Product.Services" in result.child_packages
    assert "Company.Product.API" in result.child_packages

    # Verify child query uses id LIKE parent.id || '.%' pattern
    third_sql = mock_pool.fetch.call_args_list[2][0][0]
    assert "LIKE" in third_sql or "STARTS" in third_sql, \
        f"Expected id-prefix matching for child packages; got: {third_sql}"


@pytest.mark.asyncio
async def test_get_package_context_returns_none_for_nonexistent_package(backend, mock_pool):
    """A non-existent package must return None, not raise."""
    mock_pool.fetch = AsyncMock(return_value=[])

    result = await backend.get_package_context("NonExistent.Namespace")

    assert result is None


@pytest.mark.asyncio
async def test_get_package_context_derives_files_from_member_file_paths(backend, mock_pool):
    """PackageContext.files must contain the DISTINCT file_path of every
    member with an IN_PACKAGE edge — NOT just members whose symbol_type is
    File. Constellation's parsers attach IN_PACKAGE from classes, methods,
    interfaces (etc.), never from file nodes, so the old 'symbol_type = File'
    filter always returned empty. Neo4j derives files from any member's
    file_path (neo4j.py:943), and the live Neo4j contract test explicitly
    asserts Service.java is in package_context.files."""
    resolved_row = {
        "id": "repo::com.example",
        "symbol_name": "example",
        "repository": "repo",
        "symbol_type": "Package",
    }
    # Members: 2 classes in Service.java + 1 class in Util.java + 1 method
    # in Service.java. The result should be DISTINCT file paths.
    members = [
        {"symbol_name": "Service", "symbol_type": "Class", "file_path": "src/Service.java"},
        {"symbol_name": "ServiceHelper", "symbol_type": "Class", "file_path": "src/Service.java"},
        {"symbol_name": "Util", "symbol_type": "Class", "file_path": "src/Util.java"},
        {"symbol_name": "doWork", "symbol_type": "Method", "file_path": "src/Service.java"},
    ]
    child_pkgs: list = []

    call_count = {"n": 0}
    fetch_sequence = [[resolved_row], members, child_pkgs]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_package_context("com.example")

    assert result is not None
    # Files must be DISTINCT file paths from ALL members, sorted for determinism
    assert result.files == ["src/Service.java", "src/Util.java"], \
        f"Expected distinct file paths from member classes/methods; got: {result.files}"
    # Classes still come from the symbol_type filter
    assert "Service" in result.classes
    assert "ServiceHelper" in result.classes
    assert "Util" in result.classes


@pytest.mark.asyncio
async def test_get_codebase_overview_no_repository_with_include_packages(backend, mock_pool):
    """get_codebase_overview(repository=None, include_packages=True) must
    return packages across all repositories."""
    counts = []
    langs = []
    entry_points = []
    top_classes = []
    export_count = [{"cnt": 0}]
    packages = [{"symbol_name": "repo1.pkg"}, {"symbol_name": "repo2.pkg"}]

    call_count = {"n": 0}
    fetch_sequence = [counts, langs, entry_points, top_classes, export_count, packages]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_codebase_overview(repository=None, include_packages=True)

    assert "repo1.pkg" in result.packages
    assert "repo2.pkg" in result.packages


def test_full_name_from_id_strips_repository_prefix():
    """_full_name_from_id strips 'repo::' from the entity id."""
    from telescope.backends.postgres import PostgresReadBackend
    assert PostgresReadBackend._full_name_from_id(
        "my-repo::com.example.Foo.bar", "bar"
    ) == "com.example.Foo.bar"
    # Fallback: no '::' in id → use fallback_name
    assert PostgresReadBackend._full_name_from_id("bar", "bar") == "bar"
    # Empty id → fallback
    assert PostgresReadBackend._full_name_from_id("", "bar") == "bar"


@pytest.mark.asyncio
async def test_get_function_context_populates_full_name_and_class_name(backend, mock_pool):
    """get_function_context must strip the repo prefix from full_name and
    look up the owning class_name."""
    async def fake_resolve(*args, **kwargs):
        return {
            "id": "my-repo::com.example.Service.doIt",
            "symbol_name": "doIt",
            "file_path": "src/Service.java",
            "repository": "my-repo",
            "code": "public void doIt() {}",
            "signature": "public void doIt()",
            "docstring": None,
        }
    backend._resolve_symbol = fake_resolve

    # First fetchrow call: class lookup returns the owning class
    # Any subsequent fetchrow call: fail loudly (there should only be 1)
    fetchrow_calls = {"count": 0}
    async def fetchrow_side_effect(*args, **kwargs):
        fetchrow_calls["count"] += 1
        if fetchrow_calls["count"] == 1:
            return {"symbol_name": "Service"}
        raise AssertionError(
            f"Unexpected fetchrow call #{fetchrow_calls['count']}: {args}"
        )
    mock_pool.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)

    # get_callers / get_callees short-circuit to []
    backend.get_callers = AsyncMock(return_value=[])
    backend.get_callees = AsyncMock(return_value=[])

    result = await backend.get_function_context("doIt")

    assert result is not None
    assert result.full_name == "com.example.Service.doIt"
    assert result.class_name == "Service"
    # Verify the class lookup SQL received both symbol id AND repository
    # as bind parameters (repository scoping from Fix 1)
    fetchrow_args = mock_pool.fetchrow.call_args[0]
    sql = fetchrow_args[0]
    assert "s.repository = $2" in sql, \
        f"Expected repository scoping in class lookup SQL; got: {sql}"
    assert fetchrow_args[1] == "my-repo::com.example.Service.doIt"
    assert fetchrow_args[2] == "my-repo"


@pytest.mark.asyncio
async def test_get_function_context_handles_missing_owning_class(backend, mock_pool):
    """A top-level function (no owning class) should get class_name=None."""
    async def fake_resolve(*args, **kwargs):
        return {
            "id": "my-repo::top_level_fn",
            "symbol_name": "top_level_fn",
            "file_path": "src/utils.py",
            "repository": "my-repo",
            "code": "def top_level_fn(): pass",
            "signature": "def top_level_fn()",
            "docstring": None,
        }
    backend._resolve_symbol = fake_resolve
    mock_pool.fetchrow = AsyncMock(return_value=None)  # no owning class
    backend.get_callers = AsyncMock(return_value=[])
    backend.get_callees = AsyncMock(return_value=[])

    result = await backend.get_function_context("top_level_fn")

    assert result is not None
    assert result.full_name == "top_level_fn"
    assert result.class_name is None


def test_looks_like_symbol_query_heuristic():
    """Mirrors neo4j.py:200 — identifies code-like tokens."""
    from telescope.backends.postgres import PostgresReadBackend as P
    # Positive: CamelCase, snake_case, dotted, path-like, kebab-case
    assert P._looks_like_symbol_query("UserService") is True
    assert P._looks_like_symbol_query("get_user_by_id") is True
    assert P._looks_like_symbol_query("com.example.Foo") is True
    assert P._looks_like_symbol_query("src/foo.py") is True
    assert P._looks_like_symbol_query("some-package") is True
    assert P._looks_like_symbol_query("std::vector") is True
    # Negative: empty, whitespace, multi-word, single lowercase word
    assert P._looks_like_symbol_query("") is False
    assert P._looks_like_symbol_query("  ") is False
    assert P._looks_like_symbol_query("authentication logic") is False
    assert P._looks_like_symbol_query("user") is False
    assert P._looks_like_symbol_query("get") is False


@pytest.mark.asyncio
async def test_search_code_blends_exact_symbol_results(backend, mock_pool):
    """When the query looks like an identifier, search_code must merge in
    exact symbol results alongside the vector search hits."""
    backend._get_embedding = AsyncMock(return_value=[0.0] * 1536)

    # Vector search returns 2 fuzzy matches
    vector_rows = [
        {"id": "repo::Foo.getUser", "symbol_name": "getUser",
         "symbol_type": "Method", "file_path": "Foo.java", "repository": "repo",
         "line_start": 10, "line_end": 20, "signature": "public User getUser()",
         "code": "...", "docstring": None, "language": "java",
         "return_type": "User", "modifiers": [], "stereotypes": [],
         "content_hash": "h", "properties": {}, "score": 0.85},
    ]
    # Symbol search (exact=True on "UserService") returns 1 match
    symbol_rows = [
        {"id": "repo::UserService", "symbol_name": "UserService",
         "symbol_type": "Class", "file_path": "UserService.java", "repository": "repo",
         "line_start": 1, "line_end": 100, "signature": "public class UserService",
         "code": "...", "docstring": None, "language": "java",
         "return_type": None, "modifiers": ["public"], "stereotypes": [],
         "content_hash": "h", "properties": {}},
    ]

    call_count = {"n": 0}
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx == 0:
            return vector_rows
        return symbol_rows
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.search_code("UserService")

    # Must contain BOTH the vector hit AND the exact symbol hit
    names = [e.name for e in result]
    assert "getUser" in names
    assert "UserService" in names
    # Exact symbol match must come BEFORE vector hit in the blended result
    assert names.index("UserService") < names.index("getUser"), \
        "Exact symbol match must appear before vector hit"


@pytest.mark.asyncio
async def test_search_code_applies_code_mode_to_blended_symbol_results(backend, mock_pool):
    """code_mode=none must zero out the `code` field on BOTH vector hits
    AND blended find_symbols results. A regression that only transforms
    the vector path would leak full source code on identifier queries."""
    backend._get_embedding = AsyncMock(return_value=[0.0] * 1536)

    # Vector search returns 1 fuzzy method hit with full code
    vector_rows = [
        {
            "id": "repo::Foo.getUser", "symbol_name": "getUser",
            "symbol_type": "Method", "file_path": "Foo.java", "repository": "repo",
            "line_start": 10, "line_end": 20,
            "signature": "public User getUser()",
            "code": "public User getUser() { return userRepo.find(); }",
            "docstring": None, "language": "java", "return_type": "User",
            "modifiers": [], "stereotypes": [], "content_hash": "h",
            "properties": {}, "score": 0.85,
        },
    ]
    # Symbol search returns 1 exact class hit with full code
    symbol_rows = [
        {
            "id": "repo::UserService", "symbol_name": "UserService",
            "symbol_type": "Class", "file_path": "UserService.java",
            "repository": "repo", "line_start": 1, "line_end": 100,
            "signature": "public class UserService",
            "code": "public class UserService { /* 100 lines of code */ }",
            "docstring": None, "language": "java", "return_type": None,
            "modifiers": ["public"], "stereotypes": [],
            "content_hash": "h", "properties": {},
        },
    ]

    call_count = {"n": 0}
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return vector_rows if idx == 0 else symbol_rows
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    # The query "UserService" is identifier-like → blend path fires
    result = await backend.search_code("UserService", code_mode="none")

    # Both the vector hit AND the symbol hit must have code=None
    assert len(result) == 2
    for entity in result:
        assert entity.code is None, \
            f"code_mode='none' must zero out code on {entity.name}; got: {entity.code!r}"


@pytest.mark.asyncio
async def test_search_code_applies_signature_mode_to_blended_symbol_results(backend, mock_pool):
    """code_mode=signature must replace `code` with `signature` on both
    vector hits and blended symbol results."""
    backend._get_embedding = AsyncMock(return_value=[0.0] * 1536)

    vector_rows = [
        {
            "id": "repo::Foo.getUser", "symbol_name": "getUser",
            "symbol_type": "Method", "file_path": "Foo.java", "repository": "repo",
            "line_start": 10, "line_end": 20,
            "signature": "public User getUser()",
            "code": "public User getUser() { /* body */ }",
            "docstring": None, "language": "java", "return_type": "User",
            "modifiers": [], "stereotypes": [], "content_hash": "h",
            "properties": {}, "score": 0.85,
        },
    ]
    symbol_rows = [
        {
            "id": "repo::UserService", "symbol_name": "UserService",
            "symbol_type": "Class", "file_path": "UserService.java",
            "repository": "repo", "line_start": 1, "line_end": 100,
            "signature": "public class UserService",
            "code": "public class UserService { /* body */ }",
            "docstring": None, "language": "java", "return_type": None,
            "modifiers": ["public"], "stereotypes": [],
            "content_hash": "h", "properties": {},
        },
    ]

    call_count = {"n": 0}
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return vector_rows if idx == 0 else symbol_rows
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.search_code("UserService", code_mode="signature")

    assert len(result) == 2
    # Each entity's code must equal its signature (both transformed)
    for entity in result:
        assert entity.code == entity.signature, \
            f"code_mode='signature' must set code=signature on {entity.name}; got code={entity.code!r}"


@pytest.mark.asyncio
async def test_search_code_applies_preview_mode_to_blended_symbol_results(backend, mock_pool):
    """code_mode='preview' must slice code to the first 10 lines on BOTH
    vector hits and blended symbol results. This covers the third branch
    of _apply_code_mode (the other two are tested above)."""
    backend._get_embedding = AsyncMock(return_value=[0.0] * 1536)

    # Vector hit: 15-line method body
    vector_code = "\n".join(f"line{i}" for i in range(15))
    vector_rows = [
        {
            "id": "repo::Foo.getUser", "symbol_name": "getUser",
            "symbol_type": "Method", "file_path": "Foo.java", "repository": "repo",
            "line_start": 10, "line_end": 25,
            "signature": "public User getUser()",
            "code": vector_code,
            "docstring": None, "language": "java", "return_type": "User",
            "modifiers": [], "stereotypes": [], "content_hash": "h",
            "properties": {}, "score": 0.85,
        },
    ]
    # Symbol hit: 20-line class body
    symbol_code = "\n".join(f"class_line{i}" for i in range(20))
    symbol_rows = [
        {
            "id": "repo::UserService", "symbol_name": "UserService",
            "symbol_type": "Class", "file_path": "UserService.java",
            "repository": "repo", "line_start": 1, "line_end": 100,
            "signature": "public class UserService",
            "code": symbol_code,
            "docstring": None, "language": "java", "return_type": None,
            "modifiers": ["public"], "stereotypes": [],
            "content_hash": "h", "properties": {},
        },
    ]

    call_count = {"n": 0}
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return vector_rows if idx == 0 else symbol_rows
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.search_code("UserService", code_mode="preview")

    assert len(result) == 2
    # Each entity's code must be sliced to at most 10 lines
    for entity in result:
        assert entity.code is not None
        lines = entity.code.splitlines()
        assert len(lines) <= 10, \
            f"code_mode='preview' must limit {entity.name} to 10 lines; got {len(lines)}"
    # Spot-check that the slice took the FIRST 10 lines, not a random 10
    by_name = {e.name: e for e in result}
    assert by_name["getUser"].code == "\n".join(f"line{i}" for i in range(10))
    assert by_name["UserService"].code == "\n".join(f"class_line{i}" for i in range(10))


@pytest.mark.asyncio
async def test_find_symbols_matches_file_path_in_fuzzy_mode(backend, mock_pool):
    """find_symbols fuzzy mode must match BOTH symbol_name AND file_path in an OR."""
    mock_pool.fetch = AsyncMock(return_value=[
        {"id": "repo::src/users.py", "symbol_name": "users.py",
         "symbol_type": "File", "file_path": "src/users.py", "repository": "repo",
         "line_start": 1, "line_end": 100, "signature": None,
         "code": None, "docstring": None, "language": "python",
         "return_type": None, "modifiers": [], "stereotypes": [],
         "content_hash": "h", "properties": {}},
    ])

    result = await backend.find_symbols("users.py")

    assert len(result) == 1
    assert result[0].name == "users.py"

    # Verify the SQL includes BOTH symbol_name AND file_path in an OR
    sql = mock_pool.fetch.call_args[0][0]
    assert "symbol_name ILIKE" in sql, \
        f"Expected symbol_name ILIKE clause in fuzzy find_symbols SQL; got: {sql}"
    assert "file_path ILIKE" in sql, \
        f"Expected file_path ILIKE clause in fuzzy find_symbols SQL; got: {sql}"
    # Verify the two ILIKE clauses are joined by OR, not AND, and appear
    # together in a single grouped clause (not spread across other parts
    # of the query where OR might legitimately appear).
    import re
    # Match: (symbol_name ILIKE $N OR file_path ILIKE $N) — allowing whitespace variation
    pattern = re.compile(
        r"\(\s*symbol_name\s+ILIKE\s+\$\d+\s+OR\s+file_path\s+ILIKE\s+\$\d+\s*\)",
        re.IGNORECASE,
    )
    assert pattern.search(sql), \
        f"Expected '(symbol_name ILIKE $N OR file_path ILIKE $N)' grouped clause; got: {sql}"


@pytest.mark.asyncio
async def test_get_callers_sets_truncated_when_limit_exceeded(backend, mock_pool):
    """get_callers must set truncated=True on every returned node when
    more callers exist than the limit. Currently returns truncated=False
    which gives users no signal."""
    async def fake_resolve(*args, **kwargs):
        return {"id": "repo::MyClass.doIt", "symbol_name": "doIt",
                "file_path": "MyClass.java", "repository": "repo"}
    backend._resolve_symbol = fake_resolve
    backend._resolve_method_family = AsyncMock(return_value=["repo::MyClass.doIt"])

    # Return limit + 1 rows (3 for limit=2)
    mock_pool.fetch = AsyncMock(return_value=[
        {"id": "c1", "symbol_name": "caller1", "file_path": "a.java",
         "repository": "repo", "signature": None, "line_start": 1,
         "symbol_type": "Method", "is_test": False, "is_endpoint": False,
         "depth": 1},
        {"id": "c2", "symbol_name": "caller2", "file_path": "b.java",
         "repository": "repo", "signature": None, "line_start": 2,
         "symbol_type": "Method", "is_test": False, "is_endpoint": False,
         "depth": 1},
        {"id": "c3", "symbol_name": "caller3", "file_path": "c.java",
         "repository": "repo", "signature": None, "line_start": 3,
         "symbol_type": "Method", "is_test": False, "is_endpoint": False,
         "depth": 1},
    ])

    result = await backend.get_callers("doIt", limit=2)

    # Only `limit` items returned (slicing)
    assert len(result) == 2
    # Every returned node signals truncation
    assert all(node.truncated is True for node in result), \
        f"Expected all nodes to have truncated=True; got: {[n.truncated for n in result]}"


@pytest.mark.asyncio
async def test_get_callers_does_not_set_truncated_when_under_limit(backend, mock_pool):
    """When the result count is <= limit, truncated must be False."""
    async def fake_resolve(*args, **kwargs):
        return {"id": "repo::MyClass.doIt", "symbol_name": "doIt",
                "file_path": "MyClass.java", "repository": "repo"}
    backend._resolve_symbol = fake_resolve
    backend._resolve_method_family = AsyncMock(return_value=["repo::MyClass.doIt"])

    # Return 1 row (less than limit=50)
    mock_pool.fetch = AsyncMock(return_value=[
        {"id": "c1", "symbol_name": "caller1", "file_path": "a.java",
         "repository": "repo", "signature": None, "line_start": 1,
         "symbol_type": "Method", "is_test": False, "is_endpoint": False,
         "depth": 1},
    ])

    result = await backend.get_callers("doIt", limit=50)

    assert len(result) == 1
    assert result[0].truncated is False


@pytest.mark.asyncio
async def test_get_callees_signals_truncation_when_combined_exceeds_limit(backend, mock_pool):
    """Matches Neo4j semantics: calls come first in the merged list, then
    hooks. When the combined count exceeds limit, the result is sliced to
    limit and truncated=True. If calls alone fill the limit, hooks may be
    dropped — but the caller gets truncated=True as the signal.

    The old implementation fetched calls with LIMIT limit, hooks unlimited,
    then sliced — producing silently wrong results with truncated=False.
    The new implementation over-fetches both, dedupes, and sets truncated
    correctly.
    """
    async def fake_resolve(*args, **kwargs):
        return {"id": "repo::MyClass.doIt", "symbol_name": "doIt",
                "file_path": "MyClass.java", "repository": "repo"}
    backend._resolve_symbol = fake_resolve
    backend._resolve_method_family = AsyncMock(return_value=["repo::MyClass.doIt"])

    # 3 calls + 1 hook = 4 total, limit=3 → truncated
    call_rows = [
        {"id": f"call{i}", "symbol_name": f"callee_{i}", "file_path": "x.java",
         "repository": "repo", "signature": None, "line_start": i,
         "symbol_type": "Method", "is_test": False, "is_endpoint": False,
         "depth": 1}
        for i in range(3)
    ]
    hook_rows = [
        {"id": "hook1", "symbol_name": "useAuth", "file_path": "x.java",
         "repository": "repo", "signature": None, "line_start": 10,
         "symbol_type": "Hook", "is_test": False, "is_endpoint": False,
         "depth": 1},
    ]

    call_count = {"n": 0}
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return call_rows if idx == 0 else hook_rows
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_callees("doIt", limit=3)

    # Total length is exactly limit
    assert len(result) == 3
    # Calls come first in the merged order, filling the limit
    assert all(n.relationship_type == "CALLS" for n in result), \
        "Calls should come before hooks in the merged result"
    # Truncated is True because the combined count (4) > limit (3)
    assert all(n.truncated is True for n in result), \
        f"Expected truncated=True on all nodes; got: {[(n.name, n.truncated) for n in result]}"


@pytest.mark.asyncio
async def test_get_callees_limit_one_returns_single_call(backend, mock_pool):
    """Edge case: limit=1 with one call and one hook returns exactly
    the first call, with truncated=True signaling that more exist."""
    async def fake_resolve(*args, **kwargs):
        return {"id": "repo::MyClass.doIt", "symbol_name": "doIt",
                "file_path": "MyClass.java", "repository": "repo"}
    backend._resolve_symbol = fake_resolve
    backend._resolve_method_family = AsyncMock(return_value=["repo::MyClass.doIt"])

    call_rows = [
        {"id": "call0", "symbol_name": "callee_0", "file_path": "x.java",
         "repository": "repo", "signature": None, "line_start": 1,
         "symbol_type": "Method", "is_test": False, "is_endpoint": False,
         "depth": 1},
    ]
    hook_rows = [
        {"id": "hook1", "symbol_name": "useAuth", "file_path": "x.java",
         "repository": "repo", "signature": None, "line_start": 10,
         "symbol_type": "Hook", "is_test": False, "is_endpoint": False,
         "depth": 1},
    ]

    call_count = {"n": 0}
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return call_rows if idx == 0 else hook_rows
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_callees("doIt", limit=1)

    assert len(result) == 1
    # Matches Neo4j: calls come first; with limit=1 the call wins the slot
    assert result[0].relationship_type == "CALLS"
    assert result[0].name == "callee_0"
    assert result[0].truncated is True


@pytest.mark.asyncio
async def test_get_hook_usage_sets_truncated_when_limit_exceeded(backend, mock_pool):
    """get_hook_usage must signal truncation when more hook users exist
    than the limit."""
    # limit=2, return 3 rows
    mock_pool.fetch = AsyncMock(return_value=[
        {"id": f"u{i}", "symbol_name": f"user_{i}", "file_path": "x.tsx",
         "repository": "repo", "signature": None, "line_start": i,
         "symbol_type": "Method", "is_test": False, "is_endpoint": False,
         "depth": 1}
        for i in range(3)
    ])

    result = await backend.get_hook_usage("useAuth", limit=2)

    assert len(result) == 2
    assert all(n.truncated is True for n in result)
