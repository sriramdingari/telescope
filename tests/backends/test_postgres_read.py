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


def test_parameter_suffix_from_entity_id_extracts_suffix():
    """The helper must extract the parameter suffix from a Java/C# entity id.

    Mirrors Neo4j's helper (neo4j.py:~402-409). Takes ONLY the entity_id —
    no method_name argument needed, because we just look for the trailing
    parenthesized tail.
    """
    from telescope.backends.postgres import PostgresReadBackend as P

    # Java overloaded method on a class
    assert P._parameter_suffix_from_entity_id(
        "repo::com.example.Foo.process(String,int)"
    ) == "(String,int)"

    # Parameterless method
    assert P._parameter_suffix_from_entity_id(
        "repo::com.example.Foo.process()"
    ) == "()"

    # ID with no suffix (Python, JavaScript — no overloading): returns None
    assert P._parameter_suffix_from_entity_id(
        "repo::com.example.Foo.process"
    ) is None

    # Top-level function in Java/C# (no class): still has a suffix
    assert P._parameter_suffix_from_entity_id(
        "repo::process(String)"
    ) == "(String)"

    # Empty id
    assert P._parameter_suffix_from_entity_id("") is None

    # Id with parens in the middle but not at the end
    assert P._parameter_suffix_from_entity_id(
        "repo::com.example.Foo(inner).method"
    ) is None


@pytest.mark.asyncio
async def test_resolve_method_family_filters_by_parameter_suffix(backend, mock_pool):
    """When the starting method has a parameter suffix, the family query
    must filter sibling methods to those with the SAME suffix. Otherwise
    overloaded methods pollute the family set.

    This test simulates the SQL filter behavior inside the mock's
    side_effect: it inspects the bind parameters and returns only rows
    whose id actually satisfies the right()/length() end-with check.
    A buggy implementation that passes the wrong suffix, omits the
    filter clause, or uses `LIKE` without escaping would produce the
    wrong set of family_ids and fail the assertions."""
    # Starting symbol has suffix "(String,int)"
    symbol = {
        "id": "repo::FooImpl.process(String,int)",
        "symbol_name": "process",
    }

    # The "database" contains both overloads on both Foo and FooImpl.
    # The filter should return only rows ending in process(String,int).
    all_overload_rows = [
        {"id": "repo::IFoo.process(String,int)"},
        {"id": "repo::IFoo.process(String)"},            # wrong overload
        {"id": "repo::FooImpl.process(String,int)"},
        {"id": "repo::FooImpl.process(String)"},         # wrong overload
        {"id": "repo::FooSub.process(String,int)"},
        {"id": "repo::FooSub.process(String)"},          # wrong overload
    ]

    async def fetch_side_effect(sql, *args):
        # args: (symbol_id, symbol_name, parameter_suffix)
        assert len(args) == 3, f"Expected 3 args; got {len(args)}"
        name = args[1]
        suffix = args[2]
        # Verify the SQL uses the right()/length() end-with check, NOT LIKE
        assert "right(m.id" in sql or "RIGHT(m.id" in sql, \
            f"Expected right(m.id, ...) suffix filter; got: {sql}"
        assert "LIKE" not in sql or "'%::'" in sql, \
            f"Expected end-with via right(), not LIKE pattern; got: {sql}"

        # Simulate the SQL filter behavior
        if suffix is None:
            return all_overload_rows
        target_tail = name + suffix
        return [r for r in all_overload_rows if r["id"].endswith(target_tail)]

    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    family_ids = await backend._resolve_method_family(symbol)

    # MUST contain only the matching overload's ids
    expected = {
        "repo::IFoo.process(String,int)",
        "repo::FooImpl.process(String,int)",
        "repo::FooSub.process(String,int)",
    }
    # The starting symbol is always appended if missing (helper contract)
    assert set(family_ids) >= expected, \
        f"Expected family to contain matching overloads; got: {family_ids}"
    # MUST NOT contain any wrong-overload ids
    wrong = {
        "repo::IFoo.process(String)",
        "repo::FooImpl.process(String)",
        "repo::FooSub.process(String)",
    }
    assert not (set(family_ids) & wrong), \
        f"Family should exclude wrong overloads; got: {family_ids}"


@pytest.mark.asyncio
async def test_resolve_method_family_no_filter_when_suffix_absent(backend, mock_pool):
    """Python-style IDs with no parameter suffix should match all same-name
    methods (no overload filter applied). The mock simulates the SQL
    behavior: when $3 is None, all rows are returned regardless of suffix."""
    symbol = {
        "id": "repo::foo.bar",  # no suffix
        "symbol_name": "bar",
    }

    all_rows = [
        {"id": "repo::Foo.bar"},
        {"id": "repo::FooSub.bar"},
    ]

    async def fetch_side_effect(sql, *args):
        assert len(args) == 3
        name = args[1]
        suffix = args[2]
        assert suffix is None, \
            f"Expected parameter_suffix=None for Python-style ID; got: {suffix}"
        return all_rows

    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    family_ids = await backend._resolve_method_family(symbol)

    # Both same-name methods should be included (no suffix filter)
    assert "repo::Foo.bar" in family_ids
    assert "repo::FooSub.bar" in family_ids


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
    """get_file_context must actually return exports, not hardcode [].

    Updated for the new 6-entry fetch order after Plan B Task 1's
    shared-file query rewrite. The previous test version had a
    references slot that encoded the broken USES_TYPE assumption;
    that slot is dropped because the new query doesn't traverse
    USES_TYPE for references. References parity is now covered by
    test_get_file_context_returns_constructors_fields_references_via_shared_file
    below.
    """
    file_row = {
        "id": "repo::src/foo.py",
        "symbol_name": "foo.py",
        "file_path": "src/foo.py",
        "repository": "repo",
        "language": "python",
        "content_hash": "h1",
    }
    # New 6-entry fetch order:
    # 1. resolver → file_row
    # 2. shared-file members → empty (no class-scoped members in this test)
    # 3. top-level methods (File→CONTAINS→Method) → empty (helper is
    #    surfaced via EXPORTS in this test, not as a top-level method)
    # 4. exports (File→EXPORTS→*) → the helper export row
    # 5. packages (shared-file IN_PACKAGE) → com.example
    # 6. hooks (shared-file Hook) → empty
    fetch_results = [
        [file_row],  # 1. file resolver
        [],          # 2. shared-file members
        [],          # 3. top-level methods
        [{"id": "repo::src/foo.py::helper",  # 4. exports
          "symbol_name": "helper", "symbol_type": "Method",
          "file_path": "src/foo.py", "repository": "repo",
          "line_start": 5, "line_end": 10, "signature": "def helper()",
          "code": "def helper(): pass", "docstring": None,
          "language": "python", "return_type": None,
          "modifiers": [], "stereotypes": [], "content_hash": "h1",
          "properties": {}}],
        [{"id": "repo::com.example", "symbol_name": "com.example"}],  # 5. packages
        [],          # 6. hooks
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


@pytest.mark.asyncio
async def test_get_file_context_returns_constructors_fields_references_via_shared_file(
    backend, mock_pool
):
    """Java/C# class-scoped members (constructors, fields, references) are
    attached via HAS_METHOD/HAS_CONSTRUCTOR/HAS_FIELD edges from Class, NOT
    File→CONTAINS. For Java unresolved calls, Reference entities are attached
    via CALLS from Methods, NOT USES_TYPE. Both patterns are invisible to
    an edge-traversal query rooted at File.

    Neo4j sidesteps this by matching members on shared repository +
    file_path regardless of edge type (neo4j.py:1216-1227). This test
    asserts the Postgres implementation does the same: given a file row
    and a set of shared-file member rows, the FileContext must surface
    constructors, fields, and references even though no File→CONTAINS
    edge exists for them.
    """
    file_row = {
        "id": "repo::src/Service.java",
        "symbol_name": "Service.java",
        "file_path": "src/Service.java",
        "repository": "repo",
        "language": "java",
        "content_hash": "h",
    }
    # Shared-file members — what the new query returns. Mix of entity
    # types a real Java file would produce.
    shared_file_members = [
        {"symbol_name": "Service", "symbol_type": "Class"},
        {"symbol_name": "Service", "symbol_type": "Constructor"},  # overload of class name
        {"symbol_name": "client", "symbol_type": "Field"},
        {"symbol_name": "client.fetch", "symbol_type": "Reference"},
    ]
    # Top-level methods (File→CONTAINS→Method) — empty for Java classes
    top_level_methods: list = []
    exports: list = []
    package_rows = [
        {"id": "repo::com.example", "symbol_name": "com.example"},
    ]
    hook_rows: list = []

    call_count = {"n": 0}
    fetch_sequence = [
        [file_row],          # 1. file resolver
        shared_file_members, # 2. shared-file members
        top_level_methods,   # 3. top_level_methods
        exports,             # 4. exports
        package_rows,        # 5. packages
        hook_rows,           # 6. hooks
    ]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_file_context("Service.java")

    assert result is not None
    assert "Service" in result.classes
    assert "Service" in result.constructors, \
        f"Expected Service constructor from shared-file member; got: {result.constructors}"
    assert "client" in result.fields, \
        f"Expected client field from shared-file member; got: {result.fields}"
    assert "client.fetch" in result.references, \
        f"Expected client.fetch reference from shared-file member; got: {result.references}"


@pytest.mark.asyncio
async def test_get_file_context_dedupes_overloaded_constructors(backend, mock_pool):
    """For Java overloaded constructors, the FileContext.constructors list
    must contain each name only once (matching Neo4j's collect(DISTINCT
    cls.name) semantics). A regression that drops the Python-side dedup
    would surface here as constructors == ['Service', 'Service']."""
    file_row = {
        "id": "repo::src/Service.java",
        "symbol_name": "Service.java",
        "file_path": "src/Service.java",
        "repository": "repo",
        "language": "java",
    }
    # Two overloaded constructors with the same name
    shared_file_members = [
        {"symbol_name": "Service", "symbol_type": "Class"},
        {"symbol_name": "Service", "symbol_type": "Constructor"},
        {"symbol_name": "Service", "symbol_type": "Constructor"},  # overload
    ]

    call_count = {"n": 0}
    fetch_sequence = [
        [file_row], shared_file_members, [], [], [], [],
    ]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_file_context("Service.java")

    assert result is not None
    assert result.constructors == ["Service"], \
        f"Expected dedup'd constructors == ['Service']; got: {result.constructors}"
    assert result.classes == ["Service"]


@pytest.mark.asyncio
async def test_get_file_context_uses_shared_file_sql_for_members(backend, mock_pool):
    """Lock in the SQL shape: the members query must filter by
    `repository = $1 AND file_path = $2`, not by a File→CONTAINS edge
    traversal. A regression back to edge traversal would silently lose
    Java constructors/fields/references.
    """
    file_row = {
        "id": "repo::src/Service.java",
        "symbol_name": "Service.java",
        "file_path": "src/Service.java",
        "repository": "repo",
        "language": "java",
    }

    captured_sqls: list[str] = []
    async def fetch_side_effect(sql, *args, **kwargs):
        captured_sqls.append(sql)
        if len(captured_sqls) == 1:
            return [file_row]
        return []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    await backend.get_file_context("Service.java")

    members_sqls = [
        sql for sql in captured_sqls
        if "symbol_type IN" in sql and "Constructor" in sql and "Field" in sql
    ]
    assert members_sqls, \
        f"Expected a members query with symbol_type IN (...Constructor, Field...); got SQLs: {captured_sqls}"
    members_sql = members_sqls[0]

    assert "file_path = $" in members_sql, \
        f"Expected shared-file match `file_path = $N`; got: {members_sql}"
    assert "ref_type = 'CONTAINS'" not in members_sql, \
        f"Members query should not filter by CONTAINS edge; got: {members_sql}"
    # Lock in the exact shape: single-table scan on code_symbols filtered
    # by (repository, file_path, symbol_type). A rewrite that drops the
    # repository filter would silently cross-contaminate repos sharing
    # the same file_path. A rewrite that joins through code_references
    # with a different edge type would defeat the parser-agnostic goal.
    assert "repository = $" in members_sql, \
        f"Members query must filter by repository; got: {members_sql}"
    assert "JOIN code_references" not in members_sql, \
        f"Members query must be a single-table shared-file select; got: {members_sql}"


@pytest.mark.asyncio
async def test_get_file_context_packages_query_uses_shared_file_member(backend, mock_pool):
    """The packages query must join through shared-file members, not through
    File→CONTAINS→member, so that IN_PACKAGE edges from Field/Constructor/
    Reference members (not just Classes) are also discovered. Mirrors Neo4j
    at neo4j.py:1205-1207. A regression back to File→CONTAINS traversal
    would silently drop package membership signaled through non-Class
    members.
    """
    file_row = {
        "id": "repo::src/Service.java",
        "symbol_name": "Service.java",
        "file_path": "src/Service.java",
        "repository": "repo",
        "language": "java",
    }

    captured_sqls: list[str] = []
    async def fetch_side_effect(sql, *args, **kwargs):
        captured_sqls.append(sql)
        if len(captured_sqls) == 1:
            return [file_row]
        return []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    await backend.get_file_context("Service.java")

    package_sqls = [
        sql for sql in captured_sqls
        if "IN_PACKAGE" in sql and "Package" in sql
    ]
    assert package_sqls, f"Expected a packages query; got SQLs: {captured_sqls}"
    package_sql = package_sqls[0]

    assert "file_path = $" in package_sql, \
        f"Expected shared-file packages query; got: {package_sql}"
    # Plan C requires pkg.id in the SELECT for full-name reconstruction
    # via _full_name_from_id. Lock that in here so a future refactor
    # that "tidies up unused columns" doesn't accidentally break Plan C.
    assert "pkg.id" in package_sql, \
        f"pkg.id must be in SELECT — Plan C reconstruction requires it; got: {package_sql}"


@pytest.mark.asyncio
async def test_get_file_context_packages_reconstructs_nested_dotnet_namespace(
    backend, mock_pool
):
    """file_context.packages must reconstruct full dotted names from the
    entity id, not return the raw leaf-only symbol_name. Constellation's
    .NET parser stores 'Services' in name and 'Company.Product.Services'
    in id (dotnet.py:141). Returning the leaf produces confusing output
    for users inspecting file context.

    Mirrors Task 2's get_package_context fix — same _full_name_from_id
    reconstruction pattern."""
    file_row = {
        "id": "repo::src/AuthService.cs",
        "symbol_name": "AuthService.cs",
        "file_path": "src/AuthService.cs",
        "repository": "repo",
        "language": "csharp",
        "content_hash": "h",
    }
    # Shared-file members: a single Class
    shared_file_members = [
        {"symbol_name": "AuthService", "symbol_type": "Class"},
    ]
    top_level_methods: list = []
    exports: list = []
    # The packages query returns both id and symbol_name. The leaf is
    # what the .NET parser stores; the id holds the full path.
    package_rows = [
        {"id": "repo::Company.Product.Services", "symbol_name": "Services"},
    ]
    hook_rows: list = []

    call_count = {"n": 0}
    fetch_sequence = [
        [file_row],          # 1. resolver
        shared_file_members, # 2. members
        top_level_methods,   # 3. top-level methods
        exports,             # 4. exports
        package_rows,        # 5. packages
        hook_rows,           # 6. hooks
    ]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_file_context("AuthService.cs")

    assert result is not None
    assert result.packages == ["Company.Product.Services"], \
        f"Expected full dotted namespace; got: {result.packages}"


@pytest.mark.asyncio
async def test_get_file_context_packages_java_full_name_is_idempotent(
    backend, mock_pool
):
    """For Java packages, symbol_name IS the full dotted name, so
    reconstruction via _full_name_from_id must be idempotent. This
    guards against a fix that over-normalizes and breaks Java.
    """
    file_row = {
        "id": "repo::src/Service.java",
        "symbol_name": "Service.java",
        "file_path": "src/Service.java",
        "repository": "repo",
        "language": "java",
    }
    shared_file_members = [
        {"symbol_name": "Service", "symbol_type": "Class"},
    ]
    package_rows = [
        # Java package: id has full name, symbol_name also has full name
        {"id": "repo::com.example", "symbol_name": "com.example"},
    ]
    fetch_sequence = [
        [file_row],
        shared_file_members,
        [],  # top_level_methods
        [],  # exports
        package_rows,
        [],  # hooks
    ]
    call_count = {"n": 0}
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_file_context("Service.java")

    assert result is not None
    assert result.packages == ["com.example"], \
        f"Java full-name reconstruction must be idempotent; got: {result.packages}"


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
    packages = [
        {"id": "repo::com.example.app", "symbol_name": "com.example.app"},
        {"id": "repo::com.example.util", "symbol_name": "com.example.util"},
    ]

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
async def test_get_codebase_overview_reconstructs_nested_dotnet_namespaces(
    backend, mock_pool
):
    """codebase_overview.packages must reconstruct full dotted names
    from id, not return raw leaf-only symbol_name. Same _full_name_from_id
    pattern as get_file_context.packages and get_package_context.

    Mirrors the 6-entry fetch_sequence used by the existing test
    `test_get_codebase_overview_include_packages_returns_names` at
    test_postgres_read.py:495-516, plus the new `id` field in each
    package row.
    """
    counts: list = []
    langs: list = []
    entry_points: list = []
    top_classes: list = []
    export_count = [{"cnt": 0}]
    # .NET nested namespace scenario: parser stores the full path in id
    # and the leaf segment in symbol_name. The fix must reconstruct the
    # full dotted name via _full_name_from_id(id, symbol_name).
    packages = [
        {"id": "repo::Company", "symbol_name": "Company"},
        {"id": "repo::Company.Product", "symbol_name": "Product"},
        {"id": "repo::Company.Product.Services", "symbol_name": "Services"},
    ]

    call_count = {"n": 0}
    fetch_sequence = [counts, langs, entry_points, top_classes, export_count, packages]
    async def fetch_side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else []
    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_codebase_overview(
        repository="repo", include_packages=True
    )

    # Full dotted names, not leaves
    assert "Company.Product.Services" in result.packages, \
        f"Expected full dotted nested namespace; got: {result.packages}"
    assert "Services" not in result.packages, \
        f"Leaf-only 'Services' should not appear; got: {result.packages}"
    # Top-level namespace is idempotent (id and name match)
    assert "Company" in result.packages
    # Middle-level namespace gets reconstructed too
    assert "Company.Product" in result.packages


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
    packages = [
        {"id": "repo1::repo1.pkg", "symbol_name": "repo1.pkg"},
        {"id": "repo2::repo2.pkg", "symbol_name": "repo2.pkg"},
    ]

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


@pytest.mark.asyncio
async def test_connect_registers_jsonb_codec_alongside_pgvector(monkeypatch):
    """PostgresReadBackend.connect must register a JSONB type codec on
    every pool connection so asyncpg auto-decodes the `properties` JSONB
    column into Python dicts instead of raw JSON strings. Without this,
    _row_to_code_entity crashes with `dict(str)` on real Postgres data.

    This test mocks asyncpg.create_pool to capture the init callback,
    then invokes it with a mock connection and verifies both
    register_vector (for pgvector) AND set_type_codec (for JSONB) are
    called on that connection.
    """
    # Capture the init callback asyncpg.create_pool receives
    captured_init = {}
    mock_pool = MagicMock()

    async def fake_create_pool(*args, **kwargs):
        captured_init["fn"] = kwargs.get("init")
        return mock_pool

    monkeypatch.setattr("asyncpg.create_pool", fake_create_pool)

    # Mock pgvector.register_vector so we can verify it was called
    register_vector_mock = AsyncMock()
    monkeypatch.setattr("pgvector.asyncpg.register_vector", register_vector_mock)

    backend = PostgresReadBackend(
        dsn="postgresql://test:test@localhost/test",
        openai_api_key="sk-test-key",
    )
    await backend.connect()

    # Verify create_pool was called with an init callback
    init_fn = captured_init.get("fn")
    assert init_fn is not None, "connect() must pass an init callback to create_pool"

    # Simulate the pool calling the init with a fresh connection
    mock_conn = AsyncMock()
    await init_fn(mock_conn)

    # Must register pgvector for the vector column
    register_vector_mock.assert_called_once_with(mock_conn)

    # Must register a jsonb codec for the properties column
    mock_conn.set_type_codec.assert_called()
    codec_calls = mock_conn.set_type_codec.call_args_list
    jsonb_call = next(
        (c for c in codec_calls if c.args[0] == "jsonb" or c.kwargs.get("typename") == "jsonb"),
        None,
    )
    assert jsonb_call is not None, (
        f"Expected set_type_codec to be called for 'jsonb'; got calls: {codec_calls}"
    )
    # Verify the codec uses json.loads for decoding (the whole point — asyncpg
    # gives us a JSON string, we need a dict back)
    import json
    kwargs = jsonb_call.kwargs
    assert kwargs.get("decoder") is json.loads, \
        f"JSONB codec must decode via json.loads; got decoder={kwargs.get('decoder')}"
    assert kwargs.get("schema") == "pg_catalog", \
        f"JSONB codec should be scoped to pg_catalog.jsonb; got schema={kwargs.get('schema')}"


@pytest.mark.asyncio
async def test_get_file_context_resolves_absolute_path_by_suffix(backend, mock_pool):
    """When Constellation stores absolute paths but the user passes a
    relative suffix, get_file_context must resolve via suffix match.
    Mirrors Neo4j's f.file_path ENDS WITH $file_path at neo4j.py:484.
    A regression that reverts to `file_path = $1` would break every
    contract test because Constellation stores absolute paths."""
    # Constellation stores the absolute path; user passes the relative suffix
    absolute_path = "/abs/repo/src/Service.java"
    resolved_row = {
        "id": "repo::File::/abs/repo/src/Service.java",
        "symbol_name": "Service.java",
        "file_path": absolute_path,
        "symbol_type": "File",
        "repository": "repo",
        "content_hash": "abc",
        "language": "java",
    }

    # Simulate the SQL filter: only return rows whose file_path ends with
    # the bind value. A buggy `file_path = $1` implementation would return
    # an empty list here (no exact match) and the test would fail.
    call_count = {"n": 0}

    async def fetch_side_effect(sql, *args):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx == 0:
            # First call: resolver. Must use right()/length() or some
            # suffix-matching SQL, NOT exact equality.
            assert "right(file_path, length($1)) = $1" in sql, \
                f"Expected 'right(file_path, length($1)) = $1' suffix filter with correct bind; got: {sql}"
            # The first bind is the user-provided suffix
            suffix = args[0]
            # Simulate the filter behavior
            if absolute_path.endswith(suffix):
                return [resolved_row]
            return []
        # Downstream queries (contained, exports, packages, hooks, references)
        # — empty results are fine for this test; we only care that the
        # resolver succeeded and returned a FileContext.
        return []

    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    result = await backend.get_file_context("src/Service.java")

    # Should have resolved and returned a FileContext
    assert result is not None, \
        "get_file_context must resolve an absolute path by suffix match"


@pytest.mark.asyncio
async def test_get_file_context_raises_on_ambiguous_suffix(backend, mock_pool):
    """When multiple files match the suffix, must raise ValueError with
    an actionable message that mentions the matching repositories.
    LIMIT 2 + raise pattern matches Neo4j's behavior at neo4j.py:502."""
    row1 = {
        "id": "repo1::File::/a/src/util.py",
        "symbol_name": "util.py",
        "file_path": "/a/src/util.py",
        "symbol_type": "File",
        "repository": "repo1",
    }
    row2 = {
        "id": "repo2::File::/b/src/util.py",
        "symbol_name": "util.py",
        "file_path": "/b/src/util.py",
        "symbol_type": "File",
        "repository": "repo2",
    }

    async def fetch_side_effect(sql, *args):
        # Return both rows — simulating LIMIT 2 fetch that found 2 matches
        return [row1, row2]

    mock_pool.fetch = AsyncMock(side_effect=fetch_side_effect)

    with pytest.raises(ValueError) as excinfo:
        await backend.get_file_context("util.py")

    msg = str(excinfo.value)
    assert "mbiguous" in msg, f"Expected 'ambiguous' in error; got: {msg}"
    # The new error message lists the actual matching file paths, not just repositories
    assert "/a/src/util.py" in msg, \
        f"Expected matching file path in error message; got: {msg}"
    assert "/b/src/util.py" in msg, \
        f"Expected all matching file paths in error message; got: {msg}"


@pytest.mark.asyncio
async def test_get_file_context_returns_none_when_no_match(backend, mock_pool):
    """When no file matches the suffix, return None (not raise).
    Matches Neo4j's `if not results: return None` at neo4j.py:500."""
    mock_pool.fetch = AsyncMock(return_value=[])

    result = await backend.get_file_context("nonexistent/path.py")

    assert result is None


@pytest.mark.postgres_integration
async def test_seeded_postgres_fixture_uses_relative_file_paths(
    seeded_postgres_contract_repository,
    postgres_dsn,
):
    """Regression guard for contract-fixture normalization.

    Production's IndexingPipeline normalizes parser-emitted absolute
    file_paths to repository-relative paths via _normalize_parse_result
    (constellation/indexer/pipeline.py:352). The contract fixtures used
    to skip this step and seed absolute paths, which meant the contract
    suite validated a parser-shaped graph instead of the production
    graph.

    This test connects directly to the seeded Postgres container and
    asserts that no stored file_path starts with '/' — i.e., every
    file_path has been rewritten from absolute to relative. Any Class,
    Method, or File row with a leading-slash file_path indicates the
    fixture is not running normalization.
    """
    import asyncpg
    repository = seeded_postgres_contract_repository

    conn = await asyncpg.connect(postgres_dsn)
    try:
        rows = await conn.fetch(
            "SELECT id, symbol_type, file_path FROM code_symbols "
            "WHERE repository = $1 AND file_path IS NOT NULL "
            "AND file_path LIKE '/%'",
            repository,
        )
    finally:
        await conn.close()

    assert rows == [], (
        f"Expected all file_paths to be relative (normalized), but found "
        f"{len(rows)} rows with absolute paths. Examples: "
        f"{[(r['id'], r['file_path']) for r in rows[:3]]}"
    )
