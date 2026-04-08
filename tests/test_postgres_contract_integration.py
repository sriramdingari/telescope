"""Live contract tests against a real Postgres + pgvector database seeded
by Constellation's real parsers and PostgresWriteBackend.

Mirrors the Neo4j contract suite (tests/test_contract_integration.py) but
runs Telescope's PostgresReadBackend against a pgvector container that
was populated by Constellation's real Postgres write path. Proves the
full Constellation→Postgres→Telescope contract end-to-end.

Schema and fixture data are owned by Constellation. This file only
asserts on the read-path behavior.
"""
import pytest
import pytest_asyncio

from telescope.backends.postgres import PostgresReadBackend

# Skip the module entirely if testcontainers isn't installed
_skip_reason: str | None = None
try:
    from testcontainers.postgres import PostgresContainer  # noqa: F401
except ImportError as exc:
    _skip_reason = f"testcontainers not installed: {exc}"

pytestmark = [pytest.mark.postgres_integration]
if _skip_reason:
    pytestmark.append(pytest.mark.skip(reason=_skip_reason))


@pytest_asyncio.fixture
async def pg_read_backend(postgres_dsn):
    """Telescope's PostgresReadBackend connected to the session container.

    Uses a dummy OpenAI key because contract tests exercise the graph
    read path directly; semantic search is covered by the mocked unit
    tests in tests/backends/test_postgres_read.py.
    """
    backend = PostgresReadBackend(
        dsn=postgres_dsn,
        openai_api_key="sk-test-contract-key",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
    )
    await backend.connect()
    yield backend
    await backend.close()


class TestPostgresContractHappyPaths:
    """Baseline contract parity — mirrors the Neo4j contract tests at
    tests/test_contract_integration.py:10 against the same fixture repo."""

    async def test_repository_and_package_context(
        self, pg_read_backend, seeded_postgres_contract_repository,
    ):
        repository = seeded_postgres_contract_repository

        repo_context = await pg_read_backend.get_repository_context(repository)
        assert repo_context is not None
        assert repo_context.name == repository
        # Same shape the Neo4j contract asserts: 2 files in the fixture repo
        assert repo_context.total_files == 2
        assert "java" in repo_context.languages
        assert "javascript" in repo_context.languages

        package_context = await pg_read_backend.get_package_context(
            "com.example", repository=repository,
        )
        assert package_context is not None
        assert package_context.name == "com.example"
        assert package_context.repository == repository
        assert "Service" in package_context.classes
        # Task 2 parity check: files come from member.file_path, not from
        # File-typed members. The Neo4j contract at
        # test_contract_integration.py:28 asserts the same.
        assert any(path.endswith("Service.java") for path in package_context.files), \
            f"Expected Service.java in package files; got: {package_context.files}"

    async def test_file_context_returns_contained_class(
        self, pg_read_backend, seeded_postgres_contract_repository,
    ):
        repository = seeded_postgres_contract_repository
        fc = await pg_read_backend.get_file_context(
            "src/Service.java", repository=repository,
        )
        assert fc is not None
        assert "Service" in fc.classes


class TestPostgresContractTaskCoverage:
    """Contract coverage for the exact bugs fixed by Tasks 1, 2, 3, 4.
    A regression in any of those tasks should surface here against real
    Constellation parser output + real Postgres storage."""

    async def test_task2_package_files_from_member_file_paths(
        self, pg_read_backend, seeded_postgres_contract_repository,
    ):
        """Task 2: PackageContext.files must contain member file paths.
        Derived from any member with IN_PACKAGE, not just File-typed
        symbols. The Neo4j contract already asserts this at
        test_contract_integration.py:28."""
        repository = seeded_postgres_contract_repository
        pkg = await pg_read_backend.get_package_context(
            "com.example", repository=repository,
        )
        assert pkg is not None
        assert len(pkg.files) > 0, \
            "Expected at least one file in package; got empty list"
        assert any("Service.java" in f for f in pkg.files)

    async def test_task4_overloaded_methods_resolve_with_suffix(
        self, pg_read_backend, seeded_postgres_contract_repository,
    ):
        """Task 4: if the fixture repo contains overloaded methods, resolving
        by name alone must either raise on ambiguity OR resolve to exactly
        one overload and filter the family to that overload's id. The wrong
        outcome is silently merging both overloads' callers/callees.

        Behavior depends on what's actually in the fixture repo. If no
        overloads exist, this becomes a parity guard for when overloads
        get added."""
        repository = seeded_postgres_contract_repository
        results = await pg_read_backend.find_symbols(
            "", repository=repository, exact=False, limit=200,
        )
        methods = [r for r in results if r.entity_type == "method"]
        method_names = [m.name for m in methods]
        duplicate_names = {n for n in method_names if method_names.count(n) > 1}
        if duplicate_names:
            example = next(iter(duplicate_names))
            try:
                ctx = await pg_read_backend.get_function_context(
                    example, repository=repository,
                )
                if ctx is not None:
                    assert "(" in ctx.full_name and ")" in ctx.full_name, (
                        f"Overloaded method {example} resolved without "
                        f"parameter suffix in full_name; got: {ctx.full_name}"
                    )
            except ValueError as exc:
                assert "mbiguous" in str(exc)

    async def test_task3_get_callers_propagates_truncated_correctly(
        self, pg_read_backend, seeded_postgres_contract_repository,
    ):
        """Task 3: get_callers must return truncated=False when the result
        count is within the limit. We don't know the exact call graph of
        the fixture in advance, but we can assert the flag propagation
        on any queryable method."""
        repository = seeded_postgres_contract_repository
        results = await pg_read_backend.find_symbols(
            "", repository=repository, exact=False, limit=50,
        )
        methods = [r for r in results if r.entity_type == "method"]
        if not methods:
            pytest.skip("No methods in fixture repo to exercise call graph")
        method = methods[0]
        callers = await pg_read_backend.get_callers(
            method.name, repository=repository, limit=1000,
        )
        for caller in callers:
            assert caller.truncated is False, \
                f"Expected truncated=False under-limit; got True on {caller.name}"

    async def test_task1_find_symbols_exact_match_returns_code(
        self, pg_read_backend, seeded_postgres_contract_repository,
    ):
        """Task 1: find_symbols returns raw code (which search_code's
        _apply_code_mode then transforms). This contract test proves the
        fixture has non-null code bodies — the mocked unit tests in
        tests/backends/test_postgres_read.py verify the actual
        transformation logic."""
        repository = seeded_postgres_contract_repository
        results = await pg_read_backend.find_symbols(
            "Service", repository=repository, exact=True,
        )
        assert any(r.name == "Service" for r in results), \
            f"Expected 'Service' class in fixture; got: {[r.name for r in results]}"
        service = next(r for r in results if r.name == "Service")
        assert service.code is not None, \
            "Expected non-null code body on Service class"
