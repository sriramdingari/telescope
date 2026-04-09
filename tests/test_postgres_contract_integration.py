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
        # Fixture repo has 3 files: Service.java, App.tsx, AuthService.cs
        assert repo_context.total_files == 3
        assert "java" in repo_context.languages
        assert "javascript" in repo_context.languages
        assert "csharp" in repo_context.languages

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
        """Java file-context parity with Neo4j.

        Mirrors the Neo4j contract at test_contract_integration.py:51-69
        which asserts the full Service.java shape: package, class,
        constructor, field, and unresolved call reference. Regressions
        in Plan B Task 1's shared-file query rewrite should surface here
        as missing constructors / fields / references. A regression in
        Plan A's fixture normalization (stale absolute paths in storage)
        would surface as either failed resolution or empty members.
        """
        repository = seeded_postgres_contract_repository
        # Pass the relative-suffix the user would naturally type. Constellation
        # stores absolute paths (e.g. /abs/.../src/java/com/example/Service.java),
        # so the resolver must do a suffix match — mirrors Neo4j's
        # `f.file_path ENDS WITH $file_path` (neo4j.py:484) and the Neo4j
        # contract test at test_contract_integration.py:58 which passes
        # "Service.java".
        fc = await pg_read_backend.get_file_context(
            "Service.java", repository=repository,
        )
        assert fc is not None
        assert fc.name == "Service.java"
        assert fc.packages == ["com.example"]
        assert fc.classes == ["Service"]
        assert fc.constructors == ["Service"], \
            f"Expected Service constructor; got: {fc.constructors}"
        assert fc.fields == ["client"], \
            f"Expected client field; got: {fc.fields}"
        assert fc.references == ["client.fetch"], \
            f"Expected client.fetch reference; got: {fc.references}"


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
        transformation logic.

        We query for the JavaScript top-level function `App` in App.tsx
        instead of a Java class because Constellation's Java parser
        intentionally does NOT populate `code` for Class/Method/Constructor/
        Field entities (see constellation/parsers/java.py:189-201, 508-520
        — no `code=` kwarg). The JavaScript parser DOES populate `code`
        for top-level functions (constellation/parsers/javascript.py:274,
        300, 364, 415, 487). The contract that find_symbols faithfully
        propagates parser-stored `code` is identical regardless of which
        language we exercise it through; testing it through JavaScript
        avoids a Java parser limitation that would otherwise look like
        a Postgres backend bug.

        Triage note (Plan A Task 3): the previous version of this test
        asserted `service.code is not None` for the Java `Service` class
        and was failing because Java entities never have code. The
        normalization fix in Plan A Task 1 changed the failure mode but
        not the root cause — the assertion itself was simply wrong for
        Java. This rewrite asks for a parser/language combination where
        Constellation actually populates code, which is the correct
        target for this contract assertion.

        Plan B Task 3: find_symbols now defaults to code_mode='none' to
        save tokens on identifier lookups. This contract test still
        wants to verify the fixture populates raw code bodies, so it
        explicitly requests code_mode='full' to bypass the default."""
        repository = seeded_postgres_contract_repository
        results = await pg_read_backend.find_symbols(
            "App", repository=repository, exact=True, code_mode="full",
        )
        # The fixture's App.tsx has multiple symbols matching "App"
        # (App component, App function, etc.). Pick whichever one has
        # a non-null `code` body — that's the function-level entity
        # the JS parser populates.
        with_code = [r for r in results if r.code is not None]
        assert with_code, (
            f"Expected at least one 'App' entity with non-null code in "
            f"fixture; got: {[(r.name, r.entity_type, r.code is not None) for r in results]}"
        )
        # Pick the first one with code and verify its body is non-empty
        # and contains some recognizable JS/TSX content.
        app = with_code[0]
        assert app.code, f"Expected non-empty code on {app.name}; got: {app.code!r}"
        assert len(app.code) > 0

    async def test_nested_dotnet_namespace_full_name_across_queries(
        self, pg_read_backend, seeded_postgres_contract_repository,
    ):
        """End-to-end parity for nested .NET namespaces.

        Constellation's .NET parser stores nested namespaces with the
        full dotted path in the entity id but only the leaf segment in
        symbol_name (dotnet.py:141). Telescope MUST reconstruct the full
        dotted name consistently across every query that returns package
        information:

        1. get_package_context (fixed in Task 2, 2025-round)
        2. get_file_context.packages (fixed in Plan C Task 1)
        3. get_codebase_overview.packages (fixed in Plan C Task 2)

        A regression in any of the three locations would surface here as
        the leaf 'Services' leaking into a result that should show
        'Company.Product.Services'. This test validates the end-to-end
        Constellation → Postgres write → Telescope read path against a
        single real nested-namespace fixture file (AuthService.cs).
        Mirrors the Neo4j test at test_contract_integration.py.
        """
        repository = seeded_postgres_contract_repository

        # (1) get_package_context by the full dotted name — the caller
        # provides the full path; Postgres resolves via the id-suffix
        # match + LIMIT 2 ambiguity check added in Task 2.
        pkg_ctx = await pg_read_backend.get_package_context(
            "Company.Product.Services", repository=repository,
        )
        assert pkg_ctx is not None, \
            "Expected to resolve Company.Product.Services by full name"
        assert pkg_ctx.name == "Company.Product.Services", \
            f"get_package_context must return full dotted name; got: {pkg_ctx.name}"

        # (2) get_file_context.packages must show the full name, not leaf
        fc = await pg_read_backend.get_file_context(
            "AuthService.cs", repository=repository,
        )
        assert fc is not None
        assert "Company.Product.Services" in fc.packages, \
            f"file_context.packages must show the full dotted name; got: {fc.packages}"
        assert "Services" not in fc.packages, \
            f"Leaf-only 'Services' must not appear in file_context.packages; got: {fc.packages}"

        # (3) get_codebase_overview.packages must also show the full name
        overview = await pg_read_backend.get_codebase_overview(
            repository=repository, include_packages=True,
        )
        assert "Company.Product.Services" in overview.packages, \
            f"codebase_overview.packages must show the full dotted name; got: {overview.packages}"
        assert "Services" not in overview.packages, \
            f"Leaf-only 'Services' must not appear in codebase_overview.packages; got: {overview.packages}"
