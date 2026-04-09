"""Live contract tests against a real Neo4j graph seeded from Constellation output."""

import pytest


pytestmark = pytest.mark.integration


class TestConstellationContract:
    async def test_repository_and_package_context(self, live_graph_client, seeded_contract_repository):
        repository = seeded_contract_repository

        repo_context = await live_graph_client.get_repository_context(repository)
        package_context = await live_graph_client.get_package_context(
            "com.example",
            repository=repository,
        )

        assert repo_context is not None
        assert repo_context.name == repository
        # Fixture repo has 4 files: Service.java, App.tsx, AuthService.cs, demo.py
        assert repo_context.total_files == 4
        assert "java" in repo_context.languages
        assert "javascript" in repo_context.languages
        assert "csharp" in repo_context.languages
        assert package_context is not None
        assert package_context.name == "com.example"
        assert package_context.repository == repository
        assert "Service" in package_context.classes
        assert any(path.endswith("Service.java") for path in package_context.files)

    async def test_file_context_and_hook_usage(self, live_graph_client, seeded_contract_repository):
        repository = seeded_contract_repository

        file_context = await live_graph_client.get_file_context(
            "App.tsx",
            repository=repository,
        )
        hook_usage = await live_graph_client.get_hook_usage(
            "useState",
            repository=repository,
        )

        assert file_context is not None
        assert file_context.name == "App.tsx"
        assert sorted(export.name for export in file_context.exports) == ["App", "renderApp"]
        assert file_context.hooks == ["useState"]
        assert sorted(file_context.top_level_methods) == ["App", "renderApp"]
        assert len(hook_usage) == 1
        assert hook_usage[0].name == "App"
        assert hook_usage[0].relationship_type == "USES_HOOK"

    async def test_java_file_context_includes_class_scoped_members(
        self,
        live_graph_client,
        seeded_contract_repository,
    ):
        repository = seeded_contract_repository

        file_context = await live_graph_client.get_file_context(
            "Service.java",
            repository=repository,
        )

        assert file_context is not None
        assert file_context.name == "Service.java"
        assert file_context.packages == ["com.example"]
        assert file_context.classes == ["Service"]
        assert file_context.constructors == ["Service"]
        assert file_context.fields == ["client"]
        assert file_context.references == ["client.fetch"]

    async def test_call_graph_queries_match_java_parser_output(
        self,
        live_graph_client,
        seeded_contract_repository,
    ):
        repository = seeded_contract_repository

        callers = await live_graph_client.get_callers(
            "helper",
            repository=repository,
            file_path="Service.java",
        )
        callees = await live_graph_client.get_callees(
            "run",
            repository=repository,
            file_path="Service.java",
        )
        symbols = await live_graph_client.find_symbols(
            "helper",
            repository=repository,
            exact=True,
        )

        assert [caller.name for caller in callers] == ["run"]
        assert [(callee.name, callee.entity_type) for callee in callees] == [
            ("helper", "method"),
            ("client.fetch", "reference"),
        ]
        assert any(symbol.entity_type == "method" for symbol in symbols)

    async def test_seeded_fixture_uses_relative_file_paths(
        self,
        live_graph_client,
        seeded_contract_repository,
    ):
        """Regression guard for contract-fixture normalization.

        Mirrors test_seeded_postgres_fixture_uses_relative_file_paths in
        the Postgres suite. Both fixtures should run Constellation's
        production normalization step before seeding, so the contract
        tests exercise the production graph shape instead of the raw
        parser output.
        """
        repository = seeded_contract_repository

        cypher = """
            MATCH (n {repository: $repository})
            WHERE n.file_path IS NOT NULL AND n.file_path STARTS WITH '/'
            RETURN n.id AS id, n.file_path AS file_path, labels(n) AS labels
            LIMIT 10
        """
        rows = await live_graph_client._query(cypher, repository=repository)

        assert rows == [], (
            f"Expected all file_paths to be relative (normalized), but found "
            f"{len(rows)} rows with absolute paths. Examples: {rows}"
        )

    async def test_nested_dotnet_namespace_full_name_across_queries(
        self,
        live_graph_client,
        seeded_contract_repository,
    ):
        """End-to-end parity for nested .NET namespaces.

        Constellation's .NET parser stores nested namespaces with the
        full dotted path in the entity id but only the leaf segment in
        name (dotnet.py:141). Telescope MUST reconstruct the full dotted
        name consistently across every query that returns package
        information:

        1. get_package_context (Neo4j native, matches Plan C expected behavior)
        2. get_file_context.packages (fixed in Plan C Task 3)
        3. get_codebase_overview.packages (fixed in Plan C Task 4)

        A regression in any of the three locations would surface here as
        the leaf 'Services' leaking into a result that should show
        'Company.Product.Services'. This test validates the end-to-end
        Constellation → Neo4j write → Telescope read path against a
        single real nested-namespace fixture file.
        """
        repository = seeded_contract_repository

        # (1) get_package_context by the full dotted name — the caller
        # provides the full path; Neo4j resolves via `pkg.id ENDS WITH`.
        pkg_ctx = await live_graph_client.get_package_context(
            "Company.Product.Services", repository=repository,
        )
        assert pkg_ctx is not None, \
            "Expected to resolve Company.Product.Services by full name"
        assert pkg_ctx.name == "Company.Product.Services", \
            f"get_package_context must return full dotted name; got: {pkg_ctx.name}"

        # (2) get_file_context.packages must show the full name, not leaf
        fc = await live_graph_client.get_file_context(
            "AuthService.cs", repository=repository,
        )
        assert fc is not None
        assert "Company.Product.Services" in fc.packages, \
            f"file_context.packages must show the full dotted name; got: {fc.packages}"
        assert "Services" not in fc.packages, \
            f"Leaf-only 'Services' must not appear in file_context.packages; got: {fc.packages}"

        # (3) get_codebase_overview.packages must also show the full name
        overview = await live_graph_client.get_codebase_overview(
            repository=repository, include_packages=True,
        )
        assert "Company.Product.Services" in overview.packages, \
            f"codebase_overview.packages must show the full dotted name; got: {overview.packages}"
        assert "Services" not in overview.packages, \
            f"Leaf-only 'Services' must not appear in codebase_overview.packages; got: {overview.packages}"

    async def test_python_cross_file_calls_surface_as_references(
        self,
        live_graph_client,
        seeded_contract_repository,
    ):
        """Symmetric to the Postgres contract assertion. Before Sub-plan A,
        cross-file Python CALLS created orphan Neo4j nodes with no label
        match; now they resolve to Reference nodes via the Java-parity
        pattern ported in constellation/parsers/python_parser.py."""
        repository = seeded_contract_repository
        callees = await live_graph_client.get_callees(
            "run",
            repository=repository,
            file_path="demo.py",
        )
        reference_callees = [c for c in callees if c.entity_type == "reference"]
        reference_names = {c.name for c in reference_callees}
        assert reference_names == {"load_config", "redis.from_url", "client.get"}, (
            f"Expected all three unresolved calls as Reference callees; got: "
            f"{[(c.name, c.entity_type) for c in callees]}"
        )
