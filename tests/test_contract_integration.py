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
        assert repo_context.total_files == 2
        assert "java" in repo_context.languages
        assert "javascript" in repo_context.languages
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
