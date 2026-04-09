"""Tests for the Neo4jReadBackend."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telescope.config import Config
from telescope.models import (
    CallGraphNode,
    ClassHierarchy,
    CodebaseOverview,
    CodeEntity,
    FileContext,
    FunctionContext,
    ImpactResult,
    PackageContext,
    RepositoryContext,
)


@pytest.fixture()
def test_config():
    return Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="constellation",
        openai_api_key="sk-test-key",
    )


@pytest.fixture()
def patched_config(test_config):
    """Patch get_config in backends.neo4j to return test_config."""
    with patch("telescope.backends.neo4j.get_config", return_value=test_config):
        yield test_config


class TestNeo4jReadBackendConnect:
    """Tests for Neo4jReadBackend.connect()."""

    async def test_connect_creates_driver(self, patched_config):
        """Verify AsyncGraphDatabase.driver is called with correct URI and auth."""
        with patch("telescope.backends.neo4j.AsyncGraphDatabase") as mock_gdb, \
             patch("telescope.backends.neo4j.AsyncOpenAI"):
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_gdb.driver = MagicMock(return_value=mock_driver)
            from telescope.backends.neo4j import Neo4jReadBackend
            client = Neo4jReadBackend()
            await client.connect()
            mock_gdb.driver.assert_called_once_with(
                "bolt://localhost:7687",
                auth=("neo4j", "constellation"),
            )

    async def test_connect_verifies_connectivity(self, patched_config, mock_neo4j_driver):
        """Startup should fail fast if Neo4j connectivity is broken."""
        with patch("telescope.backends.neo4j.AsyncGraphDatabase") as mock_gdb, \
             patch("telescope.backends.neo4j.AsyncOpenAI"):
            mock_gdb.driver = MagicMock(return_value=mock_neo4j_driver)
            from telescope.backends.neo4j import Neo4jReadBackend
            client = Neo4jReadBackend()
            await client.connect()
            mock_neo4j_driver.verify_connectivity.assert_awaited_once()

    async def test_connect_creates_openai_client(self, patched_config):
        """Verify AsyncOpenAI is called with api_key."""
        with patch("telescope.backends.neo4j.AsyncGraphDatabase") as mock_gdb, \
             patch("telescope.backends.neo4j.AsyncOpenAI") as mock_openai:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_gdb.driver = MagicMock(return_value=mock_driver)
            from telescope.backends.neo4j import Neo4jReadBackend
            client = Neo4jReadBackend()
            await client.connect()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "sk-test-key"

    async def test_connect_passes_base_url_when_set(self, test_config):
        """When openai_base_url is set, AsyncOpenAI is called with base_url."""
        test_config.openai_base_url = "https://my-custom-openai.example.com"
        with patch("telescope.backends.neo4j.get_config", return_value=test_config), \
             patch("telescope.backends.neo4j.AsyncGraphDatabase") as mock_gdb, \
             patch("telescope.backends.neo4j.AsyncOpenAI") as mock_openai:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_gdb.driver = MagicMock(return_value=mock_driver)
            from telescope.backends.neo4j import Neo4jReadBackend
            client = Neo4jReadBackend()
            await client.connect()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs.get("base_url") == "https://my-custom-openai.example.com"

    async def test_connect_omits_base_url_when_none(self, patched_config):
        """When openai_base_url is None, AsyncOpenAI is NOT called with base_url."""
        assert patched_config.openai_base_url is None
        with patch("telescope.backends.neo4j.AsyncGraphDatabase") as mock_gdb, \
             patch("telescope.backends.neo4j.AsyncOpenAI") as mock_openai:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_gdb.driver = MagicMock(return_value=mock_driver)
            from telescope.backends.neo4j import Neo4jReadBackend
            client = Neo4jReadBackend()
            await client.connect()
            call_kwargs = mock_openai.call_args[1]
            assert "base_url" not in call_kwargs


class TestNeo4jReadBackendClose:
    """Tests for Neo4jReadBackend.close()."""

    async def test_close_calls_driver_close(self, patched_config, mock_neo4j_driver):
        """Verify driver.close() is called when driver exists."""
        from telescope.backends.neo4j import Neo4jReadBackend
        client = Neo4jReadBackend()
        client._driver = mock_neo4j_driver
        await client.close()
        mock_neo4j_driver.close.assert_called_once()

    async def test_close_calls_openai_close(self, patched_config, mock_openai_client):
        """Verify the OpenAI client is closed when present."""
        from telescope.backends.neo4j import Neo4jReadBackend
        client = Neo4jReadBackend()
        client._openai = mock_openai_client
        await client.close()
        mock_openai_client.close.assert_awaited_once()

    async def test_close_safe_when_not_connected(self, patched_config):
        """No error when _driver is None (never connected)."""
        from telescope.backends.neo4j import Neo4jReadBackend
        client = Neo4jReadBackend()
        assert client._driver is None
        # Should not raise
        await client.close()


class TestNeo4jReadBackendQuery:
    """Tests for Neo4jReadBackend._query()."""

    async def test_query_runs_cypher_and_returns_dicts(
        self, patched_config, mock_neo4j_driver, mock_neo4j_result
    ):
        """Verify _query executes the cypher and returns list of dicts."""
        expected_data = [{"name": "foo"}, {"name": "bar"}]
        mock_neo4j_driver.session.return_value.run = AsyncMock(
            return_value=mock_neo4j_result(expected_data)
        )
        from telescope.backends.neo4j import Neo4jReadBackend
        client = Neo4jReadBackend()
        client._driver = mock_neo4j_driver
        result = await client._query("MATCH (n) RETURN n")
        assert result == expected_data

    async def test_query_reads_full_result_set(
        self, patched_config, mock_neo4j_driver, mock_neo4j_result
    ):
        """_query should consume all rows, not just an arbitrary page."""
        mock_session = mock_neo4j_driver.session.return_value
        mock_result = mock_neo4j_result([{"name": "foo"}])
        mock_session.run = AsyncMock(return_value=mock_result)
        from telescope.backends.neo4j import Neo4jReadBackend
        client = Neo4jReadBackend()
        client._driver = mock_neo4j_driver
        await client._query("MATCH (n) RETURN n")
        mock_result.data.assert_awaited_once()
        mock_result.fetch.assert_not_called()

    async def test_query_passes_parameters(
        self, patched_config, mock_neo4j_driver, mock_neo4j_result
    ):
        """Verify params are passed to session.run()."""
        mock_session = mock_neo4j_driver.session.return_value
        mock_session.run = AsyncMock(return_value=mock_neo4j_result([]))
        from telescope.backends.neo4j import Neo4jReadBackend
        client = Neo4jReadBackend()
        client._driver = mock_neo4j_driver
        await client._query("MATCH (n {name: $name}) RETURN n", name="MyClass")
        mock_session.run.assert_called_once_with(
            "MATCH (n {name: $name}) RETURN n", name="MyClass"
        )


class TestNeo4jReadBackendGetEmbedding:
    """Tests for Neo4jReadBackend._get_embedding()."""

    async def test_get_embedding_calls_openai(
        self, patched_config, mock_openai_client
    ):
        """Verify embeddings.create() is called with correct model, input, dimensions."""
        from telescope.backends.neo4j import Neo4jReadBackend
        client = Neo4jReadBackend()
        client._openai = mock_openai_client
        await client._get_embedding("search query text")
        mock_openai_client.embeddings.create.assert_called_once_with(
            model=patched_config.embedding_model,
            input="search query text",
            dimensions=patched_config.embedding_dimensions,
        )

    async def test_get_embedding_returns_vector(
        self, patched_config, mock_openai_client, mock_openai_response
    ):
        """Verify _get_embedding returns the embedding list from the response."""
        expected_vector = [0.5] * 1536
        mock_openai_client.embeddings.create = AsyncMock(
            return_value=mock_openai_response(embedding=expected_vector)
        )
        from telescope.backends.neo4j import Neo4jReadBackend
        client = Neo4jReadBackend()
        client._openai = mock_openai_client
        result = await client._get_embedding("some text")
        assert result == expected_vector


def _make_search_result(
    name="foo",
    file_path="src/Foo.java",
    repository="my-repo",
    line_number=10,
    line_end=20,
    code="def foo():\n    pass",
    signature="def foo()",
    entity_type="method",
    score=0.95,
    entity_id=None,
    language=None,
    return_type=None,
    modifiers=None,
    stereotypes=None,
    content_hash=None,
    properties=None,
):
    """Helper to build a mock Neo4j result dict for search_code."""
    return {
        "id": entity_id,
        "name": name,
        "file_path": file_path,
        "repository": repository,
        "line_number": line_number,
        "line_end": line_end,
        "code": code,
        "signature": signature,
        "entity_type": entity_type,
        "score": score,
        "language": language,
        "return_type": return_type,
        "modifiers": modifiers if modifiers is not None else [],
        "stereotypes": stereotypes if stereotypes is not None else [],
        "content_hash": content_hash,
        "properties": properties if properties is not None else {},
    }


@pytest.fixture()
def graph_client(test_config):
    """Create a Neo4jReadBackend with mocked internals for search_code tests."""
    with patch("telescope.backends.neo4j.get_config", return_value=test_config):
        from telescope.backends.neo4j import Neo4jReadBackend
        client = Neo4jReadBackend()
    client._driver = AsyncMock()
    client._openai = AsyncMock()
    return client


class TestSearchCode:
    """Tests for Neo4jReadBackend.search_code()."""

    async def test_search_code_calls_get_embedding(self, graph_client):
        """Verify _get_embedding called with query text."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.search_code("auth logic")
        graph_client._get_embedding.assert_called_once_with("auth logic")

    async def test_search_code_queries_all_indexes_when_no_entity_type(self, graph_client):
        """When entity_type=None, verify _query called 4 times (once per index)."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.search_code("auth logic")
        assert graph_client._query.call_count == 4

    async def test_search_code_queries_single_index_for_entity_type(self, graph_client):
        """When entity_type='method', verify _query called once with vector_method_embedding."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.search_code("auth logic", entity_type="method")
        assert graph_client._query.call_count == 1
        cypher_arg = graph_client._query.call_args[0][0]
        assert "vector_method_embedding" in cypher_arg

    async def test_search_code_queries_constructor_index(self, graph_client):
        """When entity_type='constructor', verify _query called with vector_constructor_embedding."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.search_code("init", entity_type="constructor")
        assert graph_client._query.call_count == 1
        cypher_arg = graph_client._query.call_args[0][0]
        assert "vector_constructor_embedding" in cypher_arg

    async def test_search_code_applies_repository_filter(self, graph_client):
        """Verify Cypher contains n.repository = $repository."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.search_code("auth", entity_type="method", repository="my-repo")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "n.repository = $repository" in cypher_arg

    async def test_search_code_applies_file_pattern_filter(self, graph_client):
        """Verify wildcard file filters are parameterized as regex."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.search_code("auth", entity_type="method", file_pattern="Service.java")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "n.file_path =~ $file_regex" in cypher_arg
        assert graph_client._query.call_args.kwargs["file_regex"] == "^.*Service\\.java.*$"

    async def test_search_code_applies_language_and_stereotype_filters(self, graph_client):
        """Language and stereotype filters should be pushed into vector search."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(return_value=[])

        await graph_client.search_code(
            "auth",
            entity_type="method",
            language="Python",
            stereotype="endpoint",
        )

        cypher_arg = graph_client._query.call_args[0][0]
        kwargs = graph_client._query.call_args.kwargs
        assert "n.language = $language" in cypher_arg
        assert "$stereotype IN coalesce(n.stereotypes, [])" in cypher_arg
        assert kwargs["language"] == "Python"
        assert kwargs["stereotype"] == "endpoint"

    async def test_search_code_maps_line_number_to_line_start(self, graph_client):
        """Verify line_number from Neo4j result maps to line_start on CodeEntity."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        mock_result = _make_search_result(line_number=42, line_end=50)
        graph_client._query = AsyncMock(return_value=[mock_result])
        results = await graph_client.search_code("auth", entity_type="method")
        assert len(results) == 1
        assert results[0].line_start == 42
        assert results[0].line_end == 50

    async def test_search_code_maps_richer_entity_metadata(self, graph_client):
        """Semantic search should preserve Constellation metadata fields."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(return_value=[
            _make_search_result(
                entity_id="repo::module.foo",
                language="Python",
                return_type="None",
                modifiers=["async"],
                stereotypes=["endpoint"],
                content_hash="deadbeef",
                properties={"visibility": "public"},
            )
        ])

        results = await graph_client.search_code("auth", entity_type="method")

        assert results[0].entity_id == "repo::module.foo"
        assert results[0].language == "Python"
        assert results[0].return_type == "None"
        assert results[0].modifiers == ["async"]
        assert results[0].stereotypes == ["endpoint"]
        assert results[0].content_hash == "deadbeef"
        assert results[0].properties == {"visibility": "public"}

    async def test_search_code_code_mode_none(self, graph_client):
        """Verify code=None when code_mode='none'."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        mock_result = _make_search_result(code="def foo():\n    pass", signature="def foo()")
        graph_client._query = AsyncMock(return_value=[mock_result])
        results = await graph_client.search_code("auth", entity_type="method", code_mode="none")
        assert results[0].code is None

    async def test_search_code_code_mode_signature(self, graph_client):
        """Verify code=signature when code_mode='signature'."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        mock_result = _make_search_result(code="def foo():\n    pass", signature="def foo()")
        graph_client._query = AsyncMock(return_value=[mock_result])
        results = await graph_client.search_code("auth", entity_type="method", code_mode="signature")
        assert results[0].code == "def foo()"

    async def test_search_code_code_mode_preview_truncates(self, graph_client):
        """Verify long code truncated to 10 lines with '... (truncated)'."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        long_code = "\n".join([f"line {i}" for i in range(20)])
        mock_result = _make_search_result(code=long_code, signature="def foo()")
        graph_client._query = AsyncMock(return_value=[mock_result])
        results = await graph_client.search_code("auth", entity_type="method", code_mode="preview")
        code = results[0].code
        lines = code.split("\n")
        assert len(lines) == 11  # 10 lines + "... (truncated)"
        assert lines[-1] == "... (truncated)"
        assert lines[0] == "line 0"
        assert lines[9] == "line 9"

    async def test_search_code_code_mode_full(self, graph_client):
        """Verify full code returned when code_mode='full'."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        long_code = "\n".join([f"line {i}" for i in range(20)])
        mock_result = _make_search_result(code=long_code, signature="def foo()")
        graph_client._query = AsyncMock(return_value=[mock_result])
        results = await graph_client.search_code("auth", entity_type="method", code_mode="full")
        assert results[0].code == long_code

    async def test_search_code_results_sorted_by_score(self, graph_client):
        """Verify results sorted descending by score."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        results_batch1 = [
            _make_search_result(name="low", score=0.5),
            _make_search_result(name="high", score=0.99),
        ]
        results_batch2 = [
            _make_search_result(name="mid", score=0.75),
        ]
        graph_client._query = AsyncMock(side_effect=[
            results_batch1, results_batch2, [], [],
        ])
        results = await graph_client.search_code("auth")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].name == "high"
        assert results[1].name == "mid"
        assert results[2].name == "low"

    async def test_search_code_results_limited(self, graph_client):
        """Verify results capped at limit."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        many_results = [_make_search_result(name=f"r{i}", score=1.0 - i * 0.01) for i in range(10)]
        graph_client._query = AsyncMock(side_effect=[
            many_results, many_results, many_results, many_results,
        ])
        results = await graph_client.search_code("auth", limit=5)
        assert len(results) == 5

    async def test_search_code_uses_hybrid_symbol_results_for_symbol_queries(self, graph_client):
        """Identifier-like queries should merge exact symbol hits ahead of semantic hits."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(side_effect=[[ _make_search_result(name="semantic", entity_id="repo::semantic") ]])
        graph_client.find_symbols = AsyncMock(side_effect=[
            [
                CodeEntity(
                    name="useState",
                    file_path="src/App.tsx",
                    repository="repo",
                    entity_id="repo::src/App.tsx::useState",
                    entity_type="hook",
                )
            ],
        ])

        results = await graph_client.search_code("useState", entity_type="method", limit=5)

        graph_client.find_symbols.assert_awaited_once_with(
            "useState",
            entity_types=["method"],
            repository=None,
            file_pattern=None,
            limit=5,
            exact=True,
            language=None,
            stereotype=None,
            code_mode="preview",
        )
        assert [result.name for result in results] == ["useState", "semantic"]

    async def test_search_code_skips_hybrid_lookup_for_natural_language_queries(self, graph_client):
        """Natural-language queries should remain semantic-only."""
        graph_client._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        graph_client._query = AsyncMock(side_effect=[[]] * 4)
        graph_client.find_symbols = AsyncMock(return_value=[])

        await graph_client.search_code("payment processing")

        graph_client.find_symbols.assert_not_awaited()


class TestGetCallers:
    """Tests for Neo4jReadBackend.get_callers()."""

    async def test_get_callers_basic(self, graph_client):
        """Returns list of CallGraphNode from mock results."""
        graph_client._query = AsyncMock(return_value=[
            {"name": "caller1", "file_path": "src/main.py", "repository": "my-repo", "signature": "def caller1()", "line_number": 10},
            {"name": "caller2", "file_path": "src/utils.py", "repository": "my-repo", "signature": "def caller2()", "line_number": 20},
        ])
        results = await graph_client.get_callers("targetMethod")
        assert len(results) == 2
        assert all(isinstance(r, CallGraphNode) for r in results)
        assert results[0].name == "caller1"
        assert results[0].file_path == "src/main.py"
        assert results[0].repository == "my-repo"
        assert results[0].signature == "def caller1()"
        assert results[1].name == "caller2"

    async def test_get_callers_matches_method_or_constructor(self, graph_client):
        """Verify Cypher contains (m:Method OR m:Constructor)."""
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.get_callers("someMethod")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "(m:Method OR m:Constructor)" in cypher_arg

    async def test_get_callers_traverses_overrides(self, graph_client):
        """Verify caller traversal uses actual Constellation type relationships."""
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.get_callers("someMethod")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "IMPLEMENTS" in cypher_arg
        assert "EXTENDS" in cypher_arg
        assert "OVERRIDES" not in cypher_arg

    async def test_get_callers_uses_parameter_suffix_when_entity_id_is_java_style(self, graph_client):
        """Resolved Java methods should constrain family expansion by parameter signature."""
        graph_client._query = AsyncMock(return_value=[])

        await graph_client.get_callers(
            "save",
            entity_id="repo::com.example.Service.save(String,int)",
        )

        cypher_arg = graph_client._query.call_args[0][0]
        kwargs = graph_client._query.call_args.kwargs
        assert "candidate_method.id ENDS WITH (candidate_method.name + $parameter_suffix)" in cypher_arg
        assert kwargs["parameter_suffix"] == "(String,int)"

    async def test_get_callers_preserves_top_level_functions(self, graph_client):
        """Top-level functions should stay queryable even without a class owner."""
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.get_callers("standalone_function")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "WHEN owner IS NULL THEN [NULL]" in cypher_arg

    async def test_get_callers_depth_clamped_to_3(self, graph_client):
        """Passing depth=5 results in CALLS*1..3 in Cypher."""
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.get_callers("someMethod", depth=5)
        cypher_arg = graph_client._query.call_args[0][0]
        assert "CALLS*1..3" in cypher_arg

    async def test_get_callers_applies_file_filter(self, graph_client):
        """Verify file_path is passed as a parameterized suffix filter."""
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.get_callers("someMethod", file_path="Service.java")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "ENDS WITH" in cypher_arg
        assert graph_client._query.call_args.kwargs["file_path"] == "Service.java"

    async def test_get_callers_applies_repo_filter(self, graph_client):
        """Verify Cypher contains repository = $repository when repository provided."""
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.get_callers("someMethod", repository="my-repo")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "repository = $repository" in cypher_arg

    async def test_get_callers_maps_line_number_to_line_start(self, graph_client):
        """Verify line_number from Neo4j result maps to line_start on CallGraphNode."""
        graph_client._query = AsyncMock(return_value=[
            {"name": "caller1", "file_path": "src/main.py", "repository": "my-repo", "signature": "def caller1()", "line_number": 42},
        ])
        results = await graph_client.get_callers("targetMethod")
        assert len(results) == 1
        assert results[0].line_start == 42

    async def test_get_callers_maps_query_depth(self, graph_client):
        """Caller traversal depth should be preserved from the query result."""
        graph_client._query = AsyncMock(return_value=[
            {
                "name": "caller1",
                "file_path": "src/main.py",
                "repository": "my-repo",
                "signature": "def caller1()",
                "line_number": 42,
                "depth": 2,
            },
        ])

        results = await graph_client.get_callers("targetMethod")

        assert len(results) == 1
        assert results[0].depth == 2

    async def test_get_callers_empty_results(self, graph_client):
        """Returns empty list when no callers found."""
        graph_client._query = AsyncMock(return_value=[])
        results = await graph_client.get_callers("unusedMethod")
        assert results == []

    async def test_get_callers_uses_limit_and_marks_truncation(self, graph_client):
        """Caller traversal should honor caller limits and mark truncated results."""
        graph_client._query = AsyncMock(return_value=[
            {"name": "caller1", "file_path": "src/a.py", "line_number": 1},
            {"name": "caller2", "file_path": "src/b.py", "line_number": 2},
            {"name": "caller3", "file_path": "src/c.py", "line_number": 3},
        ])

        results = await graph_client.get_callers("targetMethod", limit=2)

        kwargs = graph_client._query.call_args.kwargs
        assert kwargs["query_limit"] == 3
        assert [result.name for result in results] == ["caller1", "caller2"]
        assert all(result.truncated is True for result in results)

    async def test_get_callers_returns_entity_id_for_chaining(self, graph_client):
        """CallGraphNode must carry entity_id so agents can chain
        get_callers → get_callers(entity_id=...) without re-resolving
        by name. Symmetric to the Postgres fix."""
        graph_client._query = AsyncMock(return_value=[
            {
                "entity_id": "repo::Caller.call_method",
                "name": "call_method",
                "file_path": "caller.py",
                "repository": "repo",
                "signature": "def call_method(self)",
                "line_number": 5,
                "entity_type": "Method",
                "relationship_type": "CALLS",
                "depth": 1,
            },
        ])

        results = await graph_client.get_callers("method")

        assert len(results) == 1
        assert results[0].entity_id == "repo::Caller.call_method", (
            f"CallGraphNode.entity_id must be populated; got: {results[0].entity_id}"
        )
        # The Cypher RETURN must expose caller.id AS entity_id
        cypher_arg = graph_client._query.call_args[0][0]
        assert "caller.id AS entity_id" in cypher_arg, (
            f"get_callers Cypher RETURN must expose caller.id AS entity_id; got: {cypher_arg}"
        )


class TestGetCallees:
    """Tests for Neo4jReadBackend.get_callees()."""

    async def test_get_callees_basic(self, graph_client):
        """Returns list of CallGraphNode from mock results."""
        graph_client._query = AsyncMock(side_effect=[
            [
                {
                    "entity_id": "my-repo::com.example.Service.callee1",
                    "name": "callee1",
                    "file_path": "src/service.py",
                    "repository": "my-repo",
                    "signature": "def callee1()",
                    "line_number": 15,
                    "entity_type": "Method",
                    "relationship_type": "CALLS",
                    "depth": 1,
                },
                {
                    "entity_id": "my-repo::requests.get",
                    "name": "requests.get",
                    "file_path": "src/dao.py",
                    "repository": "my-repo",
                    "line_number": 30,
                    "entity_type": "Reference",
                    "relationship_type": "CALLS",
                    "depth": 1,
                },
            ],
            [],
        ])
        results = await graph_client.get_callees("sourceMethod")
        assert len(results) == 2
        assert all(isinstance(r, CallGraphNode) for r in results)
        by_name = {result.name: result for result in results}
        assert by_name["callee1"].file_path == "src/service.py"
        assert by_name["callee1"].repository == "my-repo"
        assert by_name["callee1"].signature == "def callee1()"
        assert by_name["callee1"].entity_type == "method"
        assert by_name["callee1"].entity_id == "my-repo::com.example.Service.callee1"
        assert by_name["requests.get"].entity_type == "reference"
        assert by_name["requests.get"].relationship_type == "CALLS"
        assert by_name["requests.get"].entity_id == "my-repo::requests.get"

    async def test_get_callees_matches_method_or_constructor(self, graph_client):
        """Verify Cypher contains (m:Method OR m:Constructor)."""
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.get_callees("someMethod")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "(m:Method OR m:Constructor)" in cypher_arg

    async def test_get_callees_traverses_overrides(self, graph_client):
        """Verify callee traversal uses actual Constellation type relationships."""
        graph_client._query = AsyncMock(side_effect=[[], []])
        await graph_client.get_callees("someMethod")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "IMPLEMENTS" in cypher_arg
        assert "EXTENDS" in cypher_arg
        assert "OVERRIDES" not in cypher_arg

    async def test_get_callees_avoids_hard_coded_implements_patterns(self, graph_client):
        """Method-family traversal should avoid literal absent-type patterns that emit warnings."""
        graph_client._query = AsyncMock(side_effect=[[], []])

        await graph_client.get_callees("someMethod")

        cypher_arg = graph_client._query.call_args_list[0][0][0]
        assert "type(rel) IN ['IMPLEMENTS', 'EXTENDS']" in cypher_arg
        assert "[:IMPLEMENTS|EXTENDS*1..3]" not in cypher_arg

    async def test_get_callees_uses_parameter_suffix_when_entity_id_is_java_style(self, graph_client):
        """Resolved Java methods should constrain callee family expansion by parameter signature."""
        graph_client._query = AsyncMock(side_effect=[[], []])

        await graph_client.get_callees(
            "save",
            entity_id="repo::com.example.Service.save(String,int)",
        )

        cypher_arg = graph_client._query.call_args_list[0][0][0]
        kwargs = graph_client._query.call_args_list[0].kwargs
        assert "candidate_method.id ENDS WITH (candidate_method.name + $parameter_suffix)" in cypher_arg
        assert kwargs["parameter_suffix"] == "(String,int)"

    async def test_get_callees_depth_clamped_to_3(self, graph_client):
        """Passing depth=10 results in CALLS*1..3 in Cypher."""
        graph_client._query = AsyncMock(side_effect=[[], []])
        await graph_client.get_callees("someMethod", depth=10)
        cypher_arg = graph_client._query.call_args_list[0][0][0]
        assert "CALLS*1..3" in cypher_arg

    async def test_get_callees_maps_line_number_to_line_start(self, graph_client):
        """Verify line_number from Neo4j result maps to line_start on CallGraphNode."""
        graph_client._query = AsyncMock(side_effect=[
            [
                {
                    "name": "callee1",
                    "file_path": "src/service.py",
                    "repository": "my-repo",
                    "signature": "def callee1()",
                    "line_number": 55,
                    "entity_type": "Method",
                    "relationship_type": "CALLS",
                    "depth": 1,
                },
            ],
            [],
        ])
        results = await graph_client.get_callees("sourceMethod")
        assert len(results) == 1
        assert results[0].line_start == 55

    async def test_get_callees_empty_results(self, graph_client):
        """Returns empty list when no callees found."""
        graph_client._query = AsyncMock(side_effect=[[], []])
        results = await graph_client.get_callees("leafMethod")
        assert results == []

    async def test_get_callees_includes_hook_targets(self, graph_client):
        """Hook usage should be surfaced alongside call targets."""
        graph_client._query = AsyncMock(side_effect=[
            [],
            [
                {
                    "entity_id": "repo::hook.useState",
                    "name": "useState",
                    "file_path": "src/App.tsx",
                    "repository": "my-repo",
                    "line_number": 10,
                    "entity_type": "Hook",
                    "relationship_type": "USES_HOOK",
                    "depth": 1,
                },
            ],
        ])
        results = await graph_client.get_callees("render")
        assert len(results) == 1
        assert results[0].name == "useState"
        assert results[0].entity_type == "hook"
        assert results[0].relationship_type == "USES_HOOK"
        assert results[0].entity_id == "repo::hook.useState"

    async def test_get_callees_uses_limit_and_marks_truncation(self, graph_client):
        """Callee traversal should honor the merged result limit and mark truncation."""
        graph_client._query = AsyncMock(side_effect=[
            [
                {"name": "callee1", "file_path": "src/a.py", "line_number": 1, "entity_type": "Method", "relationship_type": "CALLS", "depth": 1},
                {"name": "callee2", "file_path": "src/b.py", "line_number": 2, "entity_type": "Method", "relationship_type": "CALLS", "depth": 1},
                {"name": "callee3", "file_path": "src/c.py", "line_number": 3, "entity_type": "Method", "relationship_type": "CALLS", "depth": 1},
            ],
            [],
        ])

        results = await graph_client.get_callees("sourceMethod", limit=2)

        kwargs = graph_client._query.call_args_list[0].kwargs
        assert kwargs["query_limit"] == 3
        assert [result.name for result in results] == ["callee1", "callee2"]
        assert all(result.truncated is True for result in results)


def _make_function_context_result(
    name="getData",
    id="my-repo::com.example.Service.getData",
    file_path="src/Service.java",
    repository="my-repo",
    code="public String getData() { ... }",
    signature="public String getData()",
    docstring="Gets data",
    class_name="Service",
):
    """Helper to build a mock Neo4j result dict for get_function_context."""
    return {
        "name": name,
        "id": id,
        "file_path": file_path,
        "repository": repository,
        "code": code,
        "signature": signature,
        "docstring": docstring,
        "class_name": class_name,
    }


class TestGetFunctionContext:
    """Tests for Neo4jReadBackend.get_function_context()."""

    async def test_get_function_context_returns_context(self, graph_client):
        """Basic return with all fields populated."""
        graph_client._query = AsyncMock(return_value=[_make_function_context_result()])
        graph_client.get_callers = AsyncMock(return_value=[])
        graph_client.get_callees = AsyncMock(return_value=[])

        result = await graph_client.get_function_context("getData")

        assert isinstance(result, FunctionContext)
        assert result.name == "getData"
        assert result.file_path == "src/Service.java"
        assert result.repository == "my-repo"
        assert result.code == "public String getData() { ... }"
        assert result.signature == "public String getData()"
        assert result.docstring == "Gets data"
        assert result.class_name == "Service"

    async def test_get_function_context_derives_full_name_from_id(self, graph_client):
        """When id='my-repo::com.example.Service.getData', full_name should be 'com.example.Service.getData'."""
        graph_client._query = AsyncMock(return_value=[
            _make_function_context_result(id="my-repo::com.example.Service.getData"),
        ])
        graph_client.get_callers = AsyncMock(return_value=[])
        graph_client.get_callees = AsyncMock(return_value=[])

        result = await graph_client.get_function_context("getData")

        assert result.full_name == "com.example.Service.getData"

    async def test_get_function_context_full_name_fallback(self, graph_client):
        """When id has no '::', falls back to name."""
        graph_client._query = AsyncMock(return_value=[
            _make_function_context_result(id="getData", name="getData"),
        ])
        graph_client.get_callers = AsyncMock(return_value=[])
        graph_client.get_callees = AsyncMock(return_value=[])

        result = await graph_client.get_function_context("getData")

        assert result.full_name == "getData"

    async def test_get_function_context_includes_callers_and_callees(self, graph_client):
        """Verify get_callers and get_callees are called, and results included."""
        graph_client._query = AsyncMock(return_value=[_make_function_context_result()])
        mock_callers = [
            CallGraphNode(name="caller1", file_path="src/main.py", signature="def caller1()"),
        ]
        mock_callees = [
            CallGraphNode(name="callee1", file_path="src/dao.py", signature="def callee1()"),
        ]
        graph_client.get_callers = AsyncMock(return_value=mock_callers)
        graph_client.get_callees = AsyncMock(return_value=mock_callees)

        result = await graph_client.get_function_context("getData")

        graph_client.get_callers.assert_called_once_with(
            "getData",
            repository=None,
            file_path=None,
            depth=1,
            entity_id="my-repo::com.example.Service.getData",
        )
        graph_client.get_callees.assert_called_once_with(
            "getData",
            repository=None,
            file_path=None,
            depth=1,
            entity_id="my-repo::com.example.Service.getData",
        )
        assert result.callers == mock_callers
        assert result.callees == mock_callees

    async def test_get_function_context_returns_none_when_not_found(self, graph_client):
        """_query returns [], method returns None."""
        graph_client._query = AsyncMock(return_value=[])

        result = await graph_client.get_function_context("nonExistent")

        assert result is None

    async def test_get_function_context_matches_constructor(self, graph_client):
        """Verify Cypher contains (m:Method OR m:Constructor)."""
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.get_function_context("MyConstructor")
        cypher_arg = graph_client._query.call_args[0][0]
        assert "(m:Method OR m:Constructor)" in cypher_arg

    async def test_get_function_context_with_filters(self, graph_client):
        """Verify file_path and repository filters work."""
        graph_client._query = AsyncMock(return_value=[_make_function_context_result()])
        graph_client.get_callers = AsyncMock(return_value=[])
        graph_client.get_callees = AsyncMock(return_value=[])

        await graph_client.get_function_context(
            "getData", repository="my-repo", file_path="Service.java"
        )

        cypher_arg = graph_client._query.call_args[0][0]
        assert "ENDS WITH" in cypher_arg
        assert graph_client._query.call_args.kwargs["file_path"] == "Service.java"
        assert "repository = $repository" in cypher_arg
        # Verify callers/callees also receive the filters
        graph_client.get_callers.assert_called_once_with(
            "getData",
            repository="my-repo",
            file_path="Service.java",
            depth=1,
            entity_id="my-repo::com.example.Service.getData",
        )
        graph_client.get_callees.assert_called_once_with(
            "getData",
            repository="my-repo",
            file_path="Service.java",
            depth=1,
            entity_id="my-repo::com.example.Service.getData",
        )


def _make_class_hierarchy_result(
    name="UserService",
    id="my-repo::com.example.UserService",
    file_path="src/UserService.java",
    repository="my-repo",
    labels=None,
    parents=None,
    children=None,
    interfaces=None,
    implementors=None,
    methods=None,
    fields=None,
    constructors=None,
):
    """Helper to build a mock Neo4j result dict for get_class_hierarchy."""
    return {
        "name": name,
        "id": id,
        "file_path": file_path,
        "repository": repository,
        "labels": labels if labels is not None else ["Class"],
        "parents": parents if parents is not None else ["BaseService"],
        "children": children if children is not None else ["AdminService"],
        "interfaces": interfaces if interfaces is not None else ["IUserService"],
        "implementors": implementors if implementors is not None else [],
        "methods": methods if methods is not None else ["getUser", "updateUser"],
        "fields": fields if fields is not None else ["name", "email"],
        "constructors": constructors if constructors is not None else ["UserService"],
    }


class TestGetClassHierarchy:
    """Tests for Neo4jReadBackend.get_class_hierarchy()."""

    async def test_get_class_hierarchy_returns_hierarchy(self, graph_client):
        """Basic return with all fields populated."""
        graph_client._query = AsyncMock(return_value=[_make_class_hierarchy_result()])

        result = await graph_client.get_class_hierarchy("UserService")

        assert isinstance(result, ClassHierarchy)
        assert result.name == "UserService"
        assert result.full_name == "com.example.UserService"
        assert result.file_path == "src/UserService.java"
        assert result.repository == "my-repo"
        assert result.is_interface is False
        assert result.parents == ["BaseService"]
        assert result.children == ["AdminService"]
        assert result.interfaces == ["IUserService"]
        assert result.implementors == []
        assert result.methods == ["getUser", "updateUser"]
        assert result.fields == ["name", "email"]
        assert result.constructors == ["UserService"]

    async def test_get_class_hierarchy_detects_interface_from_labels(self, graph_client):
        """When labels=['Interface'], is_interface=True."""
        graph_client._query = AsyncMock(return_value=[
            _make_class_hierarchy_result(labels=["Interface"]),
        ])

        result = await graph_client.get_class_hierarchy("IUserService")

        assert result.is_interface is True

    async def test_get_class_hierarchy_class_not_interface(self, graph_client):
        """When labels=['Class'], is_interface=False."""
        graph_client._query = AsyncMock(return_value=[
            _make_class_hierarchy_result(labels=["Class"]),
        ])

        result = await graph_client.get_class_hierarchy("UserService")

        assert result.is_interface is False

    async def test_get_class_hierarchy_includes_constructors(self, graph_client):
        """Verify constructors list populated."""
        graph_client._query = AsyncMock(return_value=[
            _make_class_hierarchy_result(constructors=["UserService", "UserService"]),
        ])

        result = await graph_client.get_class_hierarchy("UserService")

        assert result.constructors == ["UserService", "UserService"]

    async def test_get_class_hierarchy_filters_null_from_lists(self, graph_client):
        """Null entries in collected lists are filtered out."""
        graph_client._query = AsyncMock(return_value=[
            _make_class_hierarchy_result(
                parents=[None, "BaseService", None],
                children=[None],
                interfaces=["IUserService", None],
                implementors=[None, None],
                methods=["getUser", None, "updateUser"],
                fields=[None, "name"],
                constructors=[None, "UserService"],
            ),
        ])

        result = await graph_client.get_class_hierarchy("UserService")

        assert result.parents == ["BaseService"]
        assert result.children == []
        assert result.interfaces == ["IUserService"]
        assert result.implementors == []
        assert result.methods == ["getUser", "updateUser"]
        assert result.fields == ["name"]
        assert result.constructors == ["UserService"]

    async def test_get_class_hierarchy_returns_none_when_not_found(self, graph_client):
        """_query returns [], returns None."""
        graph_client._query = AsyncMock(return_value=[])

        result = await graph_client.get_class_hierarchy("NonExistentClass")

        assert result is None

    async def test_get_class_hierarchy_derives_full_name_from_id(self, graph_client):
        """full_name is derived by stripping 'repository::' prefix from id."""
        graph_client._query = AsyncMock(return_value=[
            _make_class_hierarchy_result(id="my-repo::com.example.UserService"),
        ])

        result = await graph_client.get_class_hierarchy("UserService")

        assert result.full_name == "com.example.UserService"

    async def test_get_class_hierarchy_full_name_fallback(self, graph_client):
        """When id has no '::', falls back to name."""
        graph_client._query = AsyncMock(return_value=[
            _make_class_hierarchy_result(id="UserService", name="UserService"),
        ])

        result = await graph_client.get_class_hierarchy("UserService")

        assert result.full_name == "UserService"

    async def test_get_class_hierarchy_with_filters(self, graph_client):
        """Verify file_path and repository filters work."""
        graph_client._query = AsyncMock(return_value=[_make_class_hierarchy_result()])

        await graph_client.get_class_hierarchy(
            "UserService", repository="my-repo", file_path="UserService.java"
        )

        cypher_arg = graph_client._query.call_args[0][0]
        assert "ENDS WITH" in cypher_arg
        assert graph_client._query.call_args.kwargs["file_path"] == "UserService.java"
        assert "c.repository = $repository" in cypher_arg

    async def test_get_class_hierarchy_orders_by_returned_aliases(self, graph_client):
        """Aggregated class hierarchy query must order by returned aliases, not pre-RETURN vars."""
        graph_client._query = AsyncMock(return_value=[_make_class_hierarchy_result()])

        await graph_client.get_class_hierarchy("UserService", repository="my-repo")

        cypher_arg = graph_client._query.call_args[0][0]
        assert "c.line_number AS line_number" in cypher_arg
        assert "ORDER BY file_path, line_number" in cypher_arg
        assert "ORDER BY c.file_path, c.line_number" not in cypher_arg

    async def test_get_class_hierarchy_avoids_hard_coded_implements_patterns(self, graph_client):
        """Hierarchy queries should use generic relationship filtering to avoid warnings."""
        graph_client._query = AsyncMock(return_value=[_make_class_hierarchy_result()])

        await graph_client.get_class_hierarchy("UserService", repository="my-repo")

        cypher_arg = graph_client._query.call_args[0][0]
        assert "type(parent_rel) = 'EXTENDS'" in cypher_arg
        assert "type(impl_rel) = 'IMPLEMENTS'" in cypher_arg
        assert "[:IMPLEMENTS]" not in cypher_arg

    async def test_get_class_hierarchy_raises_on_ambiguous_match(self, graph_client):
        """Class hierarchy lookups should not silently pick one of many matches."""
        graph_client._query = AsyncMock(return_value=[
            _make_class_hierarchy_result(file_path="src/a/UserService.java"),
            _make_class_hierarchy_result(file_path="src/b/UserService.java"),
        ])
        with pytest.raises(ValueError, match="ambiguous"):
            await graph_client.get_class_hierarchy("UserService")


class TestListRepositories:
    """Tests for Neo4jReadBackend.list_repositories()."""

    async def test_list_repositories_queries_repository_nodes(self, graph_client):
        """Verify Cypher contains MATCH (r:Repository) not aggregation."""
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.list_repositories()
        cypher_arg = graph_client._query.call_args[0][0]
        assert "MATCH (r:Repository)" in cypher_arg

    async def test_list_repositories_returns_query_results(self, graph_client):
        """Returns list of dicts from _query."""
        expected = [
            {"name": "repo-a", "entity_count": 100, "last_indexed_at": "2026-01-01"},
            {"name": "repo-b", "entity_count": 200, "last_indexed_at": "2026-02-01"},
        ]
        graph_client._query = AsyncMock(return_value=expected)
        results = await graph_client.list_repositories()
        assert results == expected
        assert len(results) == 2
        assert results[0]["name"] == "repo-a"
        assert results[1]["entity_count"] == 200

    async def test_list_repositories_empty(self, graph_client):
        """Returns empty list when no repositories found."""
        graph_client._query = AsyncMock(return_value=[])
        results = await graph_client.list_repositories()
        assert results == []


class TestGetRepositoryContext:
    """Tests for Neo4jReadBackend.get_repository_context()."""

    async def test_get_repository_context_returns_repository_metadata(self, graph_client):
        graph_client._query = AsyncMock(side_effect=[
            [{
                "name": "repo-a",
                "source": "https://github.com/example/repo-a",
                "entity_count": 123,
                "last_indexed_at": "2026-03-12T10:00:00+00:00",
                "last_commit_sha": "abc123",
            }],
            [{
                "files": 20,
                "classes": 4,
                "interfaces": 1,
                "methods": 16,
                "constructors": 2,
                "fields": 9,
                "packages_count": 3,
                "hooks": 1,
                "references": 5,
                "exports": 7,
                "languages": ["Python", "TypeScript"],
                "packages": ["src", "src.services"],
            }],
            [{"name": "App"}],
            [{"name": "main", "class_name": "App"}],
        ])

        result = await graph_client.get_repository_context("repo-a")

        assert isinstance(result, RepositoryContext)
        assert result.name == "repo-a"
        assert result.source == "https://github.com/example/repo-a"
        assert result.entity_count == 123
        assert result.last_indexed_at == "2026-03-12T10:00:00+00:00"
        assert result.last_commit_sha == "abc123"
        assert result.total_files == 20
        assert result.total_classes == 4
        assert result.total_interfaces == 1
        assert result.total_methods == 16
        assert result.total_constructors == 2
        assert result.total_fields == 9
        assert result.total_packages == 3
        assert result.total_hooks == 1
        assert result.total_references == 5
        assert result.total_exports == 7
        assert result.languages == ["Python", "TypeScript"]
        assert result.top_level_classes == ["App"]
        assert result.entry_points == ["App.main"]

    async def test_get_repository_context_returns_none_when_missing(self, graph_client):
        graph_client._query = AsyncMock(return_value=[])

        result = await graph_client.get_repository_context("missing")

        assert result is None
        assert graph_client._query.await_count == 1


class TestGetPackageContext:
    """Tests for Neo4jReadBackend.get_package_context()."""

    async def test_get_package_context_returns_package_members(self, graph_client):
        graph_client._query = AsyncMock(return_value=[
            {
                "name": "src.services",
                "id": "repo-a::src.services",
                "repository": "repo-a",
                "files": ["src/services/user.py"],
                "classes": ["UserService"],
                "interfaces": ["IUserService"],
                "methods": ["build_service"],
                "hooks": ["useService"],
                "references": ["requests.get"],
                "child_packages": ["src.services.internal"],
            }
        ])

        result = await graph_client.get_package_context("src.services", repository="repo-a")

        assert isinstance(result, PackageContext)
        assert result.name == "src.services"
        assert result.package_id == "repo-a::src.services"
        assert result.repository == "repo-a"
        assert result.files == ["src/services/user.py"]
        assert result.classes == ["UserService"]
        assert result.interfaces == ["IUserService"]
        assert result.methods == ["build_service"]
        assert result.hooks == ["useService"]
        assert result.references == ["requests.get"]
        assert result.child_packages == ["src.services.internal"]

    async def test_get_package_context_uses_member_file_paths_when_file_edges_are_missing(self, graph_client):
        graph_client._query = AsyncMock(return_value=[
            {
                "name": "com.example",
                "id": "repo-a::com.example",
                "repository": "repo-a",
                "files": ["src/java/com/example/Service.java"],
                "classes": ["Service"],
                "interfaces": [],
                "methods": ["run", "helper"],
                "hooks": [],
                "references": [],
                "child_packages": [],
            }
        ])

        result = await graph_client.get_package_context("com.example", repository="repo-a")

        cypher_arg = graph_client._query.call_args[0][0]
        assert "collect(DISTINCT member.file_path) AS files" in cypher_arg
        assert result is not None
        assert result.files == ["src/java/com/example/Service.java"]

    async def test_get_package_context_raises_on_ambiguous_match(self, graph_client):
        graph_client._query = AsyncMock(return_value=[
            {"name": "src.services", "id": "repo-a::src.services", "repository": "repo-a"},
            {"name": "src.services", "id": "repo-b::src.services", "repository": "repo-b"},
        ])

        with pytest.raises(ValueError, match="ambiguous"):
            await graph_client.get_package_context("src.services")


class TestGetCodebaseOverview:
    """Tests for Neo4jReadBackend.get_codebase_overview()."""

    async def test_get_codebase_overview_returns_stats(self, graph_client):
        """Basic return with all counts."""
        graph_client._query = AsyncMock(side_effect=[
            [{
                "files": 10,
                "classes": 5,
                "interfaces": 2,
                "methods": 20,
                "constructors": 3,
                "fields": 8,
                "packages_count": 4,
                "hooks": 1,
                "references": 6,
                "exports": 9,
                "languages": ["Java", "Python"],
            }],  # overview
            [{"name": "UserService"}, {"name": "OrderService"}],  # top classes
            [{"name": "handleRequest", "class_name": "ApiController"}],  # entry points
        ])

        result = await graph_client.get_codebase_overview()

        assert isinstance(result, CodebaseOverview)
        assert result.total_files == 10
        assert result.total_classes == 5
        assert result.total_interfaces == 2
        assert result.total_methods == 20
        assert result.total_constructors == 3
        assert result.total_fields == 8
        assert result.total_packages == 4
        assert result.total_hooks == 1
        assert result.total_references == 6
        assert result.total_exports == 9
        assert result.languages == ["Java", "Python"]
        assert result.top_level_classes == ["UserService", "OrderService"]
        assert result.entry_points == ["ApiController.handleRequest"]

    async def test_get_codebase_overview_includes_constructor_count(self, graph_client):
        """Verify total_constructors populated."""
        graph_client._query = AsyncMock(side_effect=[
            [{
                "files": 5,
                "classes": 2,
                "interfaces": 1,
                "methods": 8,
                "constructors": 7,
                "fields": 3,
                "packages_count": 2,
                "hooks": 0,
                "references": 1,
                "exports": 4,
                "languages": ["Java"],
            }],
            [],  # top classes
            [],  # entry points
        ])

        result = await graph_client.get_codebase_overview()

        assert result.total_constructors == 7

    async def test_get_codebase_overview_with_repository_filter(self, graph_client):
        """Verify repo filter in Cypher."""
        graph_client._query = AsyncMock(side_effect=[
            [{
                "files": 10,
                "classes": 5,
                "interfaces": 1,
                "methods": 20,
                "constructors": 3,
                "fields": 7,
                "packages_count": 2,
                "hooks": 0,
                "references": 3,
                "exports": 5,
                "languages": ["Java"],
            }],
            [],  # top classes
            [],  # entry points
        ])

        await graph_client.get_codebase_overview(repository="my-repo")

        # Check all three queries for repository filter
        overview_cypher = graph_client._query.call_args_list[0][0][0]
        assert "f.repository = $repository" in overview_cypher
        assert "c.repository = $repository" in overview_cypher
        assert "m.repository = $repository" in overview_cypher

        top_classes_cypher = graph_client._query.call_args_list[1][0][0]
        assert "c.repository = $repository" in top_classes_cypher

        entry_cypher = graph_client._query.call_args_list[2][0][0]
        assert "m.repository = $repository" in entry_cypher

    async def test_get_codebase_overview_includes_packages_when_requested(self, graph_client):
        """include_packages=True triggers package query."""
        graph_client._query = AsyncMock(side_effect=[
            [{
                "files": 10,
                "classes": 5,
                "interfaces": 2,
                "methods": 20,
                "constructors": 3,
                "fields": 6,
                "packages_count": 2,
                "hooks": 0,
                "references": 1,
                "exports": 4,
                "languages": ["Java"],
                "packages": [
                    {"id": "repo::com.example", "name": "com.example"},
                    {"id": "repo::com.util", "name": "com.util"},
                ],
            }],
            [],  # top classes
            [],  # entry points
        ])

        result = await graph_client.get_codebase_overview(include_packages=True)

        assert result.packages == ["com.example", "com.util"]
        # Verify Cypher contains Package match
        overview_cypher = graph_client._query.call_args_list[0][0][0]
        assert "Package" in overview_cypher

    async def test_get_codebase_overview_reconstructs_nested_dotnet_namespaces(self, graph_client):
        """Neo4j parity: codebase_overview.packages must reconstruct full
        dotted names from pkg.id, not return raw leaf-only pkg.name.
        Same _full_name_from_id pattern as the get_file_context fix.

        After the Cypher change, the main query's `packages` field is
        a list of {id, name} dicts instead of a list of strings.
        """
        graph_client._query = AsyncMock(side_effect=[
            # 1. Main overview query
            [{
                "files": 1,
                "classes": 1,
                "interfaces": 0,
                "methods": 0,
                "constructors": 0,
                "fields": 0,
                "packages_count": 3,
                "hooks": 0,
                "references": 1,
                "exports": 0,
                "languages": ["csharp"],
                "packages": [
                    {"id": "repo::Company", "name": "Company"},
                    {"id": "repo::Company.Product", "name": "Product"},
                    {"id": "repo::Company.Product.Services", "name": "Services"},
                ],
            }],
            [],  # 2. top classes
            [],  # 3. entry points
        ])

        result = await graph_client.get_codebase_overview(
            repository="repo", include_packages=True,
        )

        assert "Company.Product.Services" in result.packages, \
            f"Expected full dotted namespace; got: {result.packages}"
        assert "Services" not in result.packages, \
            f"Leaf-only 'Services' should not appear; got: {result.packages}"
        assert "Company" in result.packages
        assert "Company.Product" in result.packages

    async def test_get_codebase_overview_omits_packages_by_default(self, graph_client):
        """include_packages=False returns empty packages."""
        graph_client._query = AsyncMock(side_effect=[
            [{
                "files": 10,
                "classes": 5,
                "interfaces": 2,
                "methods": 20,
                "constructors": 3,
                "fields": 6,
                "packages_count": 2,
                "hooks": 0,
                "references": 1,
                "exports": 4,
                "languages": ["Java"],
            }],
            [],  # top classes
            [],  # entry points
        ])

        result = await graph_client.get_codebase_overview()

        assert result.packages == []

    async def test_get_codebase_overview_entry_points_from_stereotypes(self, graph_client):
        """Verify Cypher uses 'endpoint' IN m.stereotypes."""
        graph_client._query = AsyncMock(side_effect=[
            [{
                "files": 10,
                "classes": 5,
                "interfaces": 0,
                "methods": 20,
                "constructors": 3,
                "fields": 0,
                "packages_count": 0,
                "hooks": 0,
                "references": 0,
                "exports": 0,
                "languages": ["Java"],
            }],
            [],  # top classes
            [{"name": "handleRequest", "class_name": "ApiController"}],  # entry points
        ])

        await graph_client.get_codebase_overview()

        entry_cypher = graph_client._query.call_args_list[2][0][0]
        assert "'endpoint' IN m.stereotypes" in entry_cypher

    async def test_get_codebase_overview_formats_entry_points(self, graph_client):
        """Entry points formatted as 'ClassName.methodName'."""
        graph_client._query = AsyncMock(side_effect=[
            [{
                "files": 10,
                "classes": 5,
                "interfaces": 0,
                "methods": 20,
                "constructors": 3,
                "fields": 0,
                "packages_count": 0,
                "hooks": 0,
                "references": 0,
                "exports": 0,
                "languages": ["Java"],
            }],
            [],  # top classes
            [
                {"name": "handleRequest", "class_name": "ApiController"},
                {"name": "main", "class_name": None},
            ],  # entry points
        ])

        result = await graph_client.get_codebase_overview()

        assert result.entry_points == ["ApiController.handleRequest", "main"]

    async def test_get_codebase_overview_excludes_test_entities(self, graph_client):
        """top_level_classes and entry_points must exclude nodes whose
        stereotypes contain 'test', via a null-safe Cypher idiom.

        Constellation never writes `is_test` as a Neo4j property — it
        only writes `stereotypes`. So the filter MUST use
        `'test' IN coalesce(x.stereotypes, [])`, NOT `x.is_test`.
        Otherwise `NOT null = null` (falsy) filters out every row.
        Covers both with-repository and without-repository branches.
        """
        # Shared stub overview row
        overview_row = {
            "files": 10, "classes": 5, "interfaces": 2,
            "methods": 20, "constructors": 3, "fields": 6,
            "packages_count": 2, "hooks": 0, "references": 1,
            "exports": 4, "languages": ["Python"],
        }

        # --- with-repository branch ---
        graph_client._query = AsyncMock(side_effect=[
            [overview_row],
            [],  # top classes
            [],  # entry points
        ])
        await graph_client.get_codebase_overview(repository="repo")

        top_classes_cypher = graph_client._query.call_args_list[1][0][0]
        entry_points_cypher = graph_client._query.call_args_list[2][0][0]

        # top_classes filter: null-safe stereotype-array idiom, negated.
        assert (
            "'test' IN coalesce(c.stereotypes, [])" in top_classes_cypher
            or "'test' IN COALESCE(c.stereotypes, [])" in top_classes_cypher
        ), (
            f"top_classes Cypher must use the null-safe test-stereotype "
            f"filter idiom; got: {top_classes_cypher}"
        )
        assert "NOT 'test' IN" in top_classes_cypher, (
            f"top_classes filter must be negated (NOT ...); "
            f"got: {top_classes_cypher}"
        )
        # Reject the old broken form
        assert "NOT c.is_test" not in top_classes_cypher, (
            f"top_classes must NOT use the broken `NOT c.is_test` filter "
            f"(is_test is never written as a Neo4j property); "
            f"got: {top_classes_cypher}"
        )

        # entry_points filter: same idiom on Method nodes.
        assert (
            "'test' IN coalesce(m.stereotypes, [])" in entry_points_cypher
            or "'test' IN COALESCE(m.stereotypes, [])" in entry_points_cypher
        ), (
            f"entry_points Cypher must use the null-safe test-stereotype "
            f"filter idiom; got: {entry_points_cypher}"
        )
        assert "NOT 'test' IN" in entry_points_cypher, (
            f"entry_points filter must be negated (NOT ...); "
            f"got: {entry_points_cypher}"
        )
        assert "NOT m.is_test" not in entry_points_cypher, (
            f"entry_points must NOT use the broken `NOT m.is_test` filter; "
            f"got: {entry_points_cypher}"
        )

        # --- without-repository branch ---
        # The Cypher templates differ between with-repo and without-repo
        # (top_classes_filter / entry_filter variables), so the filter must
        # be present in both.
        graph_client._query = AsyncMock(side_effect=[
            [overview_row],
            [],  # top classes
            [],  # entry points
        ])
        await graph_client.get_codebase_overview()  # no repository

        top_classes_cypher_no_repo = graph_client._query.call_args_list[1][0][0]
        entry_points_cypher_no_repo = graph_client._query.call_args_list[2][0][0]

        assert (
            "'test' IN coalesce(c.stereotypes, [])" in top_classes_cypher_no_repo
            or "'test' IN COALESCE(c.stereotypes, [])" in top_classes_cypher_no_repo
        ), (
            f"top_classes Cypher (no-repo branch) must use the null-safe "
            f"test-stereotype filter idiom; got: {top_classes_cypher_no_repo}"
        )
        assert "NOT 'test' IN" in top_classes_cypher_no_repo, (
            f"top_classes filter (no-repo branch) must be negated; "
            f"got: {top_classes_cypher_no_repo}"
        )
        assert (
            "'test' IN coalesce(m.stereotypes, [])" in entry_points_cypher_no_repo
            or "'test' IN COALESCE(m.stereotypes, [])" in entry_points_cypher_no_repo
        ), (
            f"entry_points Cypher (no-repo branch) must use the null-safe "
            f"test-stereotype filter idiom; got: {entry_points_cypher_no_repo}"
        )
        assert "NOT 'test' IN" in entry_points_cypher_no_repo, (
            f"entry_points filter (no-repo branch) must be negated; "
            f"got: {entry_points_cypher_no_repo}"
        )

    async def test_get_codebase_overview_empty_results(self, graph_client):
        """Returns CodebaseOverview with defaults when no data."""
        graph_client._query = AsyncMock(return_value=[])

        result = await graph_client.get_codebase_overview()

        assert isinstance(result, CodebaseOverview)
        assert result.total_files == 0
        assert result.total_classes == 0
        assert result.total_interfaces == 0
        assert result.total_methods == 0
        assert result.total_constructors == 0
        assert result.total_fields == 0
        assert result.total_packages == 0
        assert result.total_hooks == 0
        assert result.total_references == 0
        assert result.total_exports == 0
        assert result.languages == []
        assert result.packages == []
        assert result.top_level_classes == []
        assert result.entry_points == []


def _make_impact_caller(
    name="checkOrder",
    file_path="src/OrderHelper.java",
    repository="my-repo",
    signature="void checkOrder()",
    line_number=30,
    stereotypes=None,
    depth=1,
    entity_id="my-repo::Caller.default",
):
    """Helper to build a mock Neo4j result dict for get_impact callers."""
    return {
        "entity_id": entity_id,
        "name": name,
        "file_path": file_path,
        "repository": repository,
        "signature": signature,
        "line_number": line_number,
        "stereotypes": stereotypes if stereotypes is not None else [],
        "depth": depth,
    }


def _make_impact_target(
    name="processOrder",
    entity_id="my-repo::OrderService.processOrder",
    file_path="src/OrderService.java",
    repository="my-repo",
):
    return {
        "name": name,
        "id": entity_id,
        "file_path": file_path,
        "repository": repository,
    }


class TestGetImpact:
    """Tests for Neo4jReadBackend.get_impact()."""

    async def test_get_impact_returns_result(self, graph_client):
        """Basic return with target info and counts."""
        graph_client._query = AsyncMock(side_effect=[
            # First call: method info
            [_make_impact_target()],
            # Second call: callers
            [
                _make_impact_caller(name="testProcessOrder", file_path="test/OrderTest.java", stereotypes=["test"], line_number=15, depth=1),
                _make_impact_caller(name="handleOrderEndpoint", file_path="src/OrderController.java", stereotypes=["endpoint"], line_number=20, depth=2),
                _make_impact_caller(name="checkOrder", file_path="src/OrderHelper.java", stereotypes=[], line_number=30, depth=1),
            ],
        ])

        result = await graph_client.get_impact("processOrder")

        assert isinstance(result, ImpactResult)
        assert result.target_name == "processOrder"
        assert result.target_file == "src/OrderService.java"
        assert result.target_repository == "my-repo"
        assert result.total_callers == 3
        assert result.test_count == 1
        assert result.endpoint_count == 1

    async def test_get_impact_categorizes_by_stereotypes(self, graph_client):
        """Test with stereotypes=['test'] and stereotypes=['endpoint']."""
        graph_client._query = AsyncMock(side_effect=[
            [_make_impact_target()],
            [
                _make_impact_caller(
                    name="testProcessOrder",
                    file_path="test/OrderTest.java",
                    stereotypes=["test"],
                    line_number=15,
                    depth=1,
                    entity_id="my-repo::OrderTest.testProcessOrder",
                ),
                _make_impact_caller(
                    name="handleOrderEndpoint",
                    file_path="src/OrderController.java",
                    stereotypes=["endpoint"],
                    line_number=20,
                    depth=2,
                    entity_id="my-repo::OrderController.handleOrderEndpoint",
                ),
                _make_impact_caller(
                    name="checkOrder",
                    file_path="src/OrderHelper.java",
                    stereotypes=[],
                    line_number=30,
                    depth=1,
                    entity_id="my-repo::OrderHelper.checkOrder",
                ),
            ],
        ])

        result = await graph_client.get_impact("processOrder")

        assert len(result.affected_tests) == 1
        assert result.affected_tests[0].name == "testProcessOrder"
        assert result.affected_tests[0].is_test is True
        # entity_id must propagate through so agents can chain without re-resolving
        assert result.affected_tests[0].entity_id == "my-repo::OrderTest.testProcessOrder"

        assert len(result.affected_endpoints) == 1
        assert result.affected_endpoints[0].name == "handleOrderEndpoint"
        assert result.affected_endpoints[0].is_endpoint is True
        assert result.affected_endpoints[0].entity_id == "my-repo::OrderController.handleOrderEndpoint"

        assert len(result.other_callers) == 1
        assert result.other_callers[0].name == "checkOrder"
        assert result.other_callers[0].entity_id == "my-repo::OrderHelper.checkOrder"

    async def test_get_impact_fallback_heuristics(self, graph_client):
        """'test' in name -> is_test, 'controller' in name -> is_endpoint."""
        graph_client._query = AsyncMock(side_effect=[
            [_make_impact_target()],
            [
                _make_impact_caller(name="testOrderFlow", file_path="src/OrderFlow.java", stereotypes=[], line_number=10, depth=1),
                _make_impact_caller(name="orderController", file_path="src/OrderCtrl.java", stereotypes=[], line_number=20, depth=2),
            ],
        ])

        result = await graph_client.get_impact("processOrder")

        assert len(result.affected_tests) == 1
        assert result.affected_tests[0].name == "testOrderFlow"
        assert result.affected_tests[0].is_test is True

        assert len(result.affected_endpoints) == 1
        assert result.affected_endpoints[0].name == "orderController"
        assert result.affected_endpoints[0].is_endpoint is True

    async def test_get_impact_returns_none_when_not_found(self, graph_client):
        """Method query returns [], returns None."""
        graph_client._query = AsyncMock(return_value=[])

        result = await graph_client.get_impact("nonExistentMethod")

        assert result is None

    async def test_get_impact_summary_only(self, graph_client):
        """Counts populated but caller lists empty."""
        graph_client._query = AsyncMock(side_effect=[
            [_make_impact_target()],
            [
                _make_impact_caller(name="testProcessOrder", stereotypes=["test"], depth=1),
                _make_impact_caller(name="handleOrderEndpoint", stereotypes=["endpoint"], depth=2),
                _make_impact_caller(name="checkOrder", stereotypes=[], depth=1),
            ],
        ])

        result = await graph_client.get_impact("processOrder", summary_only=True)

        assert result.test_count == 1
        assert result.endpoint_count == 1
        assert result.total_callers == 3
        assert result.affected_tests == []
        assert result.affected_endpoints == []
        assert result.other_callers == []

    async def test_get_impact_limit_caps_categories(self, graph_client):
        """limit=1 caps each list."""
        graph_client._query = AsyncMock(side_effect=[
            [_make_impact_target()],
            [
                _make_impact_caller(name="test1", stereotypes=["test"], depth=1),
                _make_impact_caller(name="test2", stereotypes=["test"], depth=2),
                _make_impact_caller(name="endpoint1", stereotypes=["endpoint"], depth=1),
                _make_impact_caller(name="endpoint2", stereotypes=["endpoint"], depth=2),
                _make_impact_caller(name="other1", stereotypes=[], depth=1),
                _make_impact_caller(name="other2", stereotypes=[], depth=2),
            ],
        ])

        result = await graph_client.get_impact("processOrder", limit=1)

        assert len(result.affected_tests) == 1
        assert len(result.affected_endpoints) == 1
        assert len(result.other_callers) == 1

    async def test_get_impact_truncated_flag(self, graph_client):
        """truncated=True when limit cuts results."""
        graph_client._query = AsyncMock(side_effect=[
            [_make_impact_target()],
            [
                _make_impact_caller(name="test1", stereotypes=["test"], depth=1),
                _make_impact_caller(name="test2", stereotypes=["test"], depth=2),
                _make_impact_caller(name="endpoint1", stereotypes=["endpoint"], depth=1),
            ],
        ])

        result = await graph_client.get_impact("processOrder", limit=1)

        assert result.truncated is True

    async def test_get_impact_maps_line_number(self, graph_client):
        """line_number mapped to line_start."""
        graph_client._query = AsyncMock(side_effect=[
            [_make_impact_target()],
            [
                _make_impact_caller(name="testProcessOrder", stereotypes=["test"], line_number=42, depth=1),
            ],
        ])

        result = await graph_client.get_impact("processOrder")

        assert result.affected_tests[0].line_start == 42

    async def test_get_impact_matches_constructor(self, graph_client):
        """Cypher contains (m:Method OR m:Constructor)."""
        graph_client._query = AsyncMock(side_effect=[
            [_make_impact_target(name="MyConstructor", entity_id="my-repo::MyClass.MyConstructor", file_path="src/MyClass.java")],
            [],
        ])

        await graph_client.get_impact("MyConstructor")

        # Both Cypher queries should contain the Method OR Constructor match
        method_cypher = graph_client._query.call_args_list[0][0][0]
        callers_cypher = graph_client._query.call_args_list[1][0][0]
        assert "(m:Method OR m:Constructor)" in method_cypher
        assert "(m:Method OR m:Constructor)" in callers_cypher

    async def test_get_impact_uses_type_relationships_not_overrides(self, graph_client):
        """Impact analysis should derive method family from the current graph schema."""
        graph_client._query = AsyncMock(side_effect=[
            [_make_impact_target()],
            [],
        ])
        await graph_client.get_impact("processOrder")
        callers_cypher = graph_client._query.call_args_list[1][0][0]
        assert "IMPLEMENTS" in callers_cypher
        assert "EXTENDS" in callers_cypher
        assert "OVERRIDES" not in callers_cypher

    async def test_get_impact_uses_parameter_suffix_for_java_methods(self, graph_client):
        """Impact analysis should avoid over-merging Java overloads across type families."""
        graph_client._query = AsyncMock(side_effect=[
            [_make_impact_target(entity_id="repo::com.example.Service.save(String,int)")],
            [],
        ])

        await graph_client.get_impact("save")

        callers_cypher = graph_client._query.call_args_list[1][0][0]
        kwargs = graph_client._query.call_args_list[1].kwargs
        assert "candidate_method.id ENDS WITH (candidate_method.name + $parameter_suffix)" in callers_cypher
        assert kwargs["parameter_suffix"] == "(String,int)"


class TestFindSymbols:
    """Tests for Neo4jReadBackend.find_symbols()."""

    async def test_find_symbols_queries_requested_entity_types(self, graph_client):
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.find_symbols("use", entity_types=["hook", "reference"])
        cypher_arg = graph_client._query.call_args[0][0]
        kwargs = graph_client._query.call_args.kwargs
        assert "labels(n)" in cypher_arg
        assert kwargs["labels"] == ["Hook", "Reference"]

    async def test_find_symbols_applies_file_pattern_filter(self, graph_client):
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.find_symbols("Auth", file_pattern="*/api/*", repository="repo")
        cypher_arg = graph_client._query.call_args[0][0]
        kwargs = graph_client._query.call_args.kwargs
        assert "n.repository = $repository" in cypher_arg
        assert "n.file_path =~ $file_regex" in cypher_arg
        assert kwargs["file_regex"] == "^.*/api/.*$"

    async def test_find_symbols_applies_language_and_stereotype_filters(self, graph_client):
        graph_client._query = AsyncMock(return_value=[])

        await graph_client.find_symbols(
            "Auth",
            repository="repo",
            language="Python",
            stereotype="test",
        )

        cypher_arg = graph_client._query.call_args[0][0]
        kwargs = graph_client._query.call_args.kwargs
        assert "n.language = $language" in cypher_arg
        assert "$stereotype IN coalesce(n.stereotypes, [])" in cypher_arg
        assert kwargs["language"] == "Python"
        assert kwargs["stereotype"] == "test"

    async def test_find_symbols_maps_results(self, graph_client):
        graph_client._query = AsyncMock(return_value=[
            {
                "id": "repo::src/App.tsx::useState",
                "name": "useState",
                "file_path": "src/App.tsx",
                "repository": "repo",
                "line_number": 12,
                "line_end": 12,
                "signature": None,
                "code": None,
                "entity_type": "Hook",
                "language": "TypeScript",
                "return_type": None,
                "modifiers": [],
                "stereotypes": [],
                "content_hash": "deadbeef",
                "properties": {"symbol": "useState"},
            },
        ])
        results = await graph_client.find_symbols("useState", entity_types=["hook"])
        assert len(results) == 1
        assert isinstance(results[0], CodeEntity)
        assert results[0].entity_id == "repo::src/App.tsx::useState"
        assert results[0].entity_type == "hook"
        assert results[0].line_start == 12
        assert results[0].language == "TypeScript"
        assert results[0].content_hash == "deadbeef"
        assert results[0].properties == {"symbol": "useState"}

    async def test_find_symbols_maps_results_with_null_list_fields(self, graph_client):
        graph_client._query = AsyncMock(return_value=[
            {
                "id": "repo::ref:client.fetch",
                "name": "client.fetch",
                "file_path": "src/service.java",
                "repository": "repo",
                "line_number": 19,
                "line_end": None,
                "signature": None,
                "code": None,
                "entity_type": "Reference",
                "language": "Java",
                "return_type": None,
                "modifiers": None,
                "stereotypes": None,
                "content_hash": None,
                "properties": {"symbol": "client.fetch"},
            },
        ])

        results = await graph_client.find_symbols("client.fetch", entity_types=["reference"], exact=True)

        assert len(results) == 1
        assert results[0].entity_type == "reference"
        assert results[0].modifiers == []
        assert results[0].stereotypes == []
        assert results[0].properties == {"symbol": "client.fetch"}

    async def test_find_symbols_avoids_reserved_query_parameter_name(self, graph_client):
        graph_client._query = AsyncMock(return_value=[])

        await graph_client.find_symbols("helper", repository="repo", exact=True)

        kwargs = graph_client._query.call_args.kwargs
        cypher_arg = graph_client._query.call_args[0][0]
        assert "toLower(n.name) = toLower($search_query)" in cypher_arg
        assert "CASE\n                         WHEN toLower(n.name) = toLower($search_query)" in cypher_arg
        assert kwargs["search_query"] == "helper"
        assert "query" not in kwargs

    async def test_find_symbols_applies_code_mode_none_by_default(self, graph_client):
        """find_symbols should default to code_mode='none' so identifier
        lookups don't dump full source bodies unnecessarily."""
        graph_client._query = AsyncMock(return_value=[
            {
                "id": "repo::Foo",
                "name": "Foo",
                "file_path": "Foo.java",
                "repository": "repo",
                "line_number": 1,
                "line_end": 100,
                "signature": "public class Foo",
                "code": "public class Foo { /* 100 lines */ }",
                "entity_type": "Class",
                "language": "Java",
                "return_type": None,
                "modifiers": ["public"],
                "stereotypes": [],
                "content_hash": "h",
                "properties": {},
            },
        ])

        results = await graph_client.find_symbols("Foo")

        assert len(results) == 1
        assert results[0].code is None, \
            f"default code_mode should be 'none'; got code={results[0].code!r}"

    async def test_find_symbols_honors_code_mode_preview(self, graph_client):
        """find_symbols(code_mode='preview') should return at most 10 lines
        with a truncation marker. Neo4j's _apply_code_mode helper appends
        '\\n... (truncated)' when the source has more than 10 lines. This
        test is a direct regression guard against any future bypass of
        _apply_code_mode in find_symbols (currently the preview branch is
        only indirectly covered by search_code tests)."""
        long_code = "\n".join(f"line{i}" for i in range(20))  # 20 lines
        graph_client._query = AsyncMock(return_value=[
            {
                "id": "repo::Foo",
                "name": "Foo",
                "file_path": "Foo.java",
                "repository": "repo",
                "line_number": 1,
                "line_end": 20,
                "signature": "public class Foo",
                "code": long_code,
                "entity_type": "Class",
                "language": "Java",
                "return_type": None,
                "modifiers": ["public"],
                "stereotypes": [],
                "content_hash": "h",
                "properties": {},
            },
        ])

        results = await graph_client.find_symbols("Foo", code_mode="preview")

        assert len(results) == 1
        code = results[0].code
        assert code is not None
        assert "line0" in code
        assert "line9" in code
        assert "line10" not in code, \
            f"Preview should drop line 10 and later; got: {code!r}"
        assert "... (truncated)" in code, \
            f"Preview should append truncation marker; got: {code!r}"


class TestGetFileContext:
    """Tests for Neo4jReadBackend.get_file_context()."""

    async def test_get_file_context_reconstructs_nested_dotnet_namespace(self, graph_client):
        """Neo4j parity with the Postgres fix: get_file_context.packages
        must reconstruct full dotted names from pkg.id, not return raw
        pkg.name. Same _full_name_from_id pattern as get_package_context
        (neo4j.py:976).

        After the Cypher change, `packages` in the main query result is
        a list of {id, name} dicts instead of a list of strings.
        """
        graph_client._query = AsyncMock(side_effect=[
            # 1. _resolve_file_target → single file row
            [{
                "name": "AuthService.cs",
                "file_path": "src/AuthService.cs",
                "repository": "repo",
                "language": "csharp",
            }],
            # 2. Main get_file_context query result
            [{
                "name": "AuthService.cs",
                "file_path": "src/AuthService.cs",
                "repository": "repo",
                "language": "csharp",
                "content_hash": "h",
                # New shape: list of dicts with id + name, from the
                # `collect(DISTINCT {id: pkg.id, name: pkg.name})` change.
                "packages": [
                    {"id": "repo::Company.Product.Services", "name": "Services"},
                ],
                "classes": ["AuthService"],
                "interfaces": [],
                "top_level_methods": [],
                "constructors": [],
                "fields": [],
                "references": [],
                "exports": [],
                "hooks": [],
            }],
        ])

        result = await graph_client.get_file_context("AuthService.cs", repository="repo")

        assert result is not None
        assert result.packages == ["Company.Product.Services"], \
            f"Expected full dotted namespace; got: {result.packages}"

    async def test_get_file_context_returns_file_details(self, graph_client):
        graph_client._query = AsyncMock(side_effect=[
            [{
                "name": "App.tsx",
                "file_path": "src/App.tsx",
                "repository": "repo",
                "language": "TypeScript",
                "content_hash": "deadbeef",
            }],
            [{
                "name": "App.tsx",
                "file_path": "src/App.tsx",
                "repository": "repo",
                "language": "TypeScript",
                "content_hash": "deadbeef",
                "packages": [{"id": "repo::src.components", "name": "src.components"}],
                "classes": ["App"],
                "interfaces": ["Props"],
                "top_level_methods": ["renderApp"],
                "hooks": ["useState"],
                "constructors": ["App"],
                "fields": ["title"],
                "references": ["React.Fragment"],
                "exports": [
                    {
                        "id": "repo::src/App.tsx::App",
                        "name": "App",
                        "file_path": "src/App.tsx",
                        "repository": "repo",
                        "line_start": 3,
                        "line_end": 40,
                        "entity_type": "class",
                        "language": "TypeScript",
                        "content_hash": "deadbeef",
                        "properties": {"export_type": "default"},
                    },
                ],
            }],
        ])
        result = await graph_client.get_file_context("App.tsx")
        assert isinstance(result, FileContext)
        assert result.file_path == "src/App.tsx"
        assert result.language == "TypeScript"
        assert result.content_hash == "deadbeef"
        assert result.packages == ["src.components"]
        assert result.classes == ["App"]
        assert result.interfaces == ["Props"]
        assert result.top_level_methods == ["renderApp"]
        assert result.hooks == ["useState"]
        assert result.constructors == ["App"]
        assert result.fields == ["title"]
        assert result.references == ["React.Fragment"]
        assert result.exports[0].name == "App"
        assert result.exports[0].entity_id == "repo::src/App.tsx::App"
        assert result.exports[0].content_hash == "deadbeef"
        assert result.exports[0].properties == {"export_type": "default"}

    async def test_get_file_context_uses_file_scoped_members_for_packages_and_class_members(self, graph_client):
        graph_client._query = AsyncMock(side_effect=[
            [{
                "name": "Service.java",
                "file_path": "src/Service.java",
                "repository": "repo",
                "language": "Java",
                "content_hash": "cafebabe",
            }],
            [{
                "name": "Service.java",
                "file_path": "src/Service.java",
                "repository": "repo",
                "language": "Java",
                "content_hash": "cafebabe",
                "packages": [{"id": "repo::com.example", "name": "com.example"}],
                "classes": ["Service"],
                "interfaces": [],
                "top_level_methods": [],
                "hooks": [],
                "constructors": ["Service"],
                "fields": ["client"],
                "references": ["client.fetch"],
                "exports": [],
            }],
        ])

        result = await graph_client.get_file_context("Service.java", repository="repo")

        cypher_arg = graph_client._query.call_args_list[1][0][0]
        assert "member.file_path = f.file_path" in cypher_arg
        assert "ctor.repository = f.repository AND ctor.file_path = f.file_path" in cypher_arg
        assert "field.repository = f.repository AND field.file_path = f.file_path" in cypher_arg
        assert "ref.repository = f.repository AND ref.file_path = f.file_path" in cypher_arg
        assert "OPTIONAL MATCH (f)-[:IN_PACKAGE]->(pkg:Package)" not in cypher_arg
        assert result is not None
        assert result.packages == ["com.example"]
        assert result.constructors == ["Service"]
        assert result.fields == ["client"]
        assert result.references == ["client.fetch"]

    async def test_get_file_context_raises_on_ambiguous_match(self, graph_client):
        graph_client._query = AsyncMock(return_value=[
            {"name": "config.py", "file_path": "a/config.py", "repository": "repo", "language": "Python"},
            {"name": "config.py", "file_path": "b/config.py", "repository": "repo", "language": "Python"},
        ])
        with pytest.raises(ValueError, match="ambiguous"):
            await graph_client.get_file_context("config.py")

    async def test_get_file_context_ambiguity_message_when_repo_provided(self, graph_client):
        """Symmetric to the Postgres fix — Neo4j ambiguity message
        must not suggest 'Provide repository=' when repo was provided."""
        graph_client._query = AsyncMock(return_value=[
            {"name": "tasks.py", "file_path": "src/a/tasks.py",
             "repository": "repo", "language": "Python"},
            {"name": "tasks.py", "file_path": "src/b/tasks.py",
             "repository": "repo", "language": "Python"},
        ])
        with pytest.raises(ValueError) as excinfo:
            await graph_client.get_file_context("tasks.py", repository="repo")
        msg = str(excinfo.value)
        assert "mbiguous" in msg
        assert "Provide repository=" not in msg


class TestGetHookUsage:
    """Tests for Neo4jReadBackend.get_hook_usage()."""

    async def test_get_hook_usage_returns_callers(self, graph_client):
        graph_client._query = AsyncMock(return_value=[
            {
                "name": "renderApp",
                "file_path": "src/App.tsx",
                "repository": "repo",
                "signature": "function renderApp()",
                "line_number": 15,
                "entity_type": "Method",
                "relationship_type": "USES_HOOK",
            },
        ])
        results = await graph_client.get_hook_usage("useState")
        assert len(results) == 1
        assert results[0].name == "renderApp"
        assert results[0].entity_type == "method"
        assert results[0].relationship_type == "USES_HOOK"

    async def test_get_hook_usage_applies_filters(self, graph_client):
        graph_client._query = AsyncMock(return_value=[])
        await graph_client.get_hook_usage("useEffect", repository="repo", file_pattern="*/ui/*")
        cypher_arg = graph_client._query.call_args[0][0]
        kwargs = graph_client._query.call_args.kwargs
        assert "h.name = $hook_name" in cypher_arg
        assert "m.repository = $repository" in cypher_arg
        assert "m.file_path =~ $file_regex" in cypher_arg
        assert kwargs["file_regex"] == "^.*/ui/.*$"

    async def test_get_hook_usage_applies_language_and_stereotype_filters(self, graph_client):
        graph_client._query = AsyncMock(return_value=[])

        await graph_client.get_hook_usage(
            "useEffect",
            repository="repo",
            language="TypeScript",
            stereotype="test",
        )

        cypher_arg = graph_client._query.call_args[0][0]
        kwargs = graph_client._query.call_args.kwargs
        assert "m.language = $language" in cypher_arg
        assert "$stereotype IN coalesce(m.stereotypes, [])" in cypher_arg
        assert kwargs["language"] == "TypeScript"
        assert kwargs["stereotype"] == "test"

    async def test_get_hook_usage_uses_limit_and_marks_truncation(self, graph_client):
        graph_client._query = AsyncMock(return_value=[
            {"name": "render1", "file_path": "src/A.tsx", "line_number": 1, "entity_type": "Method", "relationship_type": "USES_HOOK"},
            {"name": "render2", "file_path": "src/B.tsx", "line_number": 2, "entity_type": "Method", "relationship_type": "USES_HOOK"},
            {"name": "render3", "file_path": "src/C.tsx", "line_number": 3, "entity_type": "Method", "relationship_type": "USES_HOOK"},
        ])

        results = await graph_client.get_hook_usage("useState", limit=2)

        kwargs = graph_client._query.call_args.kwargs
        assert kwargs["query_limit"] == 3
        assert [result.name for result in results] == ["render1", "render2"]
        assert all(result.truncated is True for result in results)

    async def test_get_hook_usage_returns_entity_id_for_chaining(self, graph_client):
        """Backend parity with Postgres and with get_callers: hook-usage
        results must carry entity_id so agents can chain to get_function_context
        or get_callers without re-resolving by name."""
        graph_client._query = AsyncMock(return_value=[
            {
                "entity_id": "repo::App.renderApp",
                "name": "renderApp",
                "file_path": "src/App.tsx",
                "repository": "repo",
                "signature": "function renderApp()",
                "line_number": 42,
                "entity_type": "Method",
                "relationship_type": "USES_HOOK",
            },
        ])

        results = await graph_client.get_hook_usage("useState")

        assert len(results) == 1
        assert results[0].entity_id == "repo::App.renderApp", (
            f"CallGraphNode.entity_id must be populated for hook usage; got: {results[0].entity_id}"
        )
        # Defensive: the Cypher RETURN must expose m.id AS entity_id
        cypher_arg = graph_client._query.call_args[0][0]
        assert "m.id AS entity_id" in cypher_arg, (
            f"get_hook_usage Cypher RETURN must expose m.id AS entity_id; got: {cypher_arg}"
        )
