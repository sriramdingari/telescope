"""Tests for Telescope MCP server tool handlers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from telescope.models import (
    CodeEntity, CallGraphNode, FileContext, FunctionContext,
    ClassHierarchy, CodebaseOverview, ImpactResult, PackageContext, RepositoryContext,
)
from telescope.server import (
    search_code, get_callers, get_callees, get_function_context,
    get_class_hierarchy, list_repositories, get_codebase_overview, get_file_context,
    get_hook_usage, get_impact, find_symbols, get_package_context, get_repository_context,
)


@pytest.fixture()
def mock_graph():
    return AsyncMock()


@pytest.fixture()
def mock_ctx(mock_graph):
    ctx = MagicMock()
    ctx.request_context.lifespan_context.graph = mock_graph
    return ctx


# =============================================================================
# search_code
# =============================================================================


class TestSearchCodeTool:
    async def test_validates_entity_type(self, mock_ctx):
        """Invalid entity_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid entity_type 'bogus'"):
            await search_code(query="auth", entity_type="bogus", ctx=mock_ctx)

    async def test_accepts_constructor_entity_type(self, mock_ctx, mock_graph):
        """'constructor' is a valid entity_type."""
        mock_graph.search_code.return_value = []
        result = await search_code(query="init", entity_type="constructor", ctx=mock_ctx)
        assert result == []
        mock_graph.search_code.assert_awaited_once()
        call_kwargs = mock_graph.search_code.call_args.kwargs
        assert call_kwargs["entity_type"] == "constructor"

    async def test_clamps_limit_to_20(self, mock_ctx, mock_graph):
        """limit=50 is clamped to 20 when passed to graph client."""
        mock_graph.search_code.return_value = []
        await search_code(query="test", limit=50, ctx=mock_ctx)
        call_kwargs = mock_graph.search_code.call_args.kwargs
        assert call_kwargs["limit"] == 20

    async def test_invalid_code_mode_defaults_to_preview(self, mock_ctx, mock_graph):
        """Invalid code_mode falls back to 'preview'."""
        mock_graph.search_code.return_value = []
        await search_code(query="test", code_mode="bad_mode", ctx=mock_ctx)
        call_kwargs = mock_graph.search_code.call_args.kwargs
        assert call_kwargs["code_mode"] == "preview"

    async def test_returns_list_of_dicts(self, mock_ctx, mock_graph):
        """Transforms CodeEntity list into list of dicts."""
        mock_graph.search_code.return_value = [
            CodeEntity(
                name="authenticate",
                file_path="src/auth.py",
                repository="my-repo",
                entity_id="my-repo::src/auth.py::authenticate",
                line_start=10,
                line_end=25,
                code="def authenticate(): ...",
                signature="def authenticate()",
                entity_type="method",
                language="Python",
                return_type="bool",
                modifiers=["async"],
                stereotypes=["endpoint"],
                content_hash="deadbeef",
                properties={"visibility": "public"},
            ),
        ]
        result = await search_code(query="auth", ctx=mock_ctx)
        assert len(result) == 1
        assert result[0] == {
            "name": "authenticate",
            "file_path": "src/auth.py",
            "repository": "my-repo",
            "entity_id": "my-repo::src/auth.py::authenticate",
            "line_start": 10,
            "line_end": 25,
            "code": "def authenticate(): ...",
            "signature": "def authenticate()",
            "entity_type": "method",
            "language": "Python",
            "return_type": "bool",
            "modifiers": ["async"],
            "stereotypes": ["endpoint"],
            "content_hash": "deadbeef",
            "properties": {"visibility": "public"},
        }

    async def test_passes_language_and_stereotype_filters(self, mock_ctx, mock_graph):
        mock_graph.search_code.return_value = []

        await search_code(
            query="auth",
            language="Python",
            stereotype="endpoint",
            ctx=mock_ctx,
        )

        call_kwargs = mock_graph.search_code.call_args.kwargs
        assert call_kwargs["language"] == "Python"
        assert call_kwargs["stereotype"] == "endpoint"


# =============================================================================
# get_callers
# =============================================================================


class TestGetCallersTool:
    async def test_clamps_depth_to_3(self, mock_ctx, mock_graph):
        """depth=5 is clamped to 3 when passed to graph client."""
        mock_graph.get_callers.return_value = []
        await get_callers(method_name="foo", depth=5, ctx=mock_ctx)
        call_kwargs = mock_graph.get_callers.call_args.kwargs
        assert call_kwargs["depth"] == 3

    async def test_returns_list_of_dicts(self, mock_ctx, mock_graph):
        """Transforms CallGraphNode list into list of dicts."""
        mock_graph.get_callers.return_value = [
            CallGraphNode(
                name="handle_request",
                file_path="src/handler.py",
                repository="my-repo",
                signature="def handle_request(req)",
                line_start=42,
                entity_type="method",
                relationship_type="CALLS",
                truncated=True,
            ),
        ]
        result = await get_callers(method_name="process", ctx=mock_ctx)
        assert len(result) == 1
        assert result[0] == {
            "name": "handle_request",
            "file_path": "src/handler.py",
            "repository": "my-repo",
            "signature": "def handle_request(req)",
            "line_start": 42,
            "depth": 1,
            "entity_type": "method",
            "relationship_type": "CALLS",
            "truncated": True,
        }

    async def test_passes_limit_through_to_graph(self, mock_ctx, mock_graph):
        mock_graph.get_callers.return_value = []
        await get_callers(method_name="foo", limit=80, ctx=mock_ctx)
        call_kwargs = mock_graph.get_callers.call_args.kwargs
        assert call_kwargs["limit"] == 80


# =============================================================================
# get_callees
# =============================================================================


class TestGetCalleesTool:
    async def test_clamps_depth_to_3(self, mock_ctx, mock_graph):
        """depth=5 is clamped to 3 when passed to graph client."""
        mock_graph.get_callees.return_value = []
        await get_callees(method_name="foo", depth=5, ctx=mock_ctx)
        call_kwargs = mock_graph.get_callees.call_args.kwargs
        assert call_kwargs["depth"] == 3

    async def test_returns_list_of_dicts(self, mock_ctx, mock_graph):
        """Transforms CallGraphNode list into list of dicts."""
        mock_graph.get_callees.return_value = [
            CallGraphNode(
                name="query_db",
                file_path="src/db.py",
                repository="my-repo",
                signature="def query_db(sql)",
                line_start=15,
                depth=2,
                entity_type="reference",
                relationship_type="CALLS",
                truncated=True,
            ),
        ]
        result = await get_callees(method_name="process", ctx=mock_ctx)
        assert len(result) == 1
        assert result[0] == {
            "name": "query_db",
            "file_path": "src/db.py",
            "repository": "my-repo",
            "signature": "def query_db(sql)",
            "line_start": 15,
            "depth": 2,
            "entity_type": "reference",
            "relationship_type": "CALLS",
            "truncated": True,
        }

    async def test_passes_limit_through_to_graph(self, mock_ctx, mock_graph):
        mock_graph.get_callees.return_value = []
        await get_callees(method_name="foo", limit=80, ctx=mock_ctx)
        call_kwargs = mock_graph.get_callees.call_args.kwargs
        assert call_kwargs["limit"] == 80


# =============================================================================
# get_function_context
# =============================================================================


class TestGetFunctionContextTool:
    async def test_raises_on_not_found(self, mock_ctx, mock_graph):
        """ValueError when graph returns None."""
        mock_graph.get_function_context.return_value = None
        with pytest.raises(ValueError, match="Method 'missing' not found"):
            await get_function_context(method_name="missing", ctx=mock_ctx)

    async def test_returns_dict_with_callers_callees(self, mock_ctx, mock_graph):
        """Full response includes callers and callees as nested dicts."""
        caller = CallGraphNode(
            name="caller_fn",
            file_path="src/a.py",
            repository="repo",
            signature="def caller_fn()",
            entity_type="method",
            relationship_type="CALLS",
        )
        callee = CallGraphNode(
            name="callee_fn",
            file_path="src/b.py",
            repository="repo",
            signature="def callee_fn()",
            entity_type="hook",
            relationship_type="USES_HOOK",
        )
        mock_graph.get_function_context.return_value = FunctionContext(
            name="my_func",
            full_name="module.MyClass.my_func",
            file_path="src/module.py",
            repository="repo",
            code="def my_func(): pass",
            signature="def my_func()",
            docstring="Does stuff.",
            class_name="MyClass",
            callers=[caller],
            callees=[callee],
        )
        result = await get_function_context(method_name="my_func", ctx=mock_ctx)
        assert result["name"] == "my_func"
        assert result["full_name"] == "module.MyClass.my_func"
        assert result["class_name"] == "MyClass"
        assert len(result["callers"]) == 1
        assert result["callers"][0]["name"] == "caller_fn"
        assert result["callers"][0]["entity_type"] == "method"
        assert len(result["callees"]) == 1
        assert result["callees"][0]["name"] == "callee_fn"
        assert result["callees"][0]["entity_type"] == "hook"


# =============================================================================
# get_class_hierarchy
# =============================================================================


class TestGetClassHierarchyTool:
    async def test_raises_on_not_found(self, mock_ctx, mock_graph):
        """ValueError when graph returns None."""
        mock_graph.get_class_hierarchy.return_value = None
        with pytest.raises(ValueError, match="Class or interface 'Missing' not found"):
            await get_class_hierarchy(class_name="Missing", ctx=mock_ctx)

    async def test_response_includes_constructors(self, mock_ctx, mock_graph):
        """Response dict includes 'constructors' key."""
        mock_graph.get_class_hierarchy.return_value = ClassHierarchy(
            name="MyClass",
            full_name="com.example.MyClass",
            file_path="src/MyClass.java",
            repository="repo",
            is_interface=False,
            parents=["BaseClass"],
            children=["ChildClass"],
            interfaces=["Serializable"],
            implementors=[],
            methods=["doWork", "cleanup"],
            fields=["name", "id"],
            constructors=["MyClass"],
        )
        result = await get_class_hierarchy(class_name="MyClass", ctx=mock_ctx)
        assert "constructors" in result
        assert result["constructors"] == ["MyClass"]
        assert result["methods"] == ["doWork", "cleanup"]
        assert result["fields"] == ["name", "id"]


# =============================================================================
# list_repositories
# =============================================================================


class TestListRepositoriesTool:
    async def test_returns_graph_results(self, mock_ctx, mock_graph):
        """Returns the graph client result directly."""
        expected = [
            {"name": "repo-a", "entity_count": 100, "last_indexed_at": "2025-01-01"},
            {"name": "repo-b", "entity_count": 200, "last_indexed_at": "2025-01-02"},
        ]
        mock_graph.list_repositories.return_value = expected
        result = await list_repositories(ctx=mock_ctx)
        assert result == expected
        mock_graph.list_repositories.assert_awaited_once()


class TestGetRepositoryContextTool:
    async def test_raises_on_not_found(self, mock_ctx, mock_graph):
        mock_graph.get_repository_context.return_value = None
        with pytest.raises(ValueError, match="Repository 'missing' not found"):
            await get_repository_context(repository="missing", ctx=mock_ctx)

    async def test_returns_repository_context(self, mock_ctx, mock_graph):
        mock_graph.get_repository_context.return_value = RepositoryContext(
            name="repo-a",
            source="https://github.com/example/repo-a",
            entity_count=123,
            last_indexed_at="2026-03-12T10:00:00+00:00",
            last_commit_sha="abc123",
            total_files=20,
            total_classes=4,
            total_interfaces=1,
            total_methods=16,
            total_constructors=2,
            total_fields=9,
            total_packages=3,
            total_hooks=1,
            total_references=5,
            total_exports=7,
            languages=["Python", "TypeScript"],
            top_level_classes=["App"],
            entry_points=["App.main"],
        )

        result = await get_repository_context(repository="repo-a", ctx=mock_ctx)

        assert result == {
            "name": "repo-a",
            "source": "https://github.com/example/repo-a",
            "entity_count": 123,
            "last_indexed_at": "2026-03-12T10:00:00+00:00",
            "last_commit_sha": "abc123",
            "total_files": 20,
            "total_classes": 4,
            "total_interfaces": 1,
            "total_methods": 16,
            "total_constructors": 2,
            "total_fields": 9,
            "total_packages": 3,
            "total_hooks": 1,
            "total_references": 5,
            "total_exports": 7,
            "languages": ["Python", "TypeScript"],
            "top_level_classes": ["App"],
            "entry_points": ["App.main"],
        }


class TestGetPackageContextTool:
    async def test_raises_on_not_found(self, mock_ctx, mock_graph):
        mock_graph.get_package_context.return_value = None
        with pytest.raises(ValueError, match="Package 'src.services' not found"):
            await get_package_context(package_name="src.services", ctx=mock_ctx)

    async def test_returns_package_context(self, mock_ctx, mock_graph):
        mock_graph.get_package_context.return_value = PackageContext(
            name="src.services",
            repository="repo-a",
            package_id="repo-a::src.services",
            files=["src/services/user.py"],
            classes=["UserService"],
            interfaces=["IUserService"],
            methods=["build_service"],
            hooks=["useService"],
            references=["requests.get"],
            child_packages=["src.services.internal"],
        )

        result = await get_package_context(package_name="src.services", repository="repo-a", ctx=mock_ctx)

        assert result == {
            "name": "src.services",
            "repository": "repo-a",
            "package_id": "repo-a::src.services",
            "files": ["src/services/user.py"],
            "classes": ["UserService"],
            "interfaces": ["IUserService"],
            "methods": ["build_service"],
            "hooks": ["useService"],
            "references": ["requests.get"],
            "child_packages": ["src.services.internal"],
        }


# =============================================================================
# get_codebase_overview
# =============================================================================


class TestGetCodebaseOverviewTool:
    async def test_response_includes_total_constructors(self, mock_ctx, mock_graph):
        """Response includes 'total_constructors' key."""
        mock_graph.get_codebase_overview.return_value = CodebaseOverview(
            total_files=50,
            total_classes=20,
            total_interfaces=4,
            total_methods=150,
            total_constructors=25,
            total_fields=40,
            total_packages=6,
            total_hooks=2,
            total_references=9,
            total_exports=12,
            languages=["Python", "Java"],
            top_level_classes=["App", "Config"],
            entry_points=["App.main"],
        )
        result = await get_codebase_overview(ctx=mock_ctx)
        assert result["total_constructors"] == 25
        assert result["total_files"] == 50
        assert result["total_classes"] == 20
        assert result["total_interfaces"] == 4
        assert result["total_methods"] == 150
        assert result["total_fields"] == 40
        assert result["total_packages"] == 6
        assert result["total_hooks"] == 2
        assert result["total_references"] == 9
        assert result["total_exports"] == 12
        assert "packages" not in result

    async def test_packages_included_when_requested(self, mock_ctx, mock_graph):
        """packages key appears only when include_packages=True."""
        mock_graph.get_codebase_overview.return_value = CodebaseOverview(
            total_files=10,
            total_classes=5,
            total_methods=30,
            total_constructors=3,
            languages=["Python"],
            packages=["com.example", "com.example.util"],
            top_level_classes=["Main"],
            entry_points=["Main.main"],
        )
        result = await get_codebase_overview(ctx=mock_ctx, include_packages=True)
        assert "packages" in result
        assert result["packages"] == ["com.example", "com.example.util"]


# =============================================================================
# find_symbols
# =============================================================================


class TestFindSymbolsTool:
    async def test_validates_entity_types(self, mock_ctx):
        with pytest.raises(ValueError, match="Invalid entity_types"):
            await find_symbols(query="auth", entity_types=["bogus"], ctx=mock_ctx)

    async def test_returns_symbol_matches(self, mock_ctx, mock_graph):
        mock_graph.find_symbols.return_value = [
            CodeEntity(
                name="useState",
                file_path="src/App.tsx",
                repository="repo",
                entity_id="repo::src/App.tsx::useState",
                line_start=12,
                entity_type="hook",
                language="TypeScript",
                content_hash="deadbeef",
                properties={"symbol": "useState"},
            ),
        ]
        result = await find_symbols(query="useState", entity_types=["hook"], ctx=mock_ctx)
        assert result == [
            {
                "name": "useState",
                "file_path": "src/App.tsx",
                "repository": "repo",
                "entity_id": "repo::src/App.tsx::useState",
                "line_start": 12,
                "line_end": None,
                "code": None,
                "signature": None,
                "entity_type": "hook",
                "language": "TypeScript",
                "return_type": None,
                "modifiers": [],
                "stereotypes": [],
                "content_hash": "deadbeef",
                "properties": {"symbol": "useState"},
            },
        ]

    async def test_passes_language_and_stereotype_filters(self, mock_ctx, mock_graph):
        mock_graph.find_symbols.return_value = []

        await find_symbols(
            query="auth",
            language="Python",
            stereotype="test",
            ctx=mock_ctx,
        )

        call_kwargs = mock_graph.find_symbols.call_args.kwargs
        assert call_kwargs["language"] == "Python"
        assert call_kwargs["stereotype"] == "test"


# =============================================================================
# get_file_context
# =============================================================================


class TestGetFileContextTool:
    async def test_raises_on_not_found(self, mock_ctx, mock_graph):
        mock_graph.get_file_context.return_value = None
        with pytest.raises(ValueError, match="File 'missing.py' not found"):
            await get_file_context(file_path="missing.py", ctx=mock_ctx)

    async def test_returns_file_context(self, mock_ctx, mock_graph):
        mock_graph.get_file_context.return_value = FileContext(
            name="App.tsx",
            file_path="src/App.tsx",
            repository="repo",
            language="TypeScript",
            content_hash="deadbeef",
            packages=["src.components"],
            exports=[
                CodeEntity(
                    name="App",
                    file_path="src/App.tsx",
                    repository="repo",
                    entity_id="repo::src/App.tsx::App",
                    entity_type="class",
                    content_hash="deadbeef",
                    properties={"export_type": "default"},
                )
            ],
            classes=["App"],
            interfaces=["Props"],
            top_level_methods=["renderApp"],
            hooks=["useState"],
            constructors=["App"],
            fields=["title"],
            references=["React.Fragment"],
        )
        result = await get_file_context(file_path="App.tsx", ctx=mock_ctx)
        assert result["name"] == "App.tsx"
        assert result["language"] == "TypeScript"
        assert result["content_hash"] == "deadbeef"
        assert result["packages"] == ["src.components"]
        assert result["classes"] == ["App"]
        assert result["interfaces"] == ["Props"]
        assert result["top_level_methods"] == ["renderApp"]
        assert result["hooks"] == ["useState"]
        assert result["constructors"] == ["App"]
        assert result["fields"] == ["title"]
        assert result["references"] == ["React.Fragment"]
        assert result["exports"][0]["name"] == "App"
        assert result["exports"][0]["entity_id"] == "repo::src/App.tsx::App"
        assert result["exports"][0]["content_hash"] == "deadbeef"
        assert result["exports"][0]["properties"] == {"export_type": "default"}


# =============================================================================
# get_hook_usage
# =============================================================================


class TestGetHookUsageTool:
    async def test_returns_methods_using_hook(self, mock_ctx, mock_graph):
        mock_graph.get_hook_usage.return_value = [
            CallGraphNode(
                name="renderApp",
                file_path="src/App.tsx",
                repository="repo",
                signature="function renderApp()",
                line_start=15,
                entity_type="method",
                relationship_type="USES_HOOK",
                truncated=True,
            ),
        ]
        result = await get_hook_usage(hook_name="useState", ctx=mock_ctx)
        assert result == [
            {
                "name": "renderApp",
                "file_path": "src/App.tsx",
                "repository": "repo",
                "signature": "function renderApp()",
                "line_start": 15,
                "depth": 1,
                "entity_type": "method",
                "relationship_type": "USES_HOOK",
                "truncated": True,
            },
        ]

    async def test_passes_limit_language_and_stereotype_filters(self, mock_ctx, mock_graph):
        mock_graph.get_hook_usage.return_value = []

        await get_hook_usage(
            hook_name="useState",
            limit=80,
            language="TypeScript",
            stereotype="test",
            ctx=mock_ctx,
        )

        call_kwargs = mock_graph.get_hook_usage.call_args.kwargs
        assert call_kwargs["limit"] == 80
        assert call_kwargs["language"] == "TypeScript"
        assert call_kwargs["stereotype"] == "test"


# =============================================================================
# get_impact
# =============================================================================


class TestGetImpactTool:
    async def test_raises_on_not_found(self, mock_ctx, mock_graph):
        """ValueError when graph returns None."""
        mock_graph.get_impact.return_value = None
        with pytest.raises(ValueError, match="Method 'ghost' not found"):
            await get_impact(method_name="ghost", ctx=mock_ctx)

    async def test_returns_categorized_callers(self, mock_ctx, mock_graph):
        """Response includes categorized caller lists."""
        test_caller = CallGraphNode(
            name="test_process",
            file_path="tests/test_proc.py",
            repository="repo",
            depth=2,
            is_test=True,
        )
        endpoint_caller = CallGraphNode(
            name="handleRequest",
            file_path="src/api.py",
            repository="repo",
            depth=1,
            is_endpoint=True,
        )
        other_caller = CallGraphNode(
            name="helper",
            file_path="src/util.py",
            repository="repo",
            depth=3,
        )
        mock_graph.get_impact.return_value = ImpactResult(
            target_name="process",
            target_file="src/proc.py",
            target_repository="repo",
            total_callers=3,
            test_count=1,
            endpoint_count=1,
            affected_tests=[test_caller],
            affected_endpoints=[endpoint_caller],
            other_callers=[other_caller],
            truncated=False,
        )
        result = await get_impact(method_name="process", ctx=mock_ctx)
        assert result["target_name"] == "process"
        assert result["total_callers"] == 3
        assert result["test_count"] == 1
        assert result["endpoint_count"] == 1
        assert len(result["affected_tests"]) == 1
        assert result["affected_tests"][0]["name"] == "test_process"
        assert result["affected_tests"][0]["depth"] == 2
        assert len(result["affected_endpoints"]) == 1
        assert result["affected_endpoints"][0]["name"] == "handleRequest"
        assert len(result["other_callers"]) == 1
        assert result["other_callers"][0]["name"] == "helper"
        assert result["truncated"] is False
