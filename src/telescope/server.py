"""MCP Server for querying Constellation's code knowledge graph."""

import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from .graph_client import GraphClient

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Context
# =============================================================================


@dataclass
class AppContext:
    """Shared resources initialized at startup."""
    graph: GraphClient


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize and cleanup resources."""
    logger.info("Initializing Telescope server...")

    graph = GraphClient()
    await graph.connect()

    logger.info("Telescope server ready")

    try:
        yield AppContext(graph=graph)
    finally:
        logger.info("Shutting down Telescope server...")
        await graph.close()


# =============================================================================
# MCP Server
# =============================================================================


mcp = FastMCP(
    "Telescope",
    instructions="""Code knowledge graph for understanding and navigating codebases.

## When to Use This MCP

USE telescope when you need to:
- Find code by concept ("authentication logic", "payment processing")
- Understand who calls a function before changing it
- Analyze blast radius of a change (affected tests, endpoints)
- Explore class hierarchies and inheritance
- Get codebase overview and statistics
- Inspect file-level graph context, exports, constructors, fields, and hook usage
- Find exact symbols across files, fields, packages, hooks, and references
- Inspect repository metadata and package/namespace membership

DON'T use telescope when you need to:
- Read exact file contents (use Read tool instead)
- Make edits to files (use Edit tool instead)
- Search for exact strings (use Grep tool instead)

## Common Workflows

**Before modifying a function:**
1. get_function_context("methodName") - see callers, callees, code
2. get_impact("methodName", summary_only=True) - check blast radius

**Finding related code:**
1. search_code("what you're looking for")
2. get_callers or get_callees to explore relationships

**Understanding a codebase:**
1. list_repositories() - see what's indexed
2. get_repository_context("name") - get source metadata and aggregate stats
3. get_codebase_overview(repository="name") - get stats and entry points

## Tools

- list_repositories: See all indexed repos
- get_repository_context: Repository metadata plus aggregate graph stats
- search_code: Semantic search across embedded entities
- find_symbols: Exact/substring graph lookup across all persisted entity types
- get_package_context: Package or namespace membership and child packages
- get_callers/get_callees: Call relationships, references, and hooks
- get_function_context: Full context before modifying
- get_file_context: Packages, exports, constructors, fields, references, and hook usage for one file
- get_hook_usage: Find React/framework hook consumers
- get_impact: Blast radius analysis
- get_class_hierarchy: Inheritance relationships
- get_codebase_overview: High-level stats

## Limitations

- Languages: Java, Python, JavaScript/TypeScript, C# (indexed by Constellation)
- Call tracing works within a language, not across languages
""",
    lifespan=app_lifespan,
)


# =============================================================================
# Tools
# =============================================================================


# Valid entity types for search_code
VALID_ENTITY_TYPES = {"method", "class", "interface", "constructor"}
VALID_SYMBOL_ENTITY_TYPES = {
    "file",
    "package",
    "class",
    "interface",
    "method",
    "constructor",
    "field",
    "hook",
    "reference",
}


def _entity_to_dict(entity) -> dict:
    """Serialize a CodeEntity-style result for MCP responses."""
    return {
        "name": entity.name,
        "file_path": entity.file_path,
        "repository": entity.repository,
        "entity_id": entity.entity_id,
        "line_start": entity.line_start,
        "line_end": entity.line_end,
        "code": entity.code,
        "signature": entity.signature,
        "entity_type": entity.entity_type,
        "language": entity.language,
        "return_type": entity.return_type,
        "modifiers": entity.modifiers,
        "stereotypes": entity.stereotypes,
        "content_hash": entity.content_hash,
        "properties": entity.properties,
    }


def _call_node_to_dict(node) -> dict:
    """Serialize a CallGraphNode-style result for MCP responses."""
    return {
        "name": node.name,
        "file_path": node.file_path,
        "repository": node.repository,
        "signature": node.signature,
        "line_start": node.line_start,
        "depth": node.depth,
        "entity_type": node.entity_type,
        "relationship_type": node.relationship_type,
        "truncated": node.truncated,
    }


@mcp.tool()
async def search_code(
    query: str,
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
    entity_type: str | None = None,
    file_pattern: str | None = None,
    language: str | None = None,
    stereotype: str | None = None,
    limit: int = 10,
    code_mode: str = "preview",
) -> list[dict]:
    """
    Search for code using semantic similarity.

    Use this to find functions, classes, or methods that match a natural
    language description. Better than grep for conceptual searches like
    "authentication logic" or "database connection handling".

    Args:
        query: Natural language description of what you're looking for
        repository: Filter by repository name (e.g., "consumer-operations", "whats-the-update")
        entity_type: Filter by entity type. Valid values:
            - "method": Functions and methods (most common)
            - "class": Class definitions
            - "interface": Interface definitions (Java)
            - "constructor": Constructor definitions
            If not specified, searches all entity types.
        file_pattern: Filter by file path pattern, e.g., "*/api/*" (optional)
        language: Filter by persisted language, e.g. "Python" or "TypeScript"
        stereotype: Filter by persisted stereotype, e.g. "endpoint" or "test"
        limit: Maximum results to return (default 10)
        code_mode: How much code to include in results (default "preview"):
            - "none": No code, just metadata (smallest response)
            - "signature": Only method/class signature
            - "preview": First 10 lines of code (default, good balance)
            - "full": Complete source code (use sparingly)

    Returns:
        List of matching code entities with source code and locations.
        Identifier-like queries automatically blend in exact graph symbol matches.
    """
    graph = ctx.request_context.lifespan_context.graph
    limit = min(limit, 20)

    # Validate code_mode
    valid_modes = {"none", "signature", "preview", "full"}
    if code_mode not in valid_modes:
        code_mode = "preview"

    # Validate entity_type
    if entity_type is not None and entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(
            f"Invalid entity_type '{entity_type}'. "
            f"Valid values: {', '.join(sorted(VALID_ENTITY_TYPES))}"
        )

    results = await graph.search_code(
        query=query,
        limit=limit,
        entity_type=entity_type,
        file_pattern=file_pattern,
        repository=repository,
        language=language,
        stereotype=stereotype,
        code_mode=code_mode,
    )

    return [
        _entity_to_dict(r)
        for r in results
    ]


@mcp.tool()
async def get_callers(
    method_name: str,
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
    file_path: str | None = None,
    depth: int = 1,
    limit: int = 50,
) -> list[dict]:
    """
    Find all functions that call the specified method.

    Use this to understand the impact of changing a function - who depends on it?

    Args:
        method_name: Name of the method to find callers for
        repository: Filter by repository name (e.g., "consumer-operations")
        file_path: Disambiguate if multiple methods have the same name
        depth: How many levels up to traverse (default 1, max 3)
        limit: Maximum callers to return (default 50, max 200)

    Returns:
        List of functions that call this method. Each result includes
        a `truncated` flag when additional callers exist beyond the limit.
    """
    graph = ctx.request_context.lifespan_context.graph

    results = await graph.get_callers(
        method_name=method_name,
        file_path=file_path,
        repository=repository,
        depth=min(depth, 3),
        limit=min(limit, 200),
    )

    return [_call_node_to_dict(r) for r in results]


@mcp.tool()
async def get_callees(
    method_name: str,
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
    file_path: str | None = None,
    depth: int = 1,
    limit: int = 50,
) -> list[dict]:
    """
    Find all graph targets that the specified method calls or uses.

    Use this to understand what a method depends on before modifying it.

    Args:
        method_name: Name of the method to analyze
        repository: Filter by repository name (e.g., "consumer-operations")
        file_path: Disambiguate if multiple methods have the same name
        depth: How many levels down to traverse (default 1, max 3)
        limit: Maximum graph targets to return (default 50, max 200)

    Returns:
        List of methods, constructors, references, and hooks reached from this method.
        Each result includes a `truncated` flag when additional targets exist.
    """
    graph = ctx.request_context.lifespan_context.graph

    results = await graph.get_callees(
        method_name=method_name,
        file_path=file_path,
        repository=repository,
        depth=min(depth, 3),
        limit=min(limit, 200),
    )

    return [_call_node_to_dict(r) for r in results]


@mcp.tool()
async def get_function_context(
    method_name: str,
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
    file_path: str | None = None,
) -> dict:
    """
    Get comprehensive context for a function before modifying it.

    Returns the function's code, callers, callees, and parent class (if method).
    This is the go-to tool before making changes to understand full impact.

    Args:
        method_name: Name of the method
        repository: Filter by repository name (e.g., "consumer-operations")
        file_path: Disambiguate if multiple methods have the same name

    Returns:
        Full context including code and relationships
    """
    graph = ctx.request_context.lifespan_context.graph

    result = await graph.get_function_context(
        method_name=method_name,
        file_path=file_path,
        repository=repository,
    )

    if not result:
        raise ValueError(f"Method '{method_name}' not found in graph")

    return {
        "name": result.name,
        "full_name": result.full_name,
        "file_path": result.file_path,
        "repository": result.repository,
        "code": result.code,
        "signature": result.signature,
        "docstring": result.docstring,
        "class_name": result.class_name,
        "callers": [_call_node_to_dict(c) for c in result.callers],
        "callees": [_call_node_to_dict(c) for c in result.callees],
    }


@mcp.tool()
async def get_class_hierarchy(
    class_name: str,
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
    file_path: str | None = None,
) -> dict:
    """
    Get inheritance hierarchy for a class or interface.

    Use before modifying a class to understand what it inherits from
    and what inherits from it. Also works for interfaces to see implementors.

    Args:
        class_name: Name of the class or interface
        repository: Filter by repository name (e.g., "consumer-operations")
        file_path: Disambiguate if multiple classes have the same name

    Returns:
        Class hierarchy including parents, children, interfaces, implementors,
        methods, fields, and constructors
    """
    graph = ctx.request_context.lifespan_context.graph

    result = await graph.get_class_hierarchy(
        class_name=class_name,
        file_path=file_path,
        repository=repository,
    )

    if not result:
        raise ValueError(f"Class or interface '{class_name}' not found in graph")

    return {
        "name": result.name,
        "full_name": result.full_name,
        "file_path": result.file_path,
        "repository": result.repository,
        "is_interface": result.is_interface,
        "parents": result.parents,
        "children": result.children,
        "interfaces": result.interfaces,
        "implementors": result.implementors,
        "methods": result.methods,
        "fields": result.fields,
        "constructors": result.constructors,
    }


@mcp.tool()
async def list_repositories(
    ctx: Context[ServerSession, AppContext],
) -> list[dict]:
    """
    List all indexed repositories.

    Returns a list of repositories with their entity counts.
    Use this to see what codebases are available to query.

    Returns:
        List of repositories with name and entity count
    """
    graph = ctx.request_context.lifespan_context.graph
    return await graph.list_repositories()


@mcp.tool()
async def get_repository_context(
    repository: str,
    ctx: Context[ServerSession, AppContext],
) -> dict:
    """
    Get repository metadata plus aggregate graph statistics for one indexed repository.
    """
    graph = ctx.request_context.lifespan_context.graph
    result = await graph.get_repository_context(repository)
    if not result:
        raise ValueError(f"Repository '{repository}' not found in graph")

    return {
        "name": result.name,
        "source": result.source,
        "entity_count": result.entity_count,
        "last_indexed_at": result.last_indexed_at,
        "last_commit_sha": result.last_commit_sha,
        "total_files": result.total_files,
        "total_classes": result.total_classes,
        "total_interfaces": result.total_interfaces,
        "total_methods": result.total_methods,
        "total_constructors": result.total_constructors,
        "total_fields": result.total_fields,
        "total_packages": result.total_packages,
        "total_hooks": result.total_hooks,
        "total_references": result.total_references,
        "total_exports": result.total_exports,
        "languages": result.languages,
        "top_level_classes": result.top_level_classes,
        "entry_points": result.entry_points,
    }


@mcp.tool()
async def get_codebase_overview(
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
    include_packages: bool = False,
) -> dict:
    """
    Get a high-level overview of the codebase structure.

    Use when first exploring a codebase or needing to understand architecture.

    Args:
        repository: Filter by repository name (e.g., "consumer-operations", "whats-the-update")
        include_packages: Include full package list (default False to save tokens).
            Only set to True if you specifically need to see all packages.

    Returns:
        Overview with statistics, key entry points, and important classes
    """
    graph = ctx.request_context.lifespan_context.graph

    result = await graph.get_codebase_overview(
        repository=repository,
        include_packages=include_packages,
    )

    response = {
        "total_files": result.total_files,
        "total_classes": result.total_classes,
        "total_interfaces": result.total_interfaces,
        "total_methods": result.total_methods,
        "total_constructors": result.total_constructors,
        "total_fields": result.total_fields,
        "total_packages": result.total_packages,
        "total_hooks": result.total_hooks,
        "total_references": result.total_references,
        "total_exports": result.total_exports,
        "languages": result.languages,
        "top_level_classes": result.top_level_classes,
        "entry_points": result.entry_points,
    }

    # Only include packages if explicitly requested
    if include_packages:
        response["packages"] = result.packages

    return response


@mcp.tool()
async def find_symbols(
    query: str,
    ctx: Context[ServerSession, AppContext],
    entity_types: list[str] | None = None,
    repository: str | None = None,
    file_pattern: str | None = None,
    language: str | None = None,
    stereotype: str | None = None,
    limit: int = 20,
    exact: bool = False,
) -> list[dict]:
    """
    Find exact or substring symbol matches across the full Constellation graph.

    Use this when you know the symbol or path fragment you want and need more than
    vector search can provide, including fields, packages, hooks, references, and files.
    You can also filter by language and stereotype.
    """
    if entity_types:
        invalid = sorted(set(entity_types) - VALID_SYMBOL_ENTITY_TYPES)
        if invalid:
            raise ValueError(
                f"Invalid entity_types {invalid}. "
                f"Valid values: {', '.join(sorted(VALID_SYMBOL_ENTITY_TYPES))}"
            )

    graph = ctx.request_context.lifespan_context.graph
    results = await graph.find_symbols(
        query=query,
        entity_types=entity_types,
        repository=repository,
        file_pattern=file_pattern,
        language=language,
        stereotype=stereotype,
        limit=min(limit, 50),
        exact=exact,
    )
    return [
        _entity_to_dict(r)
        for r in results
    ]


@mcp.tool()
async def get_package_context(
    package_name: str,
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
) -> dict:
    """
    Get package or namespace membership, including child packages and symbol counts.
    """
    graph = ctx.request_context.lifespan_context.graph
    result = await graph.get_package_context(package_name=package_name, repository=repository)
    if not result:
        raise ValueError(f"Package '{package_name}' not found in graph")

    return {
        "name": result.name,
        "repository": result.repository,
        "package_id": result.package_id,
        "files": result.files,
        "classes": result.classes,
        "interfaces": result.interfaces,
        "methods": result.methods,
        "hooks": result.hooks,
        "references": result.references,
        "child_packages": result.child_packages,
    }


@mcp.tool()
async def get_file_context(
    file_path: str,
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
) -> dict:
    """
    Get graph context for one file: packages, exports, top-level entities, and hook usage.
    """
    graph = ctx.request_context.lifespan_context.graph
    result = await graph.get_file_context(
        file_path=file_path,
        repository=repository,
    )
    if not result:
        raise ValueError(f"File '{file_path}' not found in graph")

    return {
        "name": result.name,
        "file_path": result.file_path,
        "repository": result.repository,
        "language": result.language,
        "content_hash": result.content_hash,
        "packages": result.packages,
        "exports": [_entity_to_dict(export) for export in result.exports],
        "classes": result.classes,
        "interfaces": result.interfaces,
        "top_level_methods": result.top_level_methods,
        "hooks": result.hooks,
        "constructors": result.constructors,
        "fields": result.fields,
        "references": result.references,
    }


@mcp.tool()
async def get_hook_usage(
    hook_name: str,
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
    file_pattern: str | None = None,
    language: str | None = None,
    stereotype: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Find methods and constructors that use a materialized Hook node.
    Results include a `truncated` flag when additional users exist beyond the limit.
    """
    graph = ctx.request_context.lifespan_context.graph
    results = await graph.get_hook_usage(
        hook_name=hook_name,
        repository=repository,
        file_pattern=file_pattern,
        language=language,
        stereotype=stereotype,
        limit=min(limit, 200),
    )
    return [_call_node_to_dict(r) for r in results]


@mcp.tool()
async def get_impact(
    method_name: str,
    ctx: Context[ServerSession, AppContext],
    repository: str | None = None,
    file_path: str | None = None,
    depth: int = 10,
    summary_only: bool = False,
    limit: int | None = None,
) -> dict:
    """
    Analyze the blast radius of changing a method.

    Shows all code that would be affected if you modify this method:
    transitive callers, affected tests, and affected API endpoints.

    Use this BEFORE making changes to understand full impact.

    Args:
        method_name: Name of the method to analyze
        repository: Filter by repository name
        file_path: Disambiguate if multiple methods have same name
        depth: Max call chain depth (default 10, use higher for deeper analysis)
        summary_only: If True, return only counts without caller details (fast overview)
        limit: Max callers per category (tests, endpoints, others). Use for large methods.

    Returns:
        Impact analysis with categorized callers and summary counts

    Examples:
        # Quick overview - just counts (small output)
        get_impact("extractPartsRecursive", summary_only=True)

        # Limited details - 5 per category
        get_impact("extractPartsRecursive", limit=5)

        # Full details (default, use for small methods)
        get_impact("validateUser")
    """
    graph = ctx.request_context.lifespan_context.graph

    result = await graph.get_impact(
        method_name=method_name,
        file_path=file_path,
        repository=repository,
        depth=depth,
        summary_only=summary_only,
        limit=limit,
    )

    if not result:
        raise ValueError(f"Method '{method_name}' not found in graph")

    return {
        "target_name": result.target_name,
        "target_file": result.target_file,
        "target_repository": result.target_repository,
        "total_callers": result.total_callers,
        "test_count": result.test_count,
        "endpoint_count": result.endpoint_count,
        "affected_tests": [
            {"name": c.name, "file_path": c.file_path, "repository": c.repository, "depth": c.depth}
            for c in result.affected_tests
        ],
        "affected_endpoints": [
            {"name": c.name, "file_path": c.file_path, "repository": c.repository, "depth": c.depth}
            for c in result.affected_endpoints
        ],
        "other_callers": [
            {"name": c.name, "file_path": c.file_path, "repository": c.repository, "depth": c.depth}
            for c in result.other_callers
        ],
        "truncated": result.truncated,
    }


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
