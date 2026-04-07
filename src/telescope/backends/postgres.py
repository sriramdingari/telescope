"""PostgreSQL implementation of ReadBackend using asyncpg."""

from __future__ import annotations

import logging
from typing import Any

import asyncpg
from openai import AsyncOpenAI

from telescope.backends.base import ReadBackend
from telescope.models import (
    CallGraphNode, ClassHierarchy, CodebaseOverview, CodeEntity,
    FileContext, FunctionContext, ImpactResult, PackageContext, RepositoryContext,
)

logger = logging.getLogger(__name__)


class PostgresReadBackend(ReadBackend):
    """PostgreSQL + pgvector implementation of ReadBackend."""

    def __init__(
        self,
        dsn: str,
        openai_api_key: str,
        openai_base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
    ) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._openai = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url or None,
        )

    async def connect(self) -> None:
        from pgvector.asyncpg import register_vector
        # register_vector is called for every new connection in the pool
        # so asyncpg knows how to encode/decode the vector type for search_code
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=1, max_size=5, init=register_vector
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresReadBackend: call connect() before using the backend")
        return self._pool

    # ── Private helpers ──────────────────────────────────────────────────

    async def _get_embedding(self, text: str) -> list[float]:
        response = await self._openai.embeddings.create(
            model=self._embedding_model,
            input=text,
            dimensions=self._embedding_dimensions,
        )
        return response.data[0].embedding

    async def _resolve_symbol(
        self,
        name: str,
        *,
        repository: str | None,
        file_path: str | None,
        entity_id: str | None,
        types: list[str],
    ) -> dict | None:
        """Resolve a symbol name to a unique row. Raises ValueError if ambiguous."""
        pool = self._require_pool()
        if entity_id:
            row = await pool.fetchrow(
                "SELECT * FROM code_symbols WHERE id = $1", entity_id
            )
            if row:
                return dict(row)

        conditions = ["symbol_name = $1", "symbol_type = ANY($2)"]
        params: list[Any] = [name, types]
        if repository:
            conditions.append(f"repository = ${len(params) + 1}")
            params.append(repository)
        if file_path:
            conditions.append(f"file_path ILIKE ${len(params) + 1}")
            params.append(f"%{file_path}")

        rows = await pool.fetch(
            f"SELECT * FROM code_symbols WHERE {' AND '.join(conditions)} LIMIT 2",
            *params,
        )
        if not rows:
            return None
        if len(rows) > 1:
            ids = [r["id"] for r in rows]
            raise ValueError(
                f"Ambiguous symbol {name!r} — matches: {ids}. "
                f"Provide file_path or entity_id to disambiguate."
            )
        return dict(rows[0])

    def _row_to_code_entity(self, row: dict, score: float = 0.0) -> CodeEntity:
        return CodeEntity(
            name=row["symbol_name"],
            file_path=row["file_path"],
            repository=row.get("repository"),
            entity_id=row["id"],
            line_start=row.get("line_start"),
            line_end=row.get("line_end"),
            code=row.get("code"),
            signature=row.get("signature"),
            docstring=row.get("docstring"),
            score=score,
            entity_type=str(row.get("symbol_type", "method")).lower(),
            language=row.get("language"),
            return_type=row.get("return_type"),
            modifiers=list(row.get("modifiers") or []),
            stereotypes=list(row.get("stereotypes") or []),
            content_hash=row.get("content_hash"),
            properties=dict(row.get("properties") or {}),
        )

    def _row_to_call_graph_node(self, row: dict, rel_type: str = "CALLS") -> CallGraphNode:
        return CallGraphNode(
            name=row["symbol_name"],
            file_path=row["file_path"],
            repository=row.get("repository"),
            signature=row.get("signature"),
            line_start=row.get("line_start"),
            depth=row.get("depth", 1),
            is_test=bool(row.get("is_test")),
            is_endpoint=bool(row.get("is_endpoint")),
            entity_type=str(row.get("symbol_type", "method")).lower(),
            relationship_type=rel_type,
        )

    @staticmethod
    def _normalize_repo_row(row: dict) -> dict:
        """Normalize Postgres row to match the API's expectations."""
        result = dict(row)
        if "commit_sha" in result:
            result["last_commit_sha"] = result.pop("commit_sha")
        if "last_indexed_at" in result and result["last_indexed_at"] is not None:
            result["last_indexed_at"] = result["last_indexed_at"].isoformat() if hasattr(result["last_indexed_at"], "isoformat") else str(result["last_indexed_at"])
        if "created_at" in result and result["created_at"] is not None:
            result["created_at"] = result["created_at"].isoformat() if hasattr(result["created_at"], "isoformat") else str(result["created_at"])
        return result

    # ── Semantic search ──────────────────────────────────────────────────

    async def search_code(
        self, query: str, *,
        limit: int = 10,
        entity_type: str | None = None,
        file_pattern: str | None = None,
        repository: str | None = None,
        code_mode: str = "preview",
        language: str | None = None,
        stereotype: str | None = None,
    ) -> list[CodeEntity]:
        pool = self._require_pool()
        embedding = await self._get_embedding(query)
        conditions = []
        params: list[Any] = [embedding, limit]
        if repository:
            params.append(repository)
            conditions.append(f"s.repository = ${len(params)}")
        if entity_type:
            params.append(entity_type.capitalize())
            conditions.append(f"s.symbol_type = ${len(params)}")
        if language:
            params.append(language)
            conditions.append(f"s.language = ${len(params)}")
        if stereotype:
            params.append(stereotype)
            conditions.append(f"${len(params)} = ANY(s.stereotypes)")
        if file_pattern:
            params.append(f"%{file_pattern}%")
            conditions.append(f"s.file_path ILIKE ${len(params)}")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await pool.fetch(f"""
            SELECT s.*, 1 - (e.embedding <=> $1::vector) AS score
            FROM code_embeddings e
            JOIN code_symbols s ON s.id = e.symbol_id
            {where}
            ORDER BY e.embedding <=> $1::vector
            LIMIT $2
        """, *params)

        entities = [self._row_to_code_entity(dict(r), score=r["score"]) for r in rows]
        if code_mode == "none":
            for e in entities:
                e.code = None
        elif code_mode == "signature":
            for e in entities:
                e.code = e.signature
        elif code_mode == "preview":
            for e in entities:
                if e.code:
                    e.code = "\n".join(e.code.splitlines()[:10])
        return entities

    # ── Symbol lookup ────────────────────────────────────────────────────

    async def find_symbols(
        self, query: str, *,
        entity_types: list[str] | None = None,
        repository: str | None = None,
        file_pattern: str | None = None,
        limit: int = 20,
        exact: bool = False,
        language: str | None = None,
        stereotype: str | None = None,
    ) -> list[CodeEntity]:
        pool = self._require_pool()
        conditions = []
        params: list[Any] = []

        if exact:
            params.append(query)
            conditions.append(f"symbol_name = ${len(params)}")
        else:
            params.append(f"%{query}%")
            conditions.append(f"symbol_name ILIKE ${len(params)}")

        if entity_types:
            labels = [t.capitalize() for t in entity_types]
            params.append(labels)
            conditions.append(f"symbol_type = ANY(${len(params)})")
        if repository:
            params.append(repository)
            conditions.append(f"repository = ${len(params)}")
        if language:
            params.append(language)
            conditions.append(f"language = ${len(params)}")
        if stereotype:
            params.append(stereotype)
            conditions.append(f"${len(params)} = ANY(stereotypes)")
        if file_pattern:
            params.append(f"%{file_pattern}%")
            conditions.append(f"file_path ILIKE ${len(params)}")

        params.append(limit)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await pool.fetch(
            f"SELECT * FROM code_symbols {where} LIMIT ${len(params)}",
            *params,
        )
        return [self._row_to_code_entity(dict(r)) for r in rows]

    # ── Call graph ───────────────────────────────────────────────────────

    async def get_callers(
        self, method_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
        entity_id: str | None = None,
        depth: int = 1,
        limit: int = 50,
    ) -> list[CallGraphNode]:
        pool = self._require_pool()
        symbol = await self._resolve_symbol(
            method_name, repository=repository, file_path=file_path,
            entity_id=entity_id, types=["Method", "Constructor"],
        )
        if not symbol:
            return []

        rows = await pool.fetch("""
            WITH RECURSIVE callers AS (
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, 1 AS depth
                FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                WHERE r.target_symbol_id = $1 AND r.ref_type = 'CALLS'
                UNION
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, c.depth + 1
                FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                JOIN callers c ON c.id = r.target_symbol_id
                WHERE c.depth < $2
            )
            SELECT DISTINCT ON (id) * FROM callers ORDER BY id, depth
            LIMIT $3
        """, symbol["id"], depth, limit)

        return [self._row_to_call_graph_node(dict(r)) for r in rows]

    async def get_callees(
        self, method_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
        entity_id: str | None = None,
        depth: int = 1,
        limit: int = 50,
    ) -> list[CallGraphNode]:
        pool = self._require_pool()
        symbol = await self._resolve_symbol(
            method_name, repository=repository, file_path=file_path,
            entity_id=entity_id, types=["Method", "Constructor"],
        )
        if not symbol:
            return []

        call_rows = await pool.fetch("""
            WITH RECURSIVE callees AS (
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, 1 AS depth
                FROM code_references r
                JOIN code_symbols s ON s.id = r.target_symbol_id
                WHERE r.source_symbol_id = $1 AND r.ref_type = 'CALLS'
                UNION
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, c.depth + 1
                FROM code_references r
                JOIN code_symbols s ON s.id = r.target_symbol_id
                JOIN callees c ON c.id = r.source_symbol_id
                WHERE c.depth < $2
            )
            SELECT DISTINCT ON (id) * FROM callees ORDER BY id, depth
            LIMIT $3
        """, symbol["id"], depth, limit)

        hook_rows = await pool.fetch("""
            SELECT s.id, s.symbol_name, s.file_path, s.repository,
                   s.signature, s.line_start, s.symbol_type,
                   s.is_test, s.is_endpoint, 1 AS depth
            FROM code_references r
            JOIN code_symbols s ON s.id = r.target_symbol_id
            WHERE r.source_symbol_id = $1 AND r.ref_type = 'USES_HOOK'
        """, symbol["id"])

        results = [self._row_to_call_graph_node(dict(r)) for r in call_rows]
        results += [self._row_to_call_graph_node(dict(r), rel_type="USES_HOOK") for r in hook_rows]
        return results[:limit]

    # ── Full context ─────────────────────────────────────────────────────

    async def get_function_context(
        self, method_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
    ) -> FunctionContext | None:
        symbol = await self._resolve_symbol(
            method_name, repository=repository, file_path=file_path,
            entity_id=None, types=["Method", "Constructor"],
        )
        if not symbol:
            return None

        callers = await self.get_callers(method_name, repository=repository,
                                         file_path=file_path, depth=1)
        callees = await self.get_callees(method_name, repository=repository,
                                          file_path=file_path, depth=1)

        return FunctionContext(
            name=symbol["symbol_name"],
            full_name=symbol["id"],
            file_path=symbol["file_path"],
            repository=symbol.get("repository"),
            code=symbol.get("code"),
            signature=symbol.get("signature"),
            docstring=symbol.get("docstring"),
            class_name=None,
            callers=callers,
            callees=callees,
        )

    # ── Hierarchy ────────────────────────────────────────────────────────

    async def get_class_hierarchy(
        self, class_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
    ) -> ClassHierarchy | None:
        pool = self._require_pool()
        cls = await self._resolve_symbol(
            class_name, repository=repository, file_path=file_path,
            entity_id=None, types=["Class", "Interface"],
        )
        if not cls:
            return None

        parents = await pool.fetch("""
            SELECT s.symbol_name, s.file_path, s.symbol_type
            FROM code_references r JOIN code_symbols s ON s.id = r.target_symbol_id
            WHERE r.source_symbol_id = $1 AND r.ref_type IN ('EXTENDS', 'IMPLEMENTS')
        """, cls["id"])

        children = await pool.fetch("""
            SELECT s.symbol_name, s.file_path
            FROM code_references r JOIN code_symbols s ON s.id = r.source_symbol_id
            WHERE r.target_symbol_id = $1 AND r.ref_type = 'EXTENDS'
        """, cls["id"])

        implementors = await pool.fetch("""
            SELECT s.symbol_name, s.file_path
            FROM code_references r JOIN code_symbols s ON s.id = r.source_symbol_id
            WHERE r.target_symbol_id = $1 AND r.ref_type = 'IMPLEMENTS'
        """, cls["id"])

        members = await pool.fetch("""
            SELECT s.symbol_name, s.symbol_type
            FROM code_references r JOIN code_symbols s ON s.id = r.target_symbol_id
            WHERE r.source_symbol_id = $1
              AND r.ref_type IN ('HAS_METHOD', 'HAS_FIELD', 'HAS_CONSTRUCTOR')
        """, cls["id"])

        return ClassHierarchy(
            name=cls["symbol_name"],
            full_name=cls["id"],
            file_path=cls["file_path"],
            repository=cls.get("repository"),
            is_interface=cls["symbol_type"] == "Interface",
            parents=[r["symbol_name"] for r in parents if r["symbol_type"] != "Interface"],
            children=[r["symbol_name"] for r in children],
            interfaces=[r["symbol_name"] for r in parents if r["symbol_type"] == "Interface"],
            implementors=[r["symbol_name"] for r in implementors],
            methods=[r["symbol_name"] for r in members if r["symbol_type"] == "Method"],
            fields=[r["symbol_name"] for r in members if r["symbol_type"] == "Field"],
            constructors=[r["symbol_name"] for r in members if r["symbol_type"] == "Constructor"],
        )

    # ── Context queries ──────────────────────────────────────────────────

    async def get_package_context(
        self, package_name: str, *,
        repository: str | None = None,
    ) -> PackageContext | None:
        pool = self._require_pool()
        pkg = await self._resolve_symbol(
            package_name, repository=repository, file_path=None,
            entity_id=None, types=["Package"],
        )
        if not pkg:
            return None

        members = await pool.fetch("""
            SELECT s.symbol_name, s.symbol_type, s.file_path
            FROM code_references r JOIN code_symbols s ON s.id = r.source_symbol_id
            WHERE r.target_symbol_id = $1 AND r.ref_type = 'IN_PACKAGE'
        """, pkg["id"])

        child_pkgs = await pool.fetch("""
            SELECT s.symbol_name FROM code_references r
            JOIN code_symbols s ON s.id = r.source_symbol_id
            WHERE r.target_symbol_id = $1 AND r.ref_type = 'CONTAINS'
              AND s.symbol_type = 'Package'
        """, pkg["id"])

        return PackageContext(
            name=pkg["symbol_name"],
            repository=pkg.get("repository"),
            package_id=pkg["id"],
            files=[r["symbol_name"] for r in members if r["symbol_type"] == "File"],
            classes=[r["symbol_name"] for r in members if r["symbol_type"] == "Class"],
            interfaces=[r["symbol_name"] for r in members if r["symbol_type"] == "Interface"],
            methods=[r["symbol_name"] for r in members if r["symbol_type"] == "Method"],
            hooks=[],
            references=[],
            child_packages=[r["symbol_name"] for r in child_pkgs],
        )

    async def get_file_context(
        self, file_path: str, *,
        repository: str | None = None,
    ) -> FileContext | None:
        pool = self._require_pool()
        # Query on file_path column directly — _resolve_symbol matches symbol_name
        # Use LIMIT 2 + ValueError on ambiguity, consistent with _resolve_symbol's contract
        conditions = ["file_path = $1", "symbol_type = 'File'"]
        params: list[Any] = [file_path]
        if repository:
            params.append(repository)
            conditions.append(f"repository = ${len(params)}")
        rows = await pool.fetch(
            f"SELECT * FROM code_symbols WHERE {' AND '.join(conditions)} LIMIT 2",
            *params,
        )
        if not rows:
            return None
        if len(rows) > 1:
            repos = [r["repository"] for r in rows]
            raise ValueError(
                f"Ambiguous file path {file_path!r} — found in repos: {repos}. "
                f"Provide repository= to disambiguate."
            )
        file_sym = dict(rows[0])

        contained = await pool.fetch("""
            SELECT s.symbol_name, s.symbol_type, s.line_start
            FROM code_references r JOIN code_symbols s ON s.id = r.target_symbol_id
            WHERE r.source_symbol_id = $1 AND r.ref_type = 'CONTAINS'
        """, file_sym["id"])

        exports = await pool.fetch("""
            SELECT s.symbol_name FROM code_references r
            JOIN code_symbols s ON s.id = r.target_symbol_id
            WHERE r.source_symbol_id = $1 AND r.ref_type = 'EXPORTS'
        """, file_sym["id"])

        # USES_HOOK relationships are on Method/Constructor nodes, not File nodes.
        # Traverse: File --CONTAINS--> Method/Constructor --USES_HOOK--> Hook
        hook_usages = await pool.fetch("""
            SELECT DISTINCT h.symbol_name
            FROM code_references r1
            JOIN code_symbols method ON method.id = r1.target_symbol_id
            JOIN code_references r2 ON r2.source_symbol_id = method.id
            JOIN code_symbols h ON h.id = r2.target_symbol_id
            WHERE r1.source_symbol_id = $1
              AND r1.ref_type = 'CONTAINS'
              AND method.symbol_type IN ('Method', 'Constructor')
              AND r2.ref_type = 'USES_HOOK'
        """, file_sym["id"])

        return FileContext(
            name=file_sym["symbol_name"],
            file_path=file_sym["file_path"],
            repository=file_sym.get("repository"),
            language=file_sym.get("language"),
            content_hash=file_sym.get("content_hash"),
            packages=[],
            exports=[],
            classes=[r["symbol_name"] for r in contained if r["symbol_type"] == "Class"],
            interfaces=[r["symbol_name"] for r in contained if r["symbol_type"] == "Interface"],
            top_level_methods=[r["symbol_name"] for r in contained if r["symbol_type"] == "Method"],
            hooks=[r["symbol_name"] for r in hook_usages],
            constructors=[r["symbol_name"] for r in contained if r["symbol_type"] == "Constructor"],
            fields=[r["symbol_name"] for r in contained if r["symbol_type"] == "Field"],
            references=[],
        )

    async def get_hook_usage(
        self, hook_name: str, *,
        repository: str | None = None,
        file_pattern: str | None = None,
        language: str | None = None,
        stereotype: str | None = None,
        limit: int = 50,
    ) -> list[CallGraphNode]:
        pool = self._require_pool()
        conditions = ["h.symbol_name = $1", "h.symbol_type = 'Hook'", "r.ref_type = 'USES_HOOK'"]
        params: list[Any] = [hook_name]
        if repository:
            params.append(repository)
            conditions.append(f"s.repository = ${len(params)}")
        if language:
            params.append(language)
            conditions.append(f"s.language = ${len(params)}")
        if file_pattern:
            params.append(f"%{file_pattern}%")
            conditions.append(f"s.file_path ILIKE ${len(params)}")
        if stereotype:
            params.append(stereotype)
            conditions.append(f"${len(params)} = ANY(s.stereotypes)")

        params.append(limit)
        rows = await pool.fetch(f"""
            SELECT s.id, s.symbol_name, s.file_path, s.repository,
                   s.signature, s.line_start, s.symbol_type,
                   s.is_test, s.is_endpoint, 1 AS depth
            FROM code_references r
            JOIN code_symbols s ON s.id = r.source_symbol_id
            JOIN code_symbols h ON h.id = r.target_symbol_id
            WHERE {' AND '.join(conditions)}
            LIMIT ${len(params)}
        """, *params)

        return [self._row_to_call_graph_node(dict(r), rel_type="USES_HOOK") for r in rows]

    # ── Impact analysis ──────────────────────────────────────────────────

    async def get_impact(
        self, method_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
        depth: int = 10,
        summary_only: bool = False,
        limit: int | None = None,
    ) -> ImpactResult | None:
        pool = self._require_pool()
        symbol = await self._resolve_symbol(
            method_name, repository=repository, file_path=file_path,
            entity_id=None, types=["Method", "Constructor"],
        )
        if not symbol:
            return None

        # Push the limit into the SQL when provided. Use a generous default
        # cap to avoid materializing unbounded result sets.
        sql_limit = limit if limit is not None else 10000

        rows = await pool.fetch("""
            WITH RECURSIVE callers AS (
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, 1 AS depth
                FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                WHERE r.target_symbol_id = $1 AND r.ref_type = 'CALLS'
                UNION
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, c.depth + 1
                FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                JOIN callers c ON c.id = r.target_symbol_id
                WHERE c.depth < $2
            )
            SELECT DISTINCT ON (id) * FROM callers ORDER BY id, depth
            LIMIT $3
        """, symbol["id"], depth, sql_limit + 1 if limit is not None else sql_limit)

        rows_list = list(rows)
        truncated = limit is not None and len(rows_list) > limit
        if limit is not None:  # FIX: explicit None check, not truthiness
            rows_list = rows_list[:limit]

        tests = [r for r in rows_list if r["is_test"]]
        endpoints = [r for r in rows_list if r["is_endpoint"] and not r["is_test"]]
        others = [r for r in rows_list if not r["is_test"] and not r["is_endpoint"]]

        return ImpactResult(
            target_name=symbol["symbol_name"],
            target_file=symbol["file_path"],
            target_repository=symbol.get("repository"),
            total_callers=len(rows_list),
            test_count=len(tests),
            endpoint_count=len(endpoints),
            affected_tests=[] if summary_only else [self._row_to_call_graph_node(dict(r)) for r in tests],
            affected_endpoints=[] if summary_only else [self._row_to_call_graph_node(dict(r)) for r in endpoints],
            other_callers=[] if summary_only else [self._row_to_call_graph_node(dict(r)) for r in others],
            truncated=truncated,
        )

    # ── Repository / overview ─────────────────────────────────────────────

    async def list_repositories(self) -> list[dict]:
        pool = self._require_pool()
        rows = await pool.fetch("SELECT * FROM code_repos ORDER BY name")
        return [self._normalize_repo_row(dict(r)) for r in rows]

    async def get_repository_context(self, repository: str) -> RepositoryContext | None:
        pool = self._require_pool()
        repo = await pool.fetchrow(
            "SELECT * FROM code_repos WHERE name = $1", repository
        )
        if not repo:
            return None
        repo_dict = self._normalize_repo_row(dict(repo))
        overview = await self.get_codebase_overview(repository=repository)
        return RepositoryContext(
            name=repo_dict["name"],
            source=repo_dict.get("source"),
            entity_count=repo_dict.get("entity_count", 0),
            last_indexed_at=repo_dict.get("last_indexed_at"),
            last_commit_sha=repo_dict.get("last_commit_sha"),
            total_files=overview.total_files,
            total_classes=overview.total_classes,
            total_interfaces=overview.total_interfaces,
            total_methods=overview.total_methods,
            total_constructors=overview.total_constructors,
            total_fields=overview.total_fields,
            total_packages=overview.total_packages,
            total_hooks=overview.total_hooks,
            total_references=overview.total_references,
            total_exports=0,
            languages=overview.languages,
            top_level_classes=overview.top_level_classes,
            entry_points=overview.entry_points,
        )

    async def get_codebase_overview(
        self,
        repository: str | None = None,
        include_packages: bool = False,
    ) -> CodebaseOverview:
        pool = self._require_pool()
        repo_filter = "WHERE repository = $1" if repository else ""
        params = [repository] if repository else []

        counts = await pool.fetch(f"""
            SELECT symbol_type, COUNT(*) AS cnt
            FROM code_symbols {repo_filter}
            GROUP BY symbol_type
        """, *params)

        count_map = {r["symbol_type"]: r["cnt"] for r in counts}

        langs = await pool.fetch(f"""
            SELECT DISTINCT language FROM code_symbols {repo_filter}
        """, *params)

        if repository:
            ep_query = """
                SELECT symbol_name, file_path FROM code_symbols
                WHERE repository = $1 AND ('endpoint' = ANY(stereotypes) OR symbol_name = 'main')
                LIMIT 20
            """
            entry_points = await pool.fetch(ep_query, repository)
        else:
            entry_points = await pool.fetch("""
                SELECT symbol_name, file_path FROM code_symbols
                WHERE 'endpoint' = ANY(stereotypes) OR symbol_name = 'main'
                LIMIT 20
            """)

        if repository:
            tc_query = """
                SELECT s.symbol_name FROM code_symbols s
                WHERE s.symbol_type = 'Class' AND s.repository = $1
                AND s.id NOT IN (
                    SELECT source_symbol_id FROM code_references WHERE ref_type = 'EXTENDS'
                )
                LIMIT 20
            """
            top_classes = await pool.fetch(tc_query, repository)
        else:
            top_classes = await pool.fetch("""
                SELECT s.symbol_name FROM code_symbols s
                WHERE s.symbol_type = 'Class'
                AND s.id NOT IN (
                    SELECT source_symbol_id FROM code_references WHERE ref_type = 'EXTENDS'
                )
                LIMIT 20
            """)

        return CodebaseOverview(
            total_files=count_map.get("File", 0),
            total_classes=count_map.get("Class", 0),
            total_interfaces=count_map.get("Interface", 0),
            total_methods=count_map.get("Method", 0),
            total_constructors=count_map.get("Constructor", 0),
            total_fields=count_map.get("Field", 0),
            total_packages=count_map.get("Package", 0),
            total_hooks=count_map.get("Hook", 0),
            total_references=count_map.get("Reference", 0),
            languages=[r["language"] for r in langs if r["language"]],
            top_level_classes=[r["symbol_name"] for r in top_classes],
            entry_points=[r["symbol_name"] for r in entry_points],
        )
