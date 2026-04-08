"""PostgreSQL implementation of ReadBackend using asyncpg."""

from __future__ import annotations

import json
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

        async def _init_connection(conn: asyncpg.Connection) -> None:
            # Register pgvector so asyncpg can encode/decode the vector
            # column used by search_code.
            await register_vector(conn)
            # Register a JSONB codec so asyncpg auto-decodes the
            # `properties` JSONB column into Python dicts. Without this
            # asyncpg returns raw JSON strings and _row_to_code_entity
            # crashes on `dict(str)`. Constellation's write path stores
            # properties via auto-encoded dicts; the read path needs the
            # symmetric decoder. Surfaced by Task 5's contract suite.
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )

        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=1, max_size=5, init=_init_connection
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

    async def _resolve_package(
        self,
        name: str,
        *,
        repository: str | None = None,
    ) -> dict | None:
        """Resolve a package/namespace by its full dotted name via ID suffix matching.

        Constellation's .NET parser stores nested namespaces with the full
        path in the entity id (e.g., "repo::Company.Product.Services") but
        only the leaf segment as symbol_name ("Services"). Matching on
        symbol_name directly can't distinguish nested namespaces. Matching
        on the ID suffix (id LIKE '%::Company.Product.Services') correctly
        resolves the full name.

        Mirrors Neo4j's resolver at neo4j.py:930.

        Deliberate divergence from Neo4j's resolver: neo4j.py:930 uses
        `(pkg.name = $package_name OR pkg.id ENDS WITH $suffix)`. We
        intentionally drop the `symbol_name = $name` branch here because:

        - For Java packages, symbol_name already contains the full dotted
          name, so the suffix match subsumes the name match.
        - For .NET nested namespaces, symbol_name is the leaf segment
          only (e.g., "Services" for "Company.Product.Services"). The
          name branch would silently match ANY namespace whose leaf
          equals the query, producing wrong results for nested lookups
          and inconsistent results for leaf-name lookups. Requiring the
          caller to provide the full dotted name (or hit the LIMIT 2
          ambiguity check) is strictly safer.

        Raises ValueError on ambiguity (multiple packages match the suffix).
        """
        pool = self._require_pool()
        # Use id LIKE '%::name' to match the repository prefix + "::" + name.
        # Escape LIKE metacharacters in the name to avoid false matches.
        escaped_name = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        suffix_pattern = f"%::{escaped_name}"

        conditions = [
            "symbol_type = 'Package'",
            "id LIKE $1 ESCAPE '\\'",
        ]
        params: list[Any] = [suffix_pattern]
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
            ids = [r["id"] for r in rows]
            raise ValueError(
                f"Ambiguous package {name!r} — matches: {ids}. "
                f"Provide repository= to disambiguate."
            )
        return dict(rows[0])

    @staticmethod
    def _parameter_suffix_from_entity_id(entity_id: str) -> str | None:
        """Extract the parameter suffix (e.g. '(String,int)') from a
        Constellation entity id.

        Java and C# parsers include the parameter list in the method id
        to disambiguate overloads. Python and JavaScript do not, so the
        id has no suffix. Returns None for languages without parameter
        suffixes, the suffix (including parens) otherwise.

        The approach mirrors Neo4j's _parameter_suffix_from_entity_id:
        find the last '(' in the id, and if the id ends with ')', everything
        from that '(' onward is the parameter suffix. This works for:
        - Methods on classes:  "repo::com.example.Foo.process(String,int)"
        - Top-level functions: "repo::process(String)"
        - Parameterless:       "repo::Foo.run()"

        And correctly rejects:
        - Python/JS (no suffix):           "repo::com.example.Foo.process"
        - Parens in the middle only:       "repo::com.example.Foo(inner).method"
        """
        if not entity_id or not entity_id.endswith(")"):
            return None
        open_idx = entity_id.rfind("(")
        if open_idx < 0:
            return None
        return entity_id[open_idx:]

    async def _resolve_method_family(self, symbol: dict) -> list[str]:
        """Return the list of method IDs that form the polymorphic family
        of the given method: the method itself plus all sibling overrides
        on classes/interfaces related via IMPLEMENTS/EXTENDS (up to 3 hops
        in either direction).

        If the starting method has a parameter suffix (Java/C# overloads),
        candidates are filtered to methods with the same suffix —
        ensuring callers/callees of the wrong overload don't pollute
        the family set.

        Mirrors Neo4j's _method_family_fragment at neo4j.py:451-476.
        The returned list is used as the starting set for
        get_callers/get_callees/get_impact recursive traversals, so
        polymorphic edges are followed correctly.
        """
        pool = self._require_pool()
        parameter_suffix = self._parameter_suffix_from_entity_id(symbol["id"])

        rows = await pool.fetch("""
            WITH RECURSIVE
            -- 1. Find the owning class/interface of the starting method
            owner AS (
                SELECT s.id FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                WHERE r.target_symbol_id = $1
                  AND r.ref_type IN ('HAS_METHOD', 'HAS_CONSTRUCTOR')
                  AND s.symbol_type IN ('Class', 'Interface')
            ),
            -- 2. Walk EXTENDS/IMPLEMENTS up from the owner (parents/interfaces)
            up_chain AS (
                SELECT id, 0 AS hops FROM owner
                UNION
                SELECT s.id, u.hops + 1
                FROM code_references r
                JOIN code_symbols s ON s.id = r.target_symbol_id
                JOIN up_chain u ON u.id = r.source_symbol_id
                WHERE r.ref_type IN ('EXTENDS', 'IMPLEMENTS')
                  AND s.symbol_type IN ('Class', 'Interface')
                  AND u.hops < 3
            ),
            -- 3. Walk EXTENDS/IMPLEMENTS down from the owner (subclasses/implementors)
            down_chain AS (
                SELECT id, 0 AS hops FROM owner
                UNION
                SELECT s.id, d.hops + 1
                FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                JOIN down_chain d ON d.id = r.target_symbol_id
                WHERE r.ref_type IN ('EXTENDS', 'IMPLEMENTS')
                  AND s.symbol_type IN ('Class', 'Interface')
                  AND d.hops < 3
            ),
            -- 4. Combine up+down into the full family of related owners
            family_owners AS (
                SELECT id FROM up_chain
                UNION
                SELECT id FROM down_chain
            )
            -- 5. Find all same-name methods on those owners, filtered by
            -- parameter suffix (if present) to disambiguate overloaded
            -- methods in Java/C#.
            --
            -- Suffix filter: m.id must END WITH symbol_name + parameter_suffix.
            -- We use the right() function so pattern metacharacters in the
            -- name (e.g., underscore in Python-style names — unlikely in
            -- Java/C# but defensive) can't produce false matches. This is
            -- a literal ends-with check with no pattern interpretation.
            SELECT DISTINCT m.id
            FROM code_references r
            JOIN code_symbols m ON m.id = r.target_symbol_id
            JOIN family_owners fo ON fo.id = r.source_symbol_id
            WHERE r.ref_type IN ('HAS_METHOD', 'HAS_CONSTRUCTOR')
              AND m.symbol_name = $2
              AND m.symbol_type IN ('Method', 'Constructor')
              AND (
                  $3::text IS NULL
                  OR right(m.id, length($2 || $3::text)) = ($2 || $3::text)
              )
        """, symbol["id"], symbol["symbol_name"], parameter_suffix)

        family_ids = [r["id"] for r in rows]
        # Always include the starting symbol, even if it has no owner (top-level fn)
        if symbol["id"] not in family_ids:
            family_ids.append(symbol["id"])
        return family_ids

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

    def _row_to_call_graph_node(
        self, row: dict, rel_type: str = "CALLS", truncated: bool = False,
    ) -> CallGraphNode:
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
            truncated=truncated,
        )

    @staticmethod
    def _full_name_from_id(entity_id: str, fallback_name: str) -> str:
        """Strip the repository prefix from a Constellation entity id.
        Mirrors neo4j.py:395."""
        if "::" in entity_id:
            return entity_id.split("::", 1)[1]
        return fallback_name

    @staticmethod
    def _looks_like_symbol_query(query: str) -> bool:
        """Heuristic: does this query look like a code identifier rather than
        a natural-language phrase? Mirrors neo4j.py:200.

        Rules:
        - Must be non-empty and have no spaces
        - Contains a code-like separator (/, ., ::, _, -) → identifier
        - Otherwise must contain an uppercase letter after index 0 (CamelCase) → identifier
        - Single lowercase word like 'user' or 'get' → NOT an identifier
        """
        stripped = query.strip()
        if not stripped or " " in stripped:
            return False
        if any(token in stripped for token in ("/", ".", "::", "_", "-")):
            return True
        return any(char.isupper() for char in stripped[1:])

    @staticmethod
    def _apply_code_mode(entities: list[CodeEntity], code_mode: str) -> None:
        """Mutate entities to reflect the requested code_mode.

        - "none": zero out the code field
        - "signature": replace code with the method's signature
        - "preview": keep only the first 10 lines of code
        - anything else: leave code unchanged

        Inspired by neo4j.py's process_code (neo4j.py:552) and its
        search_code merge loop (neo4j.py:598). Note: the Postgres
        preview branch is a simple ``splitlines()[:10]`` slice and does
        NOT match neo4j.py exactly — neo4j appends a ``"... (truncated)"``
        marker when it truncates, falls back to ``signature`` when ``code``
        is empty/None, and splits on ``"\\n"`` rather than using
        ``splitlines()`` (so ``\\r\\n`` is handled differently). Aligning
        the two is a separate task (not Task 1's scope); this helper
        preserves the pre-existing Postgres behavior verbatim.
        """
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
        self._apply_code_mode(entities, code_mode)

        # Blend in exact-symbol matches when the query looks like an identifier.
        # Mirrors neo4j.py:572 behavior: a user searching for "UserService"
        # expects the symbol itself to appear in results, not only semantically
        # similar methods.
        if self._looks_like_symbol_query(query):
            symbol_results = await self.find_symbols(
                query,
                entity_types=[entity_type] if entity_type else None,
                repository=repository,
                file_pattern=file_pattern,
                limit=limit,
                exact=True,
                language=language,
                stereotype=stereotype,
            )
            # Fallback to fuzzy symbol search if exact returned nothing
            if not symbol_results:
                symbol_results = await self.find_symbols(
                    query,
                    entity_types=[entity_type] if entity_type else None,
                    repository=repository,
                    file_pattern=file_pattern,
                    limit=limit,
                    exact=False,
                    language=language,
                    stereotype=stereotype,
                )
            # Apply the same code_mode transformation to symbol results so
            # the response contract is consistent across both result types.
            # Without this, code_mode="none" would leak full source code
            # for identifier queries.
            self._apply_code_mode(symbol_results, code_mode)

            # Merge: exact (or fuzzy fallback) matches first, then vector hits, dedup by entity_id
            seen_ids = {e.entity_id for e in symbol_results}
            combined = list(symbol_results)
            for e in entities:
                if e.entity_id not in seen_ids:
                    combined.append(e)
                    seen_ids.add(e.entity_id)
            return combined[:limit]

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
            # Match both symbol_name and file_path in fuzzy mode (matches Neo4j)
            conditions.append(
                f"(symbol_name ILIKE ${len(params)} OR file_path ILIKE ${len(params)})"
            )

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

        # Expand to the full polymorphic method family
        family_ids = await self._resolve_method_family(symbol)

        # Over-fetch by 1 so we can detect truncation without a separate
        # count query. Mirrors neo4j.py:652.
        rows = await pool.fetch("""
            WITH RECURSIVE callers AS (
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, 1 AS depth
                FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                WHERE r.target_symbol_id = ANY($1)
                  AND r.ref_type = 'CALLS'
                  AND s.id != ALL($1)
                UNION
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, c.depth + 1
                FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                JOIN callers c ON c.id = r.target_symbol_id
                WHERE c.depth < $2
                  AND s.id != ALL($1)
            )
            SELECT DISTINCT ON (id) * FROM callers ORDER BY id, depth
            LIMIT $3
        """, family_ids, depth, limit + 1)

        truncated = len(rows) > limit
        if truncated:
            rows = rows[:limit]
        return [
            self._row_to_call_graph_node(dict(r), truncated=truncated)
            for r in rows
        ]

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

        family_ids = await self._resolve_method_family(symbol)

        # Over-fetch calls by limit+1 so we can detect truncation.
        call_rows = await pool.fetch("""
            WITH RECURSIVE callees AS (
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, 1 AS depth
                FROM code_references r
                JOIN code_symbols s ON s.id = r.target_symbol_id
                WHERE r.source_symbol_id = ANY($1) AND r.ref_type = 'CALLS'
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
        """, family_ids, depth, limit + 1)

        # Over-fetch hooks by limit+1 too (previously unlimited, which
        # meant a full-limit call set silently dropped every hook row).
        hook_rows = await pool.fetch("""
            SELECT s.id, s.symbol_name, s.file_path, s.repository,
                   s.signature, s.line_start, s.symbol_type,
                   s.is_test, s.is_endpoint, 1 AS depth
            FROM code_references r
            JOIN code_symbols s ON s.id = r.target_symbol_id
            WHERE r.source_symbol_id = ANY($1) AND r.ref_type = 'USES_HOOK'
            LIMIT $2
        """, family_ids, limit + 1)

        # Truncation is true if either source overflowed individually OR
        # if the combined deduped list exceeds the limit. Matches Neo4j's
        # holistic truncation detection (neo4j.py:742).
        call_over = len(call_rows) > limit
        hook_over = len(hook_rows) > limit

        # Dedupe by id across the two sources. Calls come first in the
        # merged order, matching Neo4j's ordering for the returned slice
        # (Neo4j's formal sort on depth/file/line happens to coincide
        # with calls-first because depth=1 ties are broken by file/line).
        seen_ids: set[str] = set()
        merged: list[CallGraphNode] = []

        for r in call_rows:
            if r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            merged.append(
                self._row_to_call_graph_node(dict(r), rel_type="CALLS", truncated=False)
            )

        for r in hook_rows:
            if r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            merged.append(
                self._row_to_call_graph_node(dict(r), rel_type="USES_HOOK", truncated=False)
            )

        # Compute truncation AFTER merge+dedup so we know the true
        # post-dedup count. Truncate to limit.
        merged_over = len(merged) > limit
        truncated = call_over or hook_over or merged_over
        if len(merged) > limit:
            merged = merged[:limit]

        # Propagate truncated flag to every surviving node. We couldn't
        # set it at construction time because we didn't know the final
        # combined count yet.
        if truncated:
            for node in merged:
                node.truncated = True

        return merged

    # ── Full context ─────────────────────────────────────────────────────

    async def get_function_context(
        self, method_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
    ) -> FunctionContext | None:
        pool = self._require_pool()
        symbol = await self._resolve_symbol(
            method_name, repository=repository, file_path=file_path,
            entity_id=None, types=["Method", "Constructor"],
        )
        if not symbol:
            return None

        # Look up the owning class via HAS_METHOD or HAS_CONSTRUCTOR.
        # Scope by repository so cross-repo collisions with the same method
        # name can't surface the wrong class.
        class_row = await pool.fetchrow("""
            SELECT s.symbol_name FROM code_references r
            JOIN code_symbols s ON s.id = r.source_symbol_id
            WHERE r.target_symbol_id = $1
              AND r.ref_type IN ('HAS_METHOD', 'HAS_CONSTRUCTOR')
              AND s.symbol_type IN ('Class', 'Interface')
              AND s.repository = $2
            LIMIT 1
        """, symbol["id"], symbol.get("repository") or "")
        class_name = class_row["symbol_name"] if class_row else None

        callers = await self.get_callers(
            method_name, repository=repository, file_path=file_path, depth=1,
        )
        callees = await self.get_callees(
            method_name, repository=repository, file_path=file_path, depth=1,
        )

        return FunctionContext(
            name=symbol["symbol_name"],
            full_name=self._full_name_from_id(symbol["id"], symbol["symbol_name"]),
            file_path=symbol["file_path"],
            repository=symbol.get("repository"),
            code=symbol.get("code"),
            signature=symbol.get("signature"),
            docstring=symbol.get("docstring"),
            class_name=class_name,
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
            full_name=self._full_name_from_id(cls["id"], cls["symbol_name"]),
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
        pkg = await self._resolve_package(package_name, repository=repository)
        if not pkg:
            return None

        # All direct IN_PACKAGE members (files, classes, interfaces, methods,
        # hooks, references). Neo4j uses the same direct-membership model.
        members = await pool.fetch("""
            SELECT s.symbol_name, s.symbol_type, s.file_path
            FROM code_references r JOIN code_symbols s ON s.id = r.source_symbol_id
            WHERE r.target_symbol_id = $1 AND r.ref_type = 'IN_PACKAGE'
        """, pkg["id"])

        # Child packages discovered via ID prefix matching, not via
        # package-to-package CONTAINS edges (which .NET parser doesn't
        # create). Only immediate children: the child's id must have
        # EXACTLY ONE additional dotted segment beyond the parent's id.
        parent_id = pkg["id"]
        escaped_id = parent_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        child_prefix = f"{escaped_id}.%"
        child_rows = await pool.fetch("""
            SELECT id, symbol_name FROM code_symbols
            WHERE symbol_type = 'Package'
              AND id LIKE $1 ESCAPE '\\'
              AND repository = $2
        """, child_prefix, pkg["repository"])
        # Reconstruct the full dotted name from the ID for the returned name.
        full_name = self._full_name_from_id(parent_id, pkg["symbol_name"])
        # Filter to immediate children: full_name parts = parent parts + 1
        parent_depth = len(full_name.split("."))
        immediate_children: list[str] = []
        for r in child_rows:
            child_full = self._full_name_from_id(r["id"], r["symbol_name"])
            if len(child_full.split(".")) == parent_depth + 1:
                immediate_children.append(child_full)

        return PackageContext(
            name=full_name,
            repository=pkg.get("repository"),
            package_id=pkg["id"],
            # Files are ANY unique file_path across all members. This mirrors
            # Neo4j's `collect(DISTINCT member.file_path)` at neo4j.py:943.
            # The previous implementation filtered for symbol_type == "File"
            # which always returned empty because Constellation's parsers
            # attach IN_PACKAGE from classes/methods/interfaces, NOT from
            # file nodes (see constellation/parsers/java.py:204).
            files=sorted({r["file_path"] for r in members if r["file_path"]}),
            classes=[r["symbol_name"] for r in members if r["symbol_type"] == "Class"],
            interfaces=[r["symbol_name"] for r in members if r["symbol_type"] == "Interface"],
            methods=[r["symbol_name"] for r in members if r["symbol_type"] == "Method"],
            hooks=[r["symbol_name"] for r in members if r["symbol_type"] == "Hook"],
            references=[r["symbol_name"] for r in members if r["symbol_type"] == "Reference"],
            child_packages=sorted(set(immediate_children)),
        )

    async def get_file_context(
        self, file_path: str, *,
        repository: str | None = None,
    ) -> FileContext | None:
        pool = self._require_pool()
        # Suffix match on file_path + LIMIT 2 ambiguity detection.
        # Mirrors Neo4j's _resolve_file_target at neo4j.py:484, which uses
        # `f.file_path ENDS WITH $file_path`. Constellation stores absolute
        # file paths, but callers commonly pass relative suffixes like
        # "src/Service.java" — so we need suffix matching, not equality.
        #
        # We use right()+length() instead of LIKE to avoid pattern
        # metacharacter issues: file paths on some filesystems can contain
        # '_' or '%' which LIKE would interpret as wildcards. right() is
        # a literal ends-with check. Same approach as the method-family
        # suffix filter in _resolve_method_family.
        conditions = [
            "symbol_type = 'File'",
            "right(file_path, length($1)) = $1",
        ]
        params: list[Any] = [file_path]
        if repository:
            params.append(repository)
            conditions.append(f"repository = ${len(params)}")
        rows = await pool.fetch(
            f"SELECT * FROM code_symbols WHERE {' AND '.join(conditions)} "
            f"ORDER BY file_path LIMIT 2",
            *params,
        )
        if not rows:
            return None
        if len(rows) > 1:
            examples = ", ".join(r.get("file_path") or "?" for r in rows)
            raise ValueError(
                f"Ambiguous file path {file_path!r} — matches multiple files: "
                f"{examples}. Provide repository= or a longer path suffix to disambiguate."
            )
        file_sym = dict(rows[0])
        file_repo = file_sym["repository"]
        file_fpath = file_sym["file_path"]

        # ── Shared-file members ──────────────────────────────────────
        # Classes, interfaces, constructors, fields, and references are
        # attached to their enclosing class (HAS_METHOD, HAS_CONSTRUCTOR,
        # HAS_FIELD) or to their enclosing method (CALLS→Reference), NOT
        # to the file via CONTAINS. Match them by shared repository +
        # file_path, regardless of edge type. Mirrors Neo4j's approach at
        # neo4j.py:1208-1227.
        members = await pool.fetch("""
            SELECT symbol_name, symbol_type
            FROM code_symbols
            WHERE repository = $1
              AND file_path = $2
              AND symbol_type IN (
                  'Class', 'Interface', 'Constructor', 'Field', 'Reference'
              )
        """, file_repo, file_fpath)

        # ── Top-level methods via File→CONTAINS→Method ───────────────
        # This is how JS/Python top-level functions are represented.
        # Java class methods do NOT appear here because the Java parser
        # attaches them via HAS_METHOD from Class, not CONTAINS from File.
        # Neo4j uses the same edge query at neo4j.py:1214.
        top_level_method_rows = await pool.fetch("""
            SELECT s.symbol_name
            FROM code_references r
            JOIN code_symbols s ON s.id = r.target_symbol_id
            WHERE r.source_symbol_id = $1
              AND r.ref_type = 'CONTAINS'
              AND s.symbol_type = 'Method'
        """, file_sym["id"])

        # ── Exports (File→EXPORTS→*) ─────────────────────────────────
        # Unchanged from previous implementation — File→EXPORTS→* is the
        # correct pattern and matches Neo4j at neo4j.py:1228.
        export_rows = await pool.fetch("""
            SELECT s.*
            FROM code_references r
            JOIN code_symbols s ON s.id = r.target_symbol_id
            WHERE r.source_symbol_id = $1 AND r.ref_type = 'EXPORTS'
        """, file_sym["id"])

        # ── Packages via shared-file member IN_PACKAGE ───────────────
        # Any IN_PACKAGE edge from a member in this file. Mirrors Neo4j
        # at neo4j.py:1205-1207. Note: this still returns the leaf-only
        # `symbol_name` for nested .NET namespaces — Plan C
        # (2026-04-08-telescope-nested-namespace-reconstruction.md)
        # will reconstruct full dotted names via _full_name_from_id.
        package_rows = await pool.fetch("""
            SELECT DISTINCT pkg.id, pkg.symbol_name
            FROM code_symbols member
            JOIN code_references r ON r.source_symbol_id = member.id
            JOIN code_symbols pkg ON pkg.id = r.target_symbol_id
            WHERE member.repository = $1
              AND member.file_path = $2
              AND r.ref_type = 'IN_PACKAGE'
              AND pkg.symbol_type = 'Package'
        """, file_repo, file_fpath)

        # ── Hooks via shared-file match ──────────────────────────────
        # Hook entities live in whichever file their usage site is. Match
        # by shared repository + file_path. Mirrors Neo4j at
        # neo4j.py:1243-1244. Replaces the previous two-hop File→CONTAINS
        # →Method→USES_HOOK→Hook traversal.
        hook_rows = await pool.fetch("""
            SELECT symbol_name
            FROM code_symbols
            WHERE repository = $1
              AND file_path = $2
              AND symbol_type = 'Hook'
        """, file_repo, file_fpath)

        return FileContext(
            name=file_sym["symbol_name"],
            file_path=file_sym["file_path"],
            repository=file_sym.get("repository"),
            language=file_sym.get("language"),
            content_hash=file_sym.get("content_hash"),
            packages=[r["symbol_name"] for r in package_rows],
            exports=[
                self._row_to_code_entity(dict(r))
                for r in export_rows
                if r.get("symbol_name") and r.get("file_path")
            ],
            classes=[r["symbol_name"] for r in members if r["symbol_type"] == "Class"],
            interfaces=[r["symbol_name"] for r in members if r["symbol_type"] == "Interface"],
            top_level_methods=[r["symbol_name"] for r in top_level_method_rows],
            hooks=[r["symbol_name"] for r in hook_rows],
            constructors=[r["symbol_name"] for r in members if r["symbol_type"] == "Constructor"],
            fields=[r["symbol_name"] for r in members if r["symbol_type"] == "Field"],
            references=[r["symbol_name"] for r in members if r["symbol_type"] == "Reference"],
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

        # Over-fetch by 1 so we can detect truncation without a separate
        # count query. Mirrors neo4j.py:1330.
        params.append(limit + 1)
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

        truncated = len(rows) > limit
        if truncated:
            rows = rows[:limit]
        return [
            self._row_to_call_graph_node(dict(r), rel_type="USES_HOOK", truncated=truncated)
            for r in rows
        ]

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

        # Expand to the polymorphic method family (same as get_callers/get_callees)
        family_ids = await self._resolve_method_family(symbol)

        # Fetch the full transitive caller set WITHOUT the per-category LIMIT.
        # The `limit` parameter is per-category (tests/endpoints/others) —
        # applying it at the SQL level would conflate categories and
        # produce undercounted totals.
        #
        # We cap at a safe maximum (10_000) to prevent runaway traversals
        # on pathological graphs. Neo4j caps at 200 (neo4j.py:1378); we
        # use a larger cap here because recursive CTEs are cheap and the
        # per-category limit below still truncates the user-visible output.
        # If a real codebase hits 10_000, the total_callers count accurately
        # reflects that ceiling and the user can drill deeper via get_callers
        # with a specific entity_id.
        #
        # Also: exclude family methods themselves from the caller set
        # (a polymorphic override calling its own sibling shouldn't count
        # as impact on itself). Mirrors Neo4j's `caller <> m` filter at
        # neo4j.py:1370.
        rows = await pool.fetch("""
            WITH RECURSIVE callers AS (
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, 1 AS depth
                FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                WHERE r.target_symbol_id = ANY($1)
                  AND r.ref_type = 'CALLS'
                  AND s.id != ALL($1)
                UNION
                SELECT s.id, s.symbol_name, s.file_path, s.repository,
                       s.signature, s.line_start, s.symbol_type,
                       s.is_test, s.is_endpoint, c.depth + 1
                FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                JOIN callers c ON c.id = r.target_symbol_id
                WHERE c.depth < $2
                  AND s.id != ALL($1)
            )
            SELECT DISTINCT ON (id) * FROM callers ORDER BY id, depth
            LIMIT 10001
        """, family_ids, depth)

        # Detect the hard cap being hit so truncated reflects reality even
        # when the per-category limit is not set.
        hit_hard_cap = len(rows) > 10000
        if hit_hard_cap:
            rows = rows[:10000]

        # Categorize ALL rows (not truncated ones)
        tests = [r for r in rows if r["is_test"]]
        endpoints = [r for r in rows if r["is_endpoint"] and not r["is_test"]]
        others = [r for r in rows if not r["is_test"] and not r["is_endpoint"]]

        # True totals, before per-category truncation
        total_callers = len(rows)
        total_tests = len(tests)
        total_endpoints = len(endpoints)

        # Per-category limit truncation (matches Neo4j's behavior)
        truncated = hit_hard_cap
        if limit is not None:
            truncated = truncated or (
                len(tests) > limit
                or len(endpoints) > limit
                or len(others) > limit
            )
            tests = tests[:limit]
            endpoints = endpoints[:limit]
            others = others[:limit]

        return ImpactResult(
            target_name=symbol["symbol_name"],
            target_file=symbol["file_path"],
            target_repository=symbol.get("repository"),
            total_callers=total_callers,
            test_count=total_tests,
            endpoint_count=total_endpoints,
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
            total_exports=overview.total_exports,
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

        # Exports count — number of EXPORTS relationships in the repository
        if repository:
            export_rows = await pool.fetch("""
                SELECT COUNT(*) AS cnt FROM code_references r
                JOIN code_symbols s ON s.id = r.source_symbol_id
                WHERE s.repository = $1 AND r.ref_type = 'EXPORTS'
            """, repository)
        else:
            export_rows = await pool.fetch("""
                SELECT COUNT(*) AS cnt FROM code_references
                WHERE ref_type = 'EXPORTS'
            """)
        total_exports = export_rows[0]["cnt"] if export_rows else 0

        # Package names (only when include_packages=True to avoid an unnecessary query)
        packages: list[str] = []
        if include_packages:
            if repository:
                pkg_rows = await pool.fetch("""
                    SELECT symbol_name FROM code_symbols
                    WHERE symbol_type = 'Package' AND repository = $1
                    ORDER BY symbol_name
                """, repository)
            else:
                pkg_rows = await pool.fetch("""
                    SELECT symbol_name FROM code_symbols
                    WHERE symbol_type = 'Package'
                    ORDER BY symbol_name
                """)
            packages = [r["symbol_name"] for r in pkg_rows]

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
            total_exports=total_exports,
            languages=[r["language"] for r in langs if r["language"]],
            packages=packages,
            top_level_classes=[r["symbol_name"] for r in top_classes],
            entry_points=[r["symbol_name"] for r in entry_points],
        )
