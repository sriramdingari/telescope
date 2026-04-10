"""Neo4j implementation of ReadBackend for querying Constellation's code knowledge graph."""

import logging
import re
from typing import Any

from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

from telescope.config import get_config
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
from telescope.backends.base import ReadBackend

logger = logging.getLogger(__name__)

# Map entity types to Constellation's vector index names
ENTITY_TYPE_TO_INDEX = {
    "method": "vector_method_embedding",
    "class": "vector_class_embedding",
    "interface": "vector_interface_embedding",
    "constructor": "vector_constructor_embedding",
}

ENTITY_TYPE_TO_LABEL = {
    "method": "Method",
    "class": "Class",
    "interface": "Interface",
    "constructor": "Constructor",
}

SYMBOL_ENTITY_TYPE_TO_LABEL = {
    "file": "File",
    "package": "Package",
    "class": "Class",
    "interface": "Interface",
    "method": "Method",
    "constructor": "Constructor",
    "field": "Field",
    "hook": "Hook",
    "reference": "Reference",
}

FILTERED_SEARCH_MIN_CANDIDATES = 100
FILTERED_SEARCH_CANDIDATE_MULTIPLIER = 25
FILTERED_SEARCH_MAX_CANDIDATES = 5000
ENTITY_RESERVED_PROPERTIES = {
    "id",
    "name",
    "repository",
    "file_path",
    "line_number",
    "line_end",
    "language",
    "code",
    "signature",
    "return_type",
    "docstring",
    "modifiers",
    "stereotypes",
    "content_hash",
    "embedding",
}


class Neo4jReadBackend(ReadBackend):
    """Neo4j implementation of ReadBackend."""

    def __init__(self):
        self.config = get_config()
        self._driver = None
        self._openai = None

    async def connect(self):
        """Connect to Neo4j and initialize OpenAI client."""
        self._driver = AsyncGraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )
        await self._driver.verify_connectivity()
        kwargs = {"api_key": self.config.openai_api_key}
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        self._openai = AsyncOpenAI(**kwargs)
        logger.info(f"Connected to Neo4j at {self.config.neo4j_uri}")

    async def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
        if self._openai and hasattr(self._openai, "close"):
            await self._openai.close()

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Convert Neo4j-specific scalar types into JSON-safe values."""
        if isinstance(value, dict):
            return {k: Neo4jReadBackend._normalize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Neo4jReadBackend._normalize_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(Neo4jReadBackend._normalize_value(v) for v in value)

        iso_format = getattr(value, "iso_format", None)
        if callable(iso_format):
            return iso_format()

        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            try:
                return isoformat()
            except TypeError:
                pass

        return value

    async def _query(self, cypher: str, **params) -> list[dict]:
        """Execute a Cypher query and return results as dicts."""
        async with self._driver.session() as session:
            result = await session.run(cypher, **params)
            records = await result.data()
            return [self._normalize_value(record) for record in records]

    async def _get_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        response = await self._openai.embeddings.create(
            model=self.config.embedding_model,
            input=text,
            dimensions=self.config.embedding_dimensions,
        )
        return response.data[0].embedding

    @staticmethod
    def _file_pattern_to_regex(file_pattern: str) -> str:
        """Translate a simple wildcard path pattern into a Cypher regex."""
        pattern = file_pattern
        if "*" not in pattern and "?" not in pattern:
            pattern = f"*{pattern}*"

        parts: list[str] = []
        for ch in pattern:
            if ch == "*":
                parts.append(".*")
            elif ch == "?":
                parts.append(".")
            else:
                parts.append(re.escape(ch))

        return "^" + "".join(parts) + "$"

    def _build_search_filters(
        self,
        alias: str,
        repository: str | None = None,
        file_pattern: str | None = None,
        language: str | None = None,
        stereotype: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build parameterized filters for repository and wildcard file matches."""
        filters: list[str] = []
        params: dict[str, Any] = {}

        if repository:
            filters.append(f"{alias}.repository = $repository")
            params["repository"] = repository

        if file_pattern:
            filters.append(f"{alias}.file_path =~ $file_regex")
            params["file_regex"] = self._file_pattern_to_regex(file_pattern)

        if language:
            filters.append(f"{alias}.language = $language")
            params["language"] = language

        if stereotype:
            filters.append(f"$stereotype IN coalesce({alias}.stereotypes, [])")
            params["stereotype"] = stereotype

        clause = ""
        if filters:
            clause = " AND " + " AND ".join(filters)

        return clause, params

    @staticmethod
    def _clamp_result_limit(limit: int, maximum: int = 200) -> int:
        """Clamp user-provided result limits to a safe positive range."""
        return max(1, min(limit, maximum))

    @staticmethod
    def _looks_like_symbol_query(query: str) -> bool:
        """Heuristic to decide whether a query should blend in symbol search."""
        stripped = query.strip()
        if not stripped or " " in stripped:
            return False
        if any(token in stripped for token in ("/", ".", "::", "_", "-")):
            return True
        return any(char.isupper() for char in stripped[1:])

    def _build_method_match(
        self,
        alias: str,
        *,
        method_name: str | None = None,
        entity_id: str | None = None,
        repository: str | None = None,
        file_path: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a parameterized method/constructor match clause."""
        filters = [f"({alias}:Method OR {alias}:Constructor)"]
        params: dict[str, Any] = {}

        if entity_id is not None:
            filters.append(f"{alias}.id = $entity_id")
            params["entity_id"] = entity_id
        else:
            filters.append(f"{alias}.name = $method_name")
            params["method_name"] = method_name
            if repository:
                filters.append(f"{alias}.repository = $repository")
                params["repository"] = repository
            if file_path:
                filters.append(f"{alias}.file_path ENDS WITH $file_path")
                params["file_path"] = file_path

        return f"MATCH ({alias}) WHERE " + " AND ".join(filters), params

    def _build_class_match(
        self,
        alias: str,
        *,
        class_name: str,
        repository: str | None = None,
        file_path: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a parameterized class/interface match clause."""
        filters = [f"({alias}:Class OR {alias}:Interface)", f"{alias}.name = $class_name"]
        params: dict[str, Any] = {"class_name": class_name}

        if repository:
            filters.append(f"{alias}.repository = $repository")
            params["repository"] = repository
        if file_path:
            filters.append(f"{alias}.file_path ENDS WITH $file_path")
            params["file_path"] = file_path

        return f"MATCH ({alias}) WHERE " + " AND ".join(filters), params

    @staticmethod
    def _is_similarity_function_unavailable(exc: Exception) -> bool:
        """Detect older Neo4j instances that cannot score filtered vectors directly."""
        message = str(exc)
        if "vector.similarity.cosine" not in message:
            return False

        unavailable_markers = (
            "Unknown function",
            "not registered",
            "There is no function with the name",
            "Neo.ClientError.Statement.SyntaxError",
        )
        return any(marker in message for marker in unavailable_markers)

    async def _search_index(
        self,
        *,
        entity_type: str,
        index_name: str,
        limit: int,
        embedding: list[float],
        repository: str | None = None,
        file_pattern: str | None = None,
        language: str | None = None,
        stereotype: str | None = None,
    ) -> list[dict]:
        """Search a single entity type, using exact filtered scoring when available."""
        label = ENTITY_TYPE_TO_LABEL[entity_type]
        filter_clause, filter_params = self._build_search_filters(
            "n",
            repository=repository,
            file_pattern=file_pattern,
            language=language,
            stereotype=stereotype,
        )

        return_clause = f"""
            RETURN n.id AS id, n.name AS name, n.file_path AS file_path,
                   n.repository AS repository,
                   n.line_number AS line_number, n.line_end AS line_end,
                   n.code AS code, n.signature AS signature,
                   n.language AS language, n.return_type AS return_type,
                   n.modifiers AS modifiers, n.stereotypes AS stereotypes,
                   n.content_hash AS content_hash,
                   properties(n) AS properties,
                   '{entity_type}' AS entity_type, score
            ORDER BY score DESC
            LIMIT $limit
        """

        params = {"embedding": embedding, "limit": limit, **filter_params}

        if repository or file_pattern:
            similarity_cypher = f"""
                MATCH (n:{label})
                WHERE n.embedding IS NOT NULL{filter_clause}
                WITH n, vector.similarity.cosine(n.embedding, $embedding) AS score
                {return_clause}
            """
            try:
                return await self._query(similarity_cypher, **params)
            except Exception as exc:
                if not self._is_similarity_function_unavailable(exc):
                    raise
                logger.warning(
                    "Falling back to overfetched vector index search for filtered %s queries: %s",
                    entity_type,
                    exc,
                )

        candidate_limit = limit
        if repository or file_pattern:
            candidate_limit = max(
                FILTERED_SEARCH_MIN_CANDIDATES,
                limit * FILTERED_SEARCH_CANDIDATE_MULTIPLIER,
            )

        candidate_limit = min(candidate_limit, FILTERED_SEARCH_MAX_CANDIDATES)

        while True:
            cypher = f"""
                CALL db.index.vector.queryNodes('{index_name}', $candidate_limit, $embedding)
                YIELD node AS n, score
                WHERE n.embedding IS NOT NULL{filter_clause}
                {return_clause}
            """
            results = await self._query(
                cypher,
                candidate_limit=candidate_limit,
                **params,
            )
            if not (repository or file_pattern):
                return results
            if len(results) >= limit or candidate_limit >= FILTERED_SEARCH_MAX_CANDIDATES:
                return results
            candidate_limit = min(candidate_limit * 2, FILTERED_SEARCH_MAX_CANDIDATES)

    async def _resolve_method_target(
        self,
        method_name: str,
        repository: str | None = None,
        file_path: str | None = None,
        entity_id: str | None = None,
    ) -> dict | None:
        """Resolve one method/constructor for exact-context queries."""
        match_clause, params = self._build_method_match(
            "m",
            method_name=method_name,
            repository=repository,
            file_path=file_path,
            entity_id=entity_id,
        )
        cypher = f"""
            {match_clause}
            OPTIONAL MATCH (c)-[:HAS_METHOD|HAS_CONSTRUCTOR]->(m)
            WHERE c:Class OR c:Interface
            RETURN m.name AS name, m.id AS id, m.file_path AS file_path,
                   m.repository AS repository,
                   m.code AS code, m.signature AS signature, m.docstring AS docstring,
                   c.name AS class_name
            ORDER BY m.file_path, m.line_number
            LIMIT 2
        """

        results = await self._query(cypher, **params)
        if not results:
            return None
        if len(results) > 1:
            examples = ", ".join(
                f"{r.get('file_path') or '?'} [{r.get('signature') or r.get('id') or r.get('name')}]"
                for r in results
            )
            raise ValueError(
                f"Method '{method_name}' is ambiguous for the current filters. Matching entries: {examples}"
            )
        return results[0]

    @staticmethod
    def _full_name_from_id(entity_id: str, fallback_name: str) -> str:
        """Strip the repository prefix from a Constellation entity id."""
        if "::" in entity_id:
            return entity_id.split("::", 1)[1]
        return fallback_name

    @staticmethod
    def _parameter_suffix_from_entity_id(entity_id: str | None) -> str | None:
        """Extract a trailing Java-style parameter signature suffix from an entity id."""
        if not entity_id or not entity_id.endswith(")"):
            return None
        open_paren = entity_id.rfind("(")
        if open_paren == -1:
            return None
        return entity_id[open_paren:]

    @staticmethod
    def _label_to_entity_type(label: str | None) -> str:
        """Normalize a Neo4j label into Telescope's lowercase entity type names."""
        if not label:
            return "method"
        return label.lower()

    @staticmethod
    def _custom_entity_properties(properties: dict[str, Any] | None) -> dict[str, Any]:
        """Return only non-core graph properties for an entity."""
        if not properties:
            return {}
        return {
            key: value
            for key, value in properties.items()
            if key not in ENTITY_RESERVED_PROPERTIES and value is not None
        }

    def _entity_from_record(self, record: dict[str, Any]) -> CodeEntity:
        """Build a CodeEntity from a Neo4j record."""
        return CodeEntity(
            name=record["name"],
            file_path=record.get("file_path") or "",
            repository=record.get("repository"),
            entity_id=record.get("id"),
            line_start=record.get("line_number") or record.get("line_start"),
            line_end=record.get("line_end"),
            code=record.get("code"),
            signature=record.get("signature"),
            docstring=record.get("docstring"),
            score=record.get("score", 0.0),
            entity_type=self._label_to_entity_type(record.get("entity_type")),
            language=record.get("language"),
            return_type=record.get("return_type"),
            modifiers=[value for value in (record.get("modifiers") or []) if value],
            stereotypes=[value for value in (record.get("stereotypes") or []) if value],
            content_hash=record.get("content_hash"),
            properties=self._custom_entity_properties(record.get("properties")),
        )

    @staticmethod
    def _apply_code_mode(entities: list[CodeEntity], code_mode: str) -> None:
        """Mutate entities to reflect the requested code_mode.

        Mirrors the postgres.py:367-395 helper contract, but preserves
        Neo4j's pre-existing preview behavior:

        - "none": zero out the code field
        - "signature": replace code with the signature
        - "preview": if code is empty/None, fall back to signature;
          otherwise keep the first 10 lines (splitting on "\\n") and
          append "\\n... (truncated)" when the original has more than
          10 lines
        - anything else (e.g. "full"): leave code unchanged
        """
        if code_mode == "none":
            for e in entities:
                e.code = None
        elif code_mode == "signature":
            for e in entities:
                e.code = e.signature
        elif code_mode == "preview":
            for e in entities:
                if not e.code:
                    e.code = e.signature
                    continue
                lines = e.code.split("\n")
                if len(lines) <= 10:
                    continue
                e.code = "\n".join(lines[:10]) + "\n... (truncated)"

    def _method_family_fragment(self, alias: str) -> str:
        """Build a method-family expansion based on class/interface relationships."""
        return f"""
            OPTIONAL MATCH (owner)-[:HAS_METHOD|HAS_CONSTRUCTOR]->({alias})
            WHERE owner:Class OR owner:Interface
            OPTIONAL MATCH (owner)-[up_rels*1..3]->(up_owner)
            WHERE up_owner:Class OR up_owner:Interface
              AND all(rel IN up_rels WHERE type(rel) IN ['IMPLEMENTS', 'EXTENDS'])
            OPTIONAL MATCH (down_owner)-[down_rels*1..3]->(owner)
            WHERE down_owner:Class OR down_owner:Interface
              AND all(rel IN down_rels WHERE type(rel) IN ['IMPLEMENTS', 'EXTENDS'])
            WITH {alias}, owner,
                 collect(DISTINCT up_owner) + collect(DISTINCT down_owner) AS related_owners
            UNWIND CASE
                WHEN owner IS NULL THEN [NULL]
                ELSE [owner] + related_owners
            END AS candidate_owner
            OPTIONAL MATCH (candidate_owner)-[:HAS_METHOD|HAS_CONSTRUCTOR]->(candidate_method)
            WHERE candidate_method.name = {alias}.name
              AND ($parameter_suffix IS NULL
                   OR candidate_method.id ENDS WITH (candidate_method.name + $parameter_suffix))
            WITH {alias}, collect(DISTINCT candidate_method) + [{alias}] AS all_methods
            UNWIND all_methods AS method
            WITH DISTINCT method, {alias}
            WHERE method IS NOT NULL
        """

    async def _resolve_file_target(
        self,
        file_path: str,
        repository: str | None = None,
    ) -> dict | None:
        """Resolve one file by suffix and optional repository."""
        filters = ["f.file_path ENDS WITH $file_path"]
        params: dict[str, Any] = {"file_path": file_path}
        if repository:
            filters.append("f.repository = $repository")
            params["repository"] = repository

        cypher = f"""
            MATCH (f:File)
            WHERE {" AND ".join(filters)}
            RETURN f.name AS name, f.file_path AS file_path,
                   f.repository AS repository, f.language AS language
            ORDER BY f.file_path
            LIMIT 2
        """

        results = await self._query(cypher, **params)
        if not results:
            return None
        if len(results) > 1:
            examples = ", ".join(r.get("file_path") or "?" for r in results)
            if repository:
                hint = (
                    "Provide a longer path suffix or entity_id= to disambiguate"
                )
            else:
                hint = (
                    "Provide repository= or a longer path suffix to disambiguate"
                )
            raise ValueError(
                f"File '{file_path}' is ambiguous — matches: {examples}. {hint}."
            )
        return results[0]

    async def search_code(
        self,
        query: str,
        *,
        limit: int = 10,
        entity_type: str | None = None,
        file_pattern: str | None = None,
        repository: str | None = None,
        code_mode: str = "preview",
        language: str | None = None,
        stereotype: str | None = None,
    ) -> list[CodeEntity]:
        """Semantic search for code using vector similarity."""
        limit = self._clamp_result_limit(limit, maximum=20)
        embedding = await self._get_embedding(query)

        if entity_type:
            indexes_to_query = [
                (
                    entity_type,
                    ENTITY_TYPE_TO_INDEX.get(entity_type, "vector_method_embedding"),
                )
            ]
        else:
            indexes_to_query = list(ENTITY_TYPE_TO_INDEX.items())

        all_results = []
        for etype, index_name in indexes_to_query:
            results = await self._search_index(
                entity_type=etype,
                index_name=index_name,
                limit=limit,
                embedding=embedding,
                repository=repository,
                file_pattern=file_pattern,
                language=language,
                stereotype=stereotype,
            )
            all_results.extend(results)

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = all_results[:limit]

        entities: list[CodeEntity] = [
            self._entity_from_record(record) for record in results
        ]
        self._apply_code_mode(entities, code_mode)

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
                code_mode=code_mode,
            )
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
                    code_mode=code_mode,
                )
            # find_symbols(code_mode=code_mode) above has already applied the
            # requested mode to symbol_results; no further transformation
            # needed before merging.

            merged: list[CodeEntity] = []
            seen: set[tuple[str | None, str, str, str]] = set()
            for entity in symbol_results + entities:
                key = (
                    entity.entity_id,
                    entity.entity_type,
                    entity.file_path,
                    entity.name,
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(entity)
                if len(merged) >= limit:
                    break
            return merged

        return entities

    async def get_callers(
        self,
        method_name: str,
        *,
        repository: str | None = None,
        file_path: str | None = None,
        entity_id: str | None = None,
        depth: int = 1,
        limit: int = 50,
    ) -> list[CallGraphNode]:
        """Find callers of a method using the current Constellation graph schema."""
        depth = min(depth, 3)
        limit = self._clamp_result_limit(limit)
        match_clause, params = self._build_method_match(
            "m",
            method_name=method_name,
            repository=repository,
            file_path=file_path,
            entity_id=entity_id,
        )
        params["parameter_suffix"] = self._parameter_suffix_from_entity_id(entity_id)

        cypher = f"""
            {match_clause}
            {self._method_family_fragment("m")}
            MATCH path = (caller)-[:CALLS*1..{depth}]->(method)
            WHERE caller:Method OR caller:Constructor
            WITH caller, min(length(path)) AS depth
            RETURN DISTINCT caller.id AS entity_id,
                   caller.name AS name, caller.file_path AS file_path,
                   caller.repository AS repository,
                   caller.signature AS signature, caller.line_number AS line_number,
                   head(labels(caller)) AS entity_type, 'CALLS' AS relationship_type,
                   depth
            ORDER BY depth, caller.file_path, caller.line_number
            LIMIT $query_limit
        """

        results = await self._query(cypher, query_limit=limit + 1, **params)
        truncated = len(results) > limit
        results = results[:limit]

        return [
            CallGraphNode(
                name=r["name"],
                file_path=r["file_path"] or "",
                repository=r.get("repository"),
                entity_id=r.get("entity_id"),
                signature=r.get("signature"),
                line_start=r.get("line_number"),
                depth=r.get("depth", 1),
                entity_type=self._label_to_entity_type(r.get("entity_type")),
                relationship_type=r.get("relationship_type", "CALLS"),
                truncated=truncated,
            )
            for r in results
        ]

    async def get_callees(
        self,
        method_name: str,
        *,
        repository: str | None = None,
        file_path: str | None = None,
        entity_id: str | None = None,
        depth: int = 1,
        limit: int = 50,
    ) -> list[CallGraphNode]:
        """Find call targets and hook usage for a method."""
        depth = min(depth, 3)
        limit = self._clamp_result_limit(limit)
        match_clause, params = self._build_method_match(
            "m",
            method_name=method_name,
            repository=repository,
            file_path=file_path,
            entity_id=entity_id,
        )
        params["parameter_suffix"] = self._parameter_suffix_from_entity_id(entity_id)

        calls_cypher = f"""
            {match_clause}
            {self._method_family_fragment("m")}
            MATCH path = (method)-[:CALLS*1..{depth}]->(callee)
            WHERE callee:Method OR callee:Constructor OR callee:Reference
            WITH callee, min(length(path)) AS depth
            RETURN DISTINCT callee.id AS entity_id,
                   callee.name AS name, callee.file_path AS file_path,
                   callee.repository AS repository,
                   callee.signature AS signature, callee.line_number AS line_number,
                   head(labels(callee)) AS entity_type, 'CALLS' AS relationship_type,
                   depth
            ORDER BY depth, callee.file_path, callee.line_number
            LIMIT $query_limit
        """

        hooks_cypher = f"""
            {match_clause}
            {self._method_family_fragment("m")}
            MATCH (method)-[:USES_HOOK]->(hook:Hook)
            RETURN DISTINCT hook.id AS entity_id,
                   hook.name AS name, hook.file_path AS file_path,
                   hook.repository AS repository,
                   hook.line_number AS line_number,
                   'Hook' AS entity_type, 'USES_HOOK' AS relationship_type,
                   1 AS depth
            ORDER BY hook.file_path, hook.line_number
            LIMIT $query_limit
        """

        call_results = await self._query(calls_cypher, query_limit=limit + 1, **params)
        hook_results = await self._query(hooks_cypher, query_limit=limit + 1, **params)
        combined: dict[tuple[str, str], dict] = {}
        for row in call_results + hook_results:
            key = (
                row.get("relationship_type", "CALLS"),
                row.get("name") or "",
                row.get("file_path") or "",
            )
            existing = combined.get(key)
            if existing is None or row.get("depth", 1) < existing.get("depth", 1):
                combined[key] = row
        results = sorted(
            combined.values(),
            key=lambda row: (
                row.get("depth", 1),
                row.get("file_path") or "",
                row.get("line_number") or 0,
                row.get("name") or "",
            ),
        )
        truncated = len(results) > limit or len(call_results) > limit or len(hook_results) > limit
        results = results[:limit]

        return [
            CallGraphNode(
                name=r["name"],
                file_path=r["file_path"] or "",
                repository=r.get("repository"),
                entity_id=r.get("entity_id"),
                signature=r.get("signature"),
                line_start=r.get("line_number"),
                depth=r.get("depth", 1),
                entity_type=self._label_to_entity_type(r.get("entity_type")),
                relationship_type=r.get("relationship_type", "CALLS"),
                truncated=truncated,
            )
            for r in results
        ]

    async def get_function_context(
        self,
        method_name: str,
        *,
        repository: str | None = None,
        file_path: str | None = None,
        entity_id: str | None = None,
    ) -> FunctionContext | None:
        """Get full context for a function: code, callers, callees, class."""
        target = await self._resolve_method_target(
            method_name,
            repository=repository,
            file_path=file_path,
            entity_id=entity_id,
        )
        if not target:
            return None

        entity_id = target.get("id") or ""
        full_name = self._full_name_from_id(entity_id, target["name"])

        callers = await self.get_callers(
            method_name,
            repository=repository,
            file_path=file_path,
            depth=1,
            entity_id=entity_id,
        )
        callees = await self.get_callees(
            method_name,
            repository=repository,
            file_path=file_path,
            depth=1,
            entity_id=entity_id,
        )

        return FunctionContext(
            name=target["name"],
            full_name=full_name,
            file_path=target["file_path"] or "",
            repository=target.get("repository"),
            code=target.get("code"),
            signature=target.get("signature"),
            docstring=target.get("docstring"),
            class_name=target.get("class_name"),
            callers=callers,
            callees=callees,
        )

    async def get_class_hierarchy(
        self,
        class_name: str,
        *,
        repository: str | None = None,
        file_path: str | None = None,
    ) -> ClassHierarchy | None:
        """Get class/interface inheritance hierarchy."""
        match_clause, params = self._build_class_match(
            "c",
            class_name=class_name,
            repository=repository,
            file_path=file_path,
        )
        cypher = f"""
            {match_clause}
            OPTIONAL MATCH (c)-[parent_rel]->(parent)
            WHERE type(parent_rel) = 'EXTENDS'
            OPTIONAL MATCH (child)-[child_rel]->(c)
            WHERE type(child_rel) = 'EXTENDS'
            OPTIONAL MATCH (c)-[impl_rel]->(iface)
            WHERE type(impl_rel) = 'IMPLEMENTS'
            OPTIONAL MATCH (impl)-[implemented_by_rel]->(c)
            WHERE type(implemented_by_rel) = 'IMPLEMENTS'
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (c)-[:HAS_FIELD]->(f:Field)
            OPTIONAL MATCH (c)-[:HAS_CONSTRUCTOR]->(ctor:Constructor)
            RETURN c.name AS name, c.id AS id, c.file_path AS file_path,
                   c.line_number AS line_number, c.repository AS repository, labels(c) AS labels,
                   collect(DISTINCT parent.name) AS parents,
                   collect(DISTINCT child.name) AS children,
                   collect(DISTINCT iface.name) AS interfaces,
                   collect(DISTINCT impl.name) AS implementors,
                   collect(DISTINCT m.name) AS methods,
                   collect(DISTINCT f.name) AS fields,
                   collect(DISTINCT ctor.name) AS constructors
            ORDER BY file_path, line_number
            LIMIT 2
        """

        results = await self._query(cypher, **params)
        if not results:
            return None
        if len(results) > 1:
            examples = ", ".join(r.get("file_path") or "?" for r in results)
            raise ValueError(
                f"Class '{class_name}' is ambiguous for the current filters. Matching entries: {examples}"
            )

        r = results[0]
        entity_id = r.get("id") or ""
        full_name = self._full_name_from_id(entity_id, r["name"])
        node_labels = r.get("labels") or []
        is_interface = "Interface" in node_labels

        return ClassHierarchy(
            name=r["name"],
            full_name=full_name,
            file_path=r["file_path"] or "",
            repository=r.get("repository"),
            is_interface=is_interface,
            parents=[p for p in r.get("parents", []) if p],
            children=[c for c in r.get("children", []) if c],
            interfaces=[i for i in r.get("interfaces", []) if i],
            implementors=[i for i in r.get("implementors", []) if i],
            methods=[m for m in r.get("methods", []) if m],
            fields=[f for f in r.get("fields", []) if f],
            constructors=[c for c in r.get("constructors", []) if c],
        )

    async def list_repositories(self) -> list[dict]:
        """List all indexed repositories by querying Repository nodes."""
        cypher = """
            MATCH (r:Repository)
            RETURN r.name AS name, r.entity_count AS entity_count,
                   r.last_indexed_at AS last_indexed_at
            ORDER BY r.name
        """
        return await self._query(cypher)

    async def get_repository_context(self, repository: str) -> RepositoryContext | None:
        """Fetch repository metadata and aggregate graph statistics."""
        cypher = """
            MATCH (r:Repository {name: $repository})
            RETURN r.name AS name, r.source AS source,
                   r.entity_count AS entity_count,
                   r.last_indexed_at AS last_indexed_at,
                   r.last_commit_sha AS last_commit_sha
        """
        rows = await self._query(cypher, repository=repository)
        if not rows:
            return None

        row = rows[0]
        overview = await self.get_codebase_overview(repository=repository)
        return RepositoryContext(
            name=row["name"],
            source=row.get("source"),
            entity_count=row.get("entity_count", 0),
            last_indexed_at=row.get("last_indexed_at"),
            last_commit_sha=row.get("last_commit_sha"),
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

    async def get_package_context(
        self,
        package_name: str,
        *,
        repository: str | None = None,
    ) -> PackageContext | None:
        """Return package or namespace membership information."""
        filters = ["(pkg.name = $package_name OR pkg.id ENDS WITH $package_id_suffix)"]
        params: dict[str, Any] = {
            "package_name": package_name,
            "package_id_suffix": f"::{package_name}",
        }
        if repository:
            filters.append("pkg.repository = $repository")
            params["repository"] = repository

        cypher = f"""
            MATCH (pkg:Package)
            WHERE {" AND ".join(filters)}
            OPTIONAL MATCH (member)-[:IN_PACKAGE]->(pkg)
            WITH pkg, collect(DISTINCT member.file_path) AS files
            OPTIONAL MATCH (class:Class)-[:IN_PACKAGE]->(pkg)
            WITH pkg, files, collect(DISTINCT class.name) AS classes
            OPTIONAL MATCH (iface:Interface)-[:IN_PACKAGE]->(pkg)
            WITH pkg, files, classes, collect(DISTINCT iface.name) AS interfaces
            OPTIONAL MATCH (method:Method)-[:IN_PACKAGE]->(pkg)
            WITH pkg, files, classes, interfaces, collect(DISTINCT method.name) AS methods
            OPTIONAL MATCH (hook:Hook)-[:IN_PACKAGE]->(pkg)
            WITH pkg, files, classes, interfaces, methods, collect(DISTINCT hook.name) AS hooks
            OPTIONAL MATCH (ref:Reference)-[:IN_PACKAGE]->(pkg)
            WITH pkg, files, classes, interfaces, methods, hooks,
                 collect(DISTINCT ref.name) AS references
            OPTIONAL MATCH (child:Package {{repository: pkg.repository}})
            WHERE child.id STARTS WITH pkg.id + '.'
            RETURN pkg.name AS name, pkg.id AS id, pkg.repository AS repository,
                   files, classes, interfaces, methods, hooks, references,
                   collect(DISTINCT {{id: child.id, name: child.name}}) AS child_packages
            ORDER BY pkg.repository, pkg.id
            LIMIT 2
        """
        rows = await self._query(cypher, **params)
        if not rows:
            return None
        if len(rows) > 1:
            examples = ", ".join(
                self._full_name_from_id(row.get("id") or "", row.get("name") or "?")
                for row in rows
            )
            raise ValueError(
                f"Package '{package_name}' is ambiguous for the current filters. Matching entries: {examples}"
            )

        row = rows[0]
        full_name = self._full_name_from_id(row.get("id") or "", row.get("name") or package_name)
        base_segments = full_name.split(".")
        child_packages = []
        for child in row.get("child_packages", []):
            if isinstance(child, str):
                child_name = child
            else:
                child_name = self._full_name_from_id(child.get("id") or "", child.get("name") or "")
            if not child_name.startswith(full_name + "."):
                continue
            if len(child_name.split(".")) != len(base_segments) + 1:
                continue
            child_packages.append(child_name)

        return PackageContext(
            name=full_name,
            repository=row.get("repository"),
            package_id=row.get("id"),
            files=[value for value in row.get("files", []) if value],
            classes=[value for value in row.get("classes", []) if value],
            interfaces=[value for value in row.get("interfaces", []) if value],
            methods=[value for value in row.get("methods", []) if value],
            hooks=[value for value in row.get("hooks", []) if value],
            references=[value for value in row.get("references", []) if value],
            child_packages=sorted(set(child_packages)),
        )

    async def get_codebase_overview(
        self,
        repository: str | None = None,
        include_packages: bool = False,
    ) -> CodebaseOverview:
        """Get high-level codebase statistics."""
        repo_filter_f = "WHERE f.repository = $repository" if repository else ""
        repo_filter_c = "WHERE c.repository = $repository" if repository else ""
        repo_filter_i = "WHERE i.repository = $repository" if repository else ""
        repo_filter_m = "WHERE m.repository = $repository" if repository else ""
        repo_filter_ctor = "WHERE ctor.repository = $repository" if repository else ""
        repo_filter_field = "WHERE field.repository = $repository" if repository else ""
        repo_filter_hook = "WHERE hook.repository = $repository" if repository else ""
        repo_filter_ref = "WHERE ref.repository = $repository" if repository else ""
        repo_filter_export = "WHERE f.repository = $repository" if repository else ""

        if include_packages:
            repo_filter_pkg = "WHERE pkg.repository = $repository" if repository else ""
            overview_cypher = f"""
                MATCH (f:File) {repo_filter_f}
                WITH count(f) AS files, collect(DISTINCT f.language) AS languages
                OPTIONAL MATCH (c:Class) {repo_filter_c}
                WITH files, languages, count(c) AS classes
                OPTIONAL MATCH (i:Interface) {repo_filter_i}
                WITH files, languages, classes, count(i) AS interfaces
                OPTIONAL MATCH (m:Method) {repo_filter_m}
                WITH files, languages, classes, interfaces, count(m) AS methods
                OPTIONAL MATCH (ctor:Constructor) {repo_filter_ctor}
                WITH files, languages, classes, interfaces, methods, count(ctor) AS constructors
                OPTIONAL MATCH (field:Field) {repo_filter_field}
                WITH files, languages, classes, interfaces, methods, constructors, count(field) AS fields
                OPTIONAL MATCH (pkg:Package) {repo_filter_pkg}
                WITH files, languages, classes, interfaces, methods, constructors, fields,
                     count(pkg) AS packages_count, collect(DISTINCT {{id: pkg.id, name: pkg.name}}) AS packages
                OPTIONAL MATCH (hook:Hook) {repo_filter_hook}
                WITH files, languages, classes, interfaces, methods, constructors, fields,
                     packages_count, packages, count(hook) AS hooks
                OPTIONAL MATCH (ref:Reference) {repo_filter_ref}
                WITH files, languages, classes, interfaces, methods, constructors, fields,
                     packages_count, packages, hooks, count(ref) AS references
                OPTIONAL MATCH (f:File)-[:EXPORTS]->(exported) {repo_filter_export}
                RETURN files, classes, interfaces, methods, constructors, fields,
                       packages_count, hooks, references, count(exported) AS exports,
                       languages, packages
            """
        else:
            overview_cypher = f"""
                MATCH (f:File) {repo_filter_f}
                WITH count(f) AS files, collect(DISTINCT f.language) AS languages
                OPTIONAL MATCH (c:Class) {repo_filter_c}
                WITH files, languages, count(c) AS classes
                OPTIONAL MATCH (i:Interface) {repo_filter_i}
                WITH files, languages, classes, count(i) AS interfaces
                OPTIONAL MATCH (m:Method) {repo_filter_m}
                WITH files, languages, classes, interfaces, count(m) AS methods
                OPTIONAL MATCH (ctor:Constructor) {repo_filter_ctor}
                WITH files, languages, classes, interfaces, methods, count(ctor) AS constructors
                OPTIONAL MATCH (field:Field) {repo_filter_field}
                WITH files, languages, classes, interfaces, methods, constructors, count(field) AS fields
                OPTIONAL MATCH (pkg:Package) {repo_filter_f.replace('f.', 'pkg.')}
                WITH files, languages, classes, interfaces, methods, constructors, fields,
                     count(pkg) AS packages_count
                OPTIONAL MATCH (hook:Hook) {repo_filter_hook}
                WITH files, languages, classes, interfaces, methods, constructors, fields,
                     packages_count, count(hook) AS hooks
                OPTIONAL MATCH (ref:Reference) {repo_filter_ref}
                WITH files, languages, classes, interfaces, methods, constructors, fields,
                     packages_count, hooks, count(ref) AS references
                OPTIONAL MATCH (f:File)-[:EXPORTS]->(exported) {repo_filter_export}
                RETURN files, classes, interfaces, methods, constructors, fields,
                       packages_count, hooks, references, count(exported) AS exports,
                       languages
            """

        results = await self._query(overview_cypher, repository=repository)
        if not results:
            return CodebaseOverview()

        r = results[0]

        top_classes_filter = "WHERE c.repository = $repository AND" if repository else "WHERE"
        top_classes_cypher = f"""
            MATCH (c:Class)
            {top_classes_filter} NOT (c)-[:EXTENDS]->(:Class)
              AND NOT 'test' IN coalesce(c.stereotypes, [])
            OPTIONAL MATCH (child:Class)-[:EXTENDS*]->(c)
            WITH c, count(child) AS descendants
            ORDER BY descendants DESC
            LIMIT 10
            RETURN c.name AS name
        """
        top_classes_results = await self._query(top_classes_cypher, repository=repository)
        top_level_classes = [row["name"] for row in top_classes_results if row.get("name")]

        entry_filter = "WHERE m.repository = $repository AND" if repository else "WHERE"
        entry_points_cypher = f"""
            MATCH (m:Method)
            {entry_filter} ('endpoint' IN m.stereotypes OR m.name = 'main')
              AND NOT 'test' IN coalesce(m.stereotypes, [])
            OPTIONAL MATCH (c:Class)-[:HAS_METHOD]->(m)
            RETURN m.name AS name, c.name AS class_name
            LIMIT 10
        """
        entry_results = await self._query(entry_points_cypher, repository=repository)

        def format_entry_point(entry: dict) -> str:
            class_name = entry.get("class_name") or ""
            method_name = entry.get("name", "")
            if class_name:
                return f"{class_name}.{method_name}"
            return method_name

        entry_points = [format_entry_point(entry) for entry in entry_results]

        raw_packages = r.get("packages", []) or []
        packages = [
            self._full_name_from_id(pkg["id"], pkg["name"])
            for pkg in raw_packages
            if pkg and pkg.get("id")
        ] if include_packages else []

        return CodebaseOverview(
            total_files=r.get("files", 0),
            total_classes=r.get("classes", 0),
            total_interfaces=r.get("interfaces", 0),
            total_methods=r.get("methods", 0),
            total_constructors=r.get("constructors", 0),
            total_fields=r.get("fields", 0),
            total_packages=r.get("packages_count", 0),
            total_hooks=r.get("hooks", 0),
            total_references=r.get("references", 0),
            total_exports=r.get("exports", 0),
            languages=[lang for lang in r.get("languages", []) if lang],
            packages=packages,
            top_level_classes=top_level_classes,
            entry_points=entry_points,
        )

    async def find_symbols(
        self,
        query: str,
        *,
        entity_types: list[str] | None = None,
        repository: str | None = None,
        file_pattern: str | None = None,
        limit: int = 20,
        exact: bool = False,
        language: str | None = None,
        stereotype: str | None = None,
        code_mode: str = "none",
    ) -> list[CodeEntity]:
        """Find graph entities by exact or substring symbol match."""
        limit = self._clamp_result_limit(limit, maximum=50)
        labels = [
            SYMBOL_ENTITY_TYPE_TO_LABEL[entity_type]
            for entity_type in (entity_types or list(SYMBOL_ENTITY_TYPE_TO_LABEL))
        ]
        filter_clause, filter_params = self._build_search_filters(
            "n",
            repository=repository,
            file_pattern=file_pattern,
            language=language,
            stereotype=stereotype,
        )
        name_clause = "toLower(n.name) = toLower($search_query)"
        if not exact:
            name_clause = (
                "(toLower(n.name) CONTAINS toLower($search_query) "
                "OR toLower(n.file_path) CONTAINS toLower($search_query))"
            )

        cypher = f"""
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN $labels)
              AND {name_clause}{filter_clause}
            RETURN n.id AS id, n.name AS name, n.file_path AS file_path,
                   n.repository AS repository, n.line_number AS line_number,
                   n.line_end AS line_end, n.code AS code,
                   n.signature AS signature, n.language AS language,
                   n.return_type AS return_type, n.modifiers AS modifiers,
                   n.stereotypes AS stereotypes, n.content_hash AS content_hash,
                   properties(n) AS properties,
                   head(labels(n)) AS entity_type
            ORDER BY CASE
                         WHEN toLower(n.name) = toLower($search_query) THEN 0
                         ELSE 1
                     END,
                     n.file_path, n.line_number
            LIMIT $limit
        """
        results = await self._query(
            cypher,
            labels=labels,
            search_query=query,
            limit=limit,
            **filter_params,
        )
        entities = [self._entity_from_record(row) for row in results]
        self._apply_code_mode(entities, code_mode)
        return entities

    async def get_file_context(
        self,
        file_path: str,
        *,
        repository: str | None = None,
    ) -> FileContext | None:
        """Get file-level graph context, including packages, exports, and hook usage."""
        target = await self._resolve_file_target(file_path, repository=repository)
        if not target:
            return None

        cypher = """
            MATCH (f:File {repository: $repository, file_path: $file_path})
            OPTIONAL MATCH (member)-[:IN_PACKAGE]->(pkg:Package)
            WHERE member.repository = f.repository AND member.file_path = f.file_path
            WITH f, collect(DISTINCT {id: pkg.id, name: pkg.name}) AS packages
            OPTIONAL MATCH (cls:Class)
            WHERE cls.repository = f.repository AND cls.file_path = f.file_path
            WITH f, packages, collect(DISTINCT cls.name) AS classes
            OPTIONAL MATCH (iface:Interface)
            WHERE iface.repository = f.repository AND iface.file_path = f.file_path
            WITH f, packages, classes, collect(DISTINCT iface.name) AS interfaces
            OPTIONAL MATCH (f)-[:CONTAINS]->(method:Method)
            WITH f, packages, classes, interfaces, collect(DISTINCT method.name) AS top_level_methods
            OPTIONAL MATCH (ctor:Constructor)
            WHERE ctor.repository = f.repository AND ctor.file_path = f.file_path
            WITH f, packages, classes, interfaces, top_level_methods,
                 collect(DISTINCT ctor.name) AS constructors
            OPTIONAL MATCH (field:Field)
            WHERE field.repository = f.repository AND field.file_path = f.file_path
            WITH f, packages, classes, interfaces, top_level_methods, constructors,
                 collect(DISTINCT field.name) AS fields
            OPTIONAL MATCH (ref:Reference)
            WHERE ref.repository = f.repository AND ref.file_path = f.file_path
            WITH f, packages, classes, interfaces, top_level_methods, constructors, fields,
                 collect(DISTINCT ref.name) AS references
            OPTIONAL MATCH (f)-[export_rel:EXPORTS]->(exported)
            WITH f, packages, classes, interfaces, top_level_methods, constructors, fields, references,
                 collect(DISTINCT {
                     id: exported.id,
                     name: exported.name,
                     file_path: exported.file_path,
                     repository: exported.repository,
                     line_start: exported.line_number,
                     line_end: exported.line_end,
                     entity_type: toLower(head(labels(exported))),
                     language: exported.language,
                     content_hash: exported.content_hash,
                     entity_properties: properties(exported),
                     relationship_properties: properties(export_rel)
                 }) AS exports
            OPTIONAL MATCH (hook:Hook)
            WHERE hook.repository = f.repository AND hook.file_path = f.file_path
            RETURN f.name AS name, f.file_path AS file_path,
                   f.repository AS repository, f.language AS language,
                   f.content_hash AS content_hash,
                   packages, classes, interfaces, top_level_methods,
                   constructors, fields, references,
                   collect(DISTINCT hook.name) AS hooks, exports
        """
        results = await self._query(
            cypher,
            repository=target["repository"],
            file_path=target["file_path"],
        )
        if not results:
            return None

        row = results[0]
        exports = [
            CodeEntity(
                name=export["name"],
                file_path=export.get("file_path") or "",
                repository=export.get("repository"),
                entity_id=export.get("id"),
                line_start=export.get("line_start"),
                line_end=export.get("line_end"),
                entity_type=export.get("entity_type") or "class",
                language=export.get("language"),
                content_hash=export.get("content_hash"),
                properties={
                    **self._custom_entity_properties(export.get("properties")),
                    **self._custom_entity_properties(export.get("entity_properties")),
                    **(export.get("relationship_properties") or {}),
                },
            )
            for export in row.get("exports", [])
            if export.get("name")
        ]
        raw_packages = row.get("packages", []) or []
        packages = [
            self._full_name_from_id(pkg["id"], pkg["name"])
            for pkg in raw_packages
            if pkg and pkg.get("id")
        ]
        return FileContext(
            name=row["name"],
            file_path=row["file_path"] or "",
            repository=row.get("repository"),
            language=row.get("language"),
            content_hash=row.get("content_hash"),
            packages=packages,
            exports=exports,
            classes=[name for name in row.get("classes", []) if name],
            interfaces=[name for name in row.get("interfaces", []) if name],
            top_level_methods=[name for name in row.get("top_level_methods", []) if name],
            hooks=[name for name in row.get("hooks", []) if name],
            constructors=[name for name in row.get("constructors", []) if name],
            fields=[name for name in row.get("fields", []) if name],
            references=[name for name in row.get("references", []) if name],
        )

    async def get_hook_usage(
        self,
        hook_name: str,
        *,
        repository: str | None = None,
        file_pattern: str | None = None,
        language: str | None = None,
        stereotype: str | None = None,
        limit: int = 50,
    ) -> list[CallGraphNode]:
        """Find methods and constructors that use a materialized hook node."""
        limit = self._clamp_result_limit(limit)
        filter_clause, filter_params = self._build_search_filters(
            "m",
            repository=repository,
            file_pattern=file_pattern,
            language=language,
            stereotype=stereotype,
        )
        cypher = f"""
            MATCH (h:Hook)
            WHERE h.name = $hook_name
            MATCH (m)-[:USES_HOOK]->(h)
            WHERE (m:Method OR m:Constructor){filter_clause}
            RETURN m.id AS entity_id,
                   m.name AS name, m.file_path AS file_path,
                   m.repository AS repository,
                   m.signature AS signature, m.line_number AS line_number,
                   head(labels(m)) AS entity_type, 'USES_HOOK' AS relationship_type
            ORDER BY m.file_path, m.line_number
            LIMIT $query_limit
        """
        results = await self._query(cypher, hook_name=hook_name, query_limit=limit + 1, **filter_params)
        truncated = len(results) > limit
        results = results[:limit]
        return [
            CallGraphNode(
                name=row["name"],
                file_path=row["file_path"] or "",
                repository=row.get("repository"),
                entity_id=row.get("entity_id"),
                signature=row.get("signature"),
                line_start=row.get("line_number"),
                entity_type=self._label_to_entity_type(row.get("entity_type")),
                relationship_type=row.get("relationship_type", "USES_HOOK"),
                truncated=truncated,
            )
            for row in results
        ]

    async def get_impact(
        self,
        method_name: str,
        *,
        repository: str | None = None,
        file_path: str | None = None,
        entity_id: str | None = None,
        depth: int = 10,
        summary_only: bool = False,
        limit: int | None = None,
    ) -> ImpactResult | None:
        """Analyze blast radius of changing a method."""
        target = await self._resolve_method_target(
            method_name,
            repository=repository,
            file_path=file_path,
            entity_id=entity_id,
        )
        if not target:
            return None

        callers_cypher = f"""
            MATCH (m)
            WHERE (m:Method OR m:Constructor) AND m.id = $entity_id
            {self._method_family_fragment("m")}
            MATCH path = (caller)-[:CALLS*1..{depth}]->(method)
            WHERE (caller:Method OR caller:Constructor) AND caller <> m AND caller <> method
            WITH caller, min(length(path)) AS depth
            RETURN caller.id AS entity_id,
                   caller.name AS name, caller.file_path AS file_path,
                   caller.repository AS repository,
                   caller.signature AS signature, caller.line_number AS line_number,
                   caller.stereotypes AS stereotypes,
                   depth
            ORDER BY depth, caller.file_path, caller.line_number
            LIMIT 200
        """
        caller_results = await self._query(
            callers_cypher,
            entity_id=target["id"],
            parameter_suffix=self._parameter_suffix_from_entity_id(target.get("id")),
        )

        tests = []
        endpoints = []
        others = []

        for r in caller_results:
            stereotypes = r.get("stereotypes") or []
            is_test = "test" in stereotypes
            is_endpoint = "endpoint" in stereotypes

            if not is_test and not is_endpoint:
                name_lower = (r["name"] or "").lower()
                path_lower = (r["file_path"] or "").lower()
                if "test" in name_lower or "test" in path_lower:
                    is_test = True
                elif any(x in name_lower for x in ["controller", "endpoint", "handler", "api"]):
                    is_endpoint = True

            node = CallGraphNode(
                name=r["name"],
                file_path=r["file_path"] or "",
                repository=r.get("repository"),
                entity_id=r.get("entity_id"),
                signature=r.get("signature"),
                line_start=r.get("line_number"),
                depth=r.get("depth", 1),
                is_test=is_test,
                is_endpoint=is_endpoint,
            )

            if is_test:
                tests.append(node)
            elif is_endpoint:
                endpoints.append(node)
            else:
                others.append(node)

        total_tests = len(tests)
        total_endpoints = len(endpoints)

        truncated = False
        if limit:
            truncated = len(tests) > limit or len(endpoints) > limit or len(others) > limit
            tests = tests[:limit]
            endpoints = endpoints[:limit]
            others = others[:limit]

        if summary_only:
            tests = []
            endpoints = []
            others = []

        return ImpactResult(
            target_name=target["name"],
            target_file=target["file_path"] or "",
            target_repository=target.get("repository"),
            total_callers=len(caller_results),
            test_count=total_tests,
            endpoint_count=total_endpoints,
            affected_tests=tests,
            affected_endpoints=endpoints,
            other_callers=others,
            truncated=truncated,
        )
