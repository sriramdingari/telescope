"""Neo4j graph client for querying Constellation's code knowledge graph."""

import logging
import re
from typing import Any

from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

from .config import get_config
from .models import (
    CallGraphNode,
    ClassHierarchy,
    CodebaseOverview,
    CodeEntity,
    FileContext,
    FunctionContext,
    ImpactResult,
)

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


class GraphClient:
    """Client for querying Constellation's code knowledge graph."""

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
            return {k: GraphClient._normalize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [GraphClient._normalize_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(GraphClient._normalize_value(v) for v in value)

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

        clause = ""
        if filters:
            clause = " AND " + " AND ".join(filters)

        return clause, params

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
    ) -> list[dict]:
        """Search a single entity type, using exact filtered scoring when available."""
        label = ENTITY_TYPE_TO_LABEL[entity_type]
        filter_clause, filter_params = self._build_search_filters(
            "n",
            repository=repository,
            file_pattern=file_pattern,
        )

        return_clause = f"""
            RETURN n.name AS name, n.file_path AS file_path,
                   n.repository AS repository,
                   n.line_number AS line_number, n.line_end AS line_end,
                   n.code AS code, n.signature AS signature,
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
    ) -> dict | None:
        """Resolve one method/constructor for exact-context queries."""
        match_clause, params = self._build_method_match(
            "m",
            method_name=method_name,
            repository=repository,
            file_path=file_path,
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
    def _label_to_entity_type(label: str | None) -> str:
        """Normalize a Neo4j label into Telescope's lowercase entity type names."""
        if not label:
            return "method"
        return label.lower()

    def _method_family_fragment(self, alias: str) -> str:
        """Build a method-family expansion based on class/interface relationships."""
        return f"""
            OPTIONAL MATCH (owner)-[:HAS_METHOD|HAS_CONSTRUCTOR]->({alias})
            WHERE owner:Class OR owner:Interface
            OPTIONAL MATCH (owner)-[:IMPLEMENTS|EXTENDS*1..3]->(up_owner)
            WHERE up_owner:Class OR up_owner:Interface
            OPTIONAL MATCH (down_owner)-[:IMPLEMENTS|EXTENDS*1..3]->(owner)
            WHERE down_owner:Class OR down_owner:Interface
            WITH {alias}, owner,
                 collect(DISTINCT up_owner) + collect(DISTINCT down_owner) AS related_owners
            UNWIND CASE
                WHEN owner IS NULL THEN [NULL]
                ELSE [owner] + related_owners
            END AS candidate_owner
            OPTIONAL MATCH (candidate_owner)-[:HAS_METHOD|HAS_CONSTRUCTOR]->(candidate_method)
            WHERE candidate_method.name = {alias}.name
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
            raise ValueError(
                f"File '{file_path}' is ambiguous for the current filters. Matching entries: {examples}"
            )
        return results[0]

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        entity_type: str | None = None,
        file_pattern: str | None = None,
        repository: str | None = None,
        code_mode: str = "preview",
    ) -> list[CodeEntity]:
        """Semantic search for code using vector similarity."""
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
            )
            all_results.extend(results)

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = all_results[:limit]

        def process_code(code: str | None, signature: str | None) -> str | None:
            if code_mode == "none":
                return None
            if code_mode == "signature":
                return signature
            if code_mode == "preview":
                if not code:
                    return signature
                lines = code.split("\n")
                if len(lines) <= 10:
                    return code
                return "\n".join(lines[:10]) + "\n... (truncated)"
            return code

        return [
            CodeEntity(
                name=r["name"],
                file_path=r["file_path"] or "",
                repository=r.get("repository"),
                line_start=r.get("line_number"),
                line_end=r.get("line_end"),
                code=process_code(r.get("code"), r.get("signature")),
                signature=r.get("signature"),
                score=r.get("score", 0),
                entity_type=r.get("entity_type", "method"),
            )
            for r in results
        ]

    async def get_callers(
        self,
        method_name: str,
        repository: str | None = None,
        file_path: str | None = None,
        depth: int = 1,
        entity_id: str | None = None,
    ) -> list[CallGraphNode]:
        """Find callers of a method using the current Constellation graph schema."""
        depth = min(depth, 3)
        match_clause, params = self._build_method_match(
            "m",
            method_name=method_name,
            repository=repository,
            file_path=file_path,
            entity_id=entity_id,
        )

        cypher = f"""
            {match_clause}
            {self._method_family_fragment("m")}
            MATCH (caller)-[:CALLS*1..{depth}]->(method)
            WHERE caller:Method OR caller:Constructor
            RETURN DISTINCT caller.name AS name, caller.file_path AS file_path,
                   caller.repository AS repository,
                   caller.signature AS signature, caller.line_number AS line_number,
                   head(labels(caller)) AS entity_type, 'CALLS' AS relationship_type
            ORDER BY caller.file_path, caller.line_number
            LIMIT 50
        """

        results = await self._query(cypher, **params)

        return [
            CallGraphNode(
                name=r["name"],
                file_path=r["file_path"] or "",
                repository=r.get("repository"),
                signature=r.get("signature"),
                line_start=r.get("line_number"),
                entity_type=self._label_to_entity_type(r.get("entity_type")),
                relationship_type=r.get("relationship_type", "CALLS"),
            )
            for r in results
        ]

    async def get_callees(
        self,
        method_name: str,
        repository: str | None = None,
        file_path: str | None = None,
        depth: int = 1,
        entity_id: str | None = None,
    ) -> list[CallGraphNode]:
        """Find call targets and hook usage for a method."""
        depth = min(depth, 3)
        match_clause, params = self._build_method_match(
            "m",
            method_name=method_name,
            repository=repository,
            file_path=file_path,
            entity_id=entity_id,
        )

        calls_cypher = f"""
            {match_clause}
            {self._method_family_fragment("m")}
            MATCH path = (method)-[:CALLS*1..{depth}]->(callee)
            WHERE callee:Method OR callee:Constructor OR callee:Reference
            WITH callee, min(length(path)) AS depth
            RETURN DISTINCT callee.name AS name, callee.file_path AS file_path,
                   callee.repository AS repository,
                   callee.signature AS signature, callee.line_number AS line_number,
                   head(labels(callee)) AS entity_type, 'CALLS' AS relationship_type,
                   depth
            ORDER BY depth, callee.file_path, callee.line_number
            LIMIT 50
        """

        hooks_cypher = f"""
            {match_clause}
            {self._method_family_fragment("m")}
            MATCH (method)-[:USES_HOOK]->(hook:Hook)
            RETURN DISTINCT hook.name AS name, hook.file_path AS file_path,
                   hook.repository AS repository,
                   hook.line_number AS line_number,
                   'Hook' AS entity_type, 'USES_HOOK' AS relationship_type,
                   1 AS depth
            ORDER BY hook.file_path, hook.line_number
            LIMIT 50
        """

        call_results = await self._query(calls_cypher, **params)
        hook_results = await self._query(hooks_cypher, **params)
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
        )[:50]

        return [
            CallGraphNode(
                name=r["name"],
                file_path=r["file_path"] or "",
                repository=r.get("repository"),
                signature=r.get("signature"),
                line_start=r.get("line_number"),
                depth=r.get("depth", 1),
                entity_type=self._label_to_entity_type(r.get("entity_type")),
                relationship_type=r.get("relationship_type", "CALLS"),
            )
            for r in results
        ]

    async def get_function_context(
        self,
        method_name: str,
        repository: str | None = None,
        file_path: str | None = None,
    ) -> FunctionContext | None:
        """Get full context for a function: code, callers, callees, class."""
        target = await self._resolve_method_target(
            method_name,
            repository=repository,
            file_path=file_path,
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
            OPTIONAL MATCH (c)-[:EXTENDS]->(parent)
            OPTIONAL MATCH (child)-[:EXTENDS]->(c)
            OPTIONAL MATCH (c)-[:IMPLEMENTS]->(iface)
            OPTIONAL MATCH (impl)-[:IMPLEMENTS]->(c)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (c)-[:HAS_FIELD]->(f:Field)
            OPTIONAL MATCH (c)-[:HAS_CONSTRUCTOR]->(ctor:Constructor)
            RETURN c.name AS name, c.id AS id, c.file_path AS file_path,
                   c.repository AS repository, labels(c) AS labels,
                   collect(DISTINCT parent.name) AS parents,
                   collect(DISTINCT child.name) AS children,
                   collect(DISTINCT iface.name) AS interfaces,
                   collect(DISTINCT impl.name) AS implementors,
                   collect(DISTINCT m.name) AS methods,
                   collect(DISTINCT f.name) AS fields,
                   collect(DISTINCT ctor.name) AS constructors
            ORDER BY c.file_path, c.line_number
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
                     count(pkg) AS packages_count, collect(DISTINCT pkg.name) AS packages
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
            packages=[p for p in r.get("packages", []) if p] if include_packages else [],
            top_level_classes=top_level_classes,
            entry_points=entry_points,
        )

    async def find_symbols(
        self,
        query: str,
        entity_types: list[str] | None = None,
        repository: str | None = None,
        file_pattern: str | None = None,
        limit: int = 20,
        exact: bool = False,
    ) -> list[CodeEntity]:
        """Find graph entities by exact or substring symbol match."""
        labels = [
            SYMBOL_ENTITY_TYPE_TO_LABEL[entity_type]
            for entity_type in (entity_types or list(SYMBOL_ENTITY_TYPE_TO_LABEL))
        ]
        filter_clause, filter_params = self._build_search_filters(
            "n",
            repository=repository,
            file_pattern=file_pattern,
        )
        name_clause = "toLower(n.name) = toLower($query)"
        if not exact:
            name_clause = "(toLower(n.name) CONTAINS toLower($query) OR toLower(n.file_path) CONTAINS toLower($query))"

        cypher = f"""
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN $labels)
              AND {name_clause}{filter_clause}
            RETURN n.name AS name, n.file_path AS file_path,
                   n.repository AS repository, n.line_number AS line_number,
                   n.line_end AS line_end, n.code AS code,
                   n.signature AS signature,
                   head(labels(n)) AS entity_type
            ORDER BY CASE
                         WHEN toLower(n.name) = toLower($query) THEN 0
                         ELSE 1
                     END,
                     n.file_path, n.line_number
            LIMIT $limit
        """
        results = await self._query(
            cypher,
            labels=labels,
            query=query,
            limit=limit,
            **filter_params,
        )
        return [
            CodeEntity(
                name=row["name"],
                file_path=row["file_path"] or "",
                repository=row.get("repository"),
                line_start=row.get("line_number"),
                line_end=row.get("line_end"),
                code=row.get("code"),
                signature=row.get("signature"),
                entity_type=self._label_to_entity_type(row.get("entity_type")),
            )
            for row in results
        ]

    async def get_file_context(
        self,
        file_path: str,
        repository: str | None = None,
    ) -> FileContext | None:
        """Get file-level graph context, including packages, exports, and hook usage."""
        target = await self._resolve_file_target(file_path, repository=repository)
        if not target:
            return None

        cypher = """
            MATCH (f:File {repository: $repository, file_path: $file_path})
            OPTIONAL MATCH (f)-[:IN_PACKAGE]->(pkg:Package)
            WITH f, collect(DISTINCT pkg.name) AS packages
            OPTIONAL MATCH (f)-[:CONTAINS]->(class:Class)
            WITH f, packages, collect(DISTINCT class.name) AS classes
            OPTIONAL MATCH (f)-[:CONTAINS]->(iface:Interface)
            WITH f, packages, classes, collect(DISTINCT iface.name) AS interfaces
            OPTIONAL MATCH (f)-[:CONTAINS]->(method:Method)
            WITH f, packages, classes, interfaces, collect(DISTINCT method.name) AS top_level_methods
            OPTIONAL MATCH (f)-[:EXPORTS]->(exported)
            WITH f, packages, classes, interfaces, top_level_methods,
                 collect(DISTINCT {
                     name: exported.name,
                     file_path: exported.file_path,
                     repository: exported.repository,
                     line_start: exported.line_number,
                     entity_type: toLower(head(labels(exported)))
                 }) AS exports
            OPTIONAL MATCH (caller)-[:USES_HOOK]->(hook:Hook)
            WHERE caller.repository = f.repository AND caller.file_path = f.file_path
            RETURN f.name AS name, f.file_path AS file_path,
                   f.repository AS repository, f.language AS language,
                   packages, classes, interfaces, top_level_methods,
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
                line_start=export.get("line_start"),
                entity_type=export.get("entity_type") or "class",
            )
            for export in row.get("exports", [])
            if export.get("name")
        ]
        return FileContext(
            name=row["name"],
            file_path=row["file_path"] or "",
            repository=row.get("repository"),
            language=row.get("language"),
            packages=[pkg for pkg in row.get("packages", []) if pkg],
            exports=exports,
            classes=[name for name in row.get("classes", []) if name],
            interfaces=[name for name in row.get("interfaces", []) if name],
            top_level_methods=[name for name in row.get("top_level_methods", []) if name],
            hooks=[name for name in row.get("hooks", []) if name],
        )

    async def get_hook_usage(
        self,
        hook_name: str,
        repository: str | None = None,
        file_pattern: str | None = None,
    ) -> list[CallGraphNode]:
        """Find methods and constructors that use a materialized hook node."""
        filter_clause, filter_params = self._build_search_filters(
            "m",
            repository=repository,
            file_pattern=file_pattern,
        )
        cypher = f"""
            MATCH (h:Hook)
            WHERE h.name = $hook_name
            MATCH (m)-[:USES_HOOK]->(h)
            WHERE (m:Method OR m:Constructor){filter_clause}
            RETURN m.name AS name, m.file_path AS file_path,
                   m.repository AS repository,
                   m.signature AS signature, m.line_number AS line_number,
                   head(labels(m)) AS entity_type, 'USES_HOOK' AS relationship_type
            ORDER BY m.file_path, m.line_number
            LIMIT 50
        """
        results = await self._query(cypher, hook_name=hook_name, **filter_params)
        return [
            CallGraphNode(
                name=row["name"],
                file_path=row["file_path"] or "",
                repository=row.get("repository"),
                signature=row.get("signature"),
                line_start=row.get("line_number"),
                entity_type=self._label_to_entity_type(row.get("entity_type")),
                relationship_type=row.get("relationship_type", "USES_HOOK"),
            )
            for row in results
        ]

    async def get_impact(
        self,
        method_name: str,
        repository: str | None = None,
        file_path: str | None = None,
        depth: int = 10,
        summary_only: bool = False,
        limit: int | None = None,
    ) -> ImpactResult | None:
        """Analyze blast radius of changing a method."""
        target = await self._resolve_method_target(
            method_name,
            repository=repository,
            file_path=file_path,
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
            RETURN caller.name AS name, caller.file_path AS file_path,
                   caller.repository AS repository,
                   caller.signature AS signature, caller.line_number AS line_number,
                   caller.stereotypes AS stereotypes,
                   depth
            ORDER BY depth, caller.file_path, caller.line_number
            LIMIT 200
        """
        caller_results = await self._query(callers_cypher, entity_id=target["id"])

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
