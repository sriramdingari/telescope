"""Shared test fixtures for Telescope tests."""

import json
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase

from telescope.config import Config


TESTS_ROOT = Path(__file__).resolve().parent
CONTRACT_FIXTURE_ROOT = TESTS_ROOT / "fixtures" / "contract_repo"
DEFAULT_CONSTELLATION_ROOT = Path("/Users/d.sriram/Desktop/personal/Constellation")
ALLOWED_ENTITY_LABELS = {
    "File",
    "Package",
    "Class",
    "Interface",
    "Method",
    "Constructor",
    "Field",
    "Hook",
    "Reference",
}
ALLOWED_RELATIONSHIP_TYPES = {
    "CONTAINS",
    "IN_PACKAGE",
    "HAS_METHOD",
    "HAS_CONSTRUCTOR",
    "HAS_FIELD",
    "DECLARES",
    "EXTENDS",
    "IMPLEMENTS",
    "CALLS",
    "USES_HOOK",
    "EXPORTS",
}


@pytest.fixture()
def mock_neo4j_result():
    """Create a mock Neo4j result that returns configurable data."""
    def _make_result(data):
        result = AsyncMock()
        records = [MagicMock(data=MagicMock(return_value=d)) for d in data]
        result.fetch = AsyncMock(return_value=records)
        result.data = AsyncMock(return_value=data)
        return result
    return _make_result


@pytest.fixture()
def mock_neo4j_session(mock_neo4j_result):
    """Create a mock Neo4j async session."""
    session = AsyncMock()
    session.run = AsyncMock(return_value=mock_neo4j_result([]))
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture()
def mock_neo4j_driver(mock_neo4j_session):
    """Create a mock Neo4j async driver."""
    driver = AsyncMock()
    driver.session = MagicMock(return_value=mock_neo4j_session)
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    return driver


@pytest.fixture()
def mock_openai_response():
    """Create a mock OpenAI embedding response."""
    def _make_response(embedding=None):
        if embedding is None:
            embedding = [0.1] * 1536
        response = MagicMock()
        data_item = MagicMock()
        data_item.embedding = embedding
        response.data = [data_item]
        return response
    return _make_response


@pytest.fixture()
def mock_openai_client(mock_openai_response):
    """Create a mock AsyncOpenAI client."""
    client = AsyncMock()
    client.embeddings = AsyncMock()
    client.embeddings.create = AsyncMock(return_value=mock_openai_response())
    client.close = AsyncMock()
    return client


def _require_integration() -> None:
    """Skip integration fixtures unless explicitly enabled."""
    if os.environ.get("TELESCOPE_RUN_INTEGRATION") != "1":
        pytest.skip("Set TELESCOPE_RUN_INTEGRATION=1 to run live Neo4j contract tests")


def _constellation_root() -> Path:
    return Path(os.environ.get("CONSTELLATION_ROOT", DEFAULT_CONSTELLATION_ROOT))


def _constellation_python() -> Path:
    return Path(
        os.environ.get(
            "CONSTELLATION_PYTHON",
            _constellation_root() / ".venv" / "bin" / "python",
        )
    )


def _neo4j_config() -> Config:
    return Config(
        neo4j_uri=os.environ.get("TELESCOPE_TEST_NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("TELESCOPE_TEST_NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("TELESCOPE_TEST_NEO4J_PASSWORD", "constellation"),
        openai_api_key="sk-test-key",
    )


def _parse_contract_fixture(repository: str) -> dict:
    """Parse the contract fixture repo with Constellation from its own venv."""
    script = f"""
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(_constellation_root())!r})

from constellation.parsers.registry import get_default_registry

root = Path({str(CONTRACT_FIXTURE_ROOT)!r})
repository = {repository!r}
registry = get_default_registry()
entities = []
relationships = []
errors = []

for file_path in sorted(path for path in root.rglob('*') if path.is_file()):
    parser = registry.get_parser_for_file(file_path)
    if parser is None:
        continue
    result = parser.parse_file(file_path, repository)
    errors.extend(result.errors)
    for entity in result.entities:
        entities.append({{
            'id': entity.id,
            'name': entity.name,
            'entity_type': entity.entity_type.value,
            'repository': entity.repository,
            'file_path': entity.file_path,
            'line_number': entity.line_number,
            'line_end': entity.line_end,
            'language': entity.language,
            'code': entity.code,
            'signature': entity.signature,
            'return_type': entity.return_type,
            'docstring': entity.docstring,
            'modifiers': entity.modifiers,
            'stereotypes': entity.stereotypes,
            'properties': entity.properties,
            'content_hash': entity.content_hash,
        }})
    for relationship in result.relationships:
        relationships.append({{
            'source_id': relationship.source_id,
            'target_id': relationship.target_id,
            'relationship_type': relationship.relationship_type.value,
            'properties': relationship.properties,
        }})

if errors:
    raise SystemExit('\\n'.join(errors))

print(json.dumps({{'entities': entities, 'relationships': relationships}}))
"""
    result = subprocess.run(
        [str(_constellation_python()), "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    entities_by_id: dict[str, dict] = {}
    for entity in payload["entities"]:
        entities_by_id[entity["id"]] = entity

    relationships: list[dict] = []
    seen_relationships: set[tuple[str, str, str, str]] = set()
    for relationship in payload["relationships"]:
        key = (
            relationship["source_id"],
            relationship["target_id"],
            relationship["relationship_type"],
            json.dumps(relationship.get("properties") or {}, sort_keys=True),
        )
        if key in seen_relationships:
            continue
        seen_relationships.add(key)
        relationships.append(relationship)

    return {
        "entities": list(entities_by_id.values()),
        "relationships": relationships,
    }


async def _run_write(driver, cypher: str, **params) -> None:
    async with driver.session() as session:
        result = await session.run(cypher, **params)
        await result.consume()


async def _delete_repository_graph(driver, repository: str) -> None:
    await _run_write(driver, "MATCH (n {repository: $repository}) DETACH DELETE n", repository=repository)
    await _run_write(driver, "MATCH (r:Repository {name: $repository}) DETACH DELETE r", repository=repository)


async def _seed_repository_graph(driver, repository: str, payload: dict) -> None:
    await _delete_repository_graph(driver, repository)

    entity_groups: dict[str, list[dict]] = {}
    for entity in payload["entities"]:
        label = entity["entity_type"]
        assert label in ALLOWED_ENTITY_LABELS
        custom_properties = entity.get("properties") or {}
        props = {
            key: value
            for key, value in entity.items()
            if key not in {"entity_type", "properties"} and value is not None
        }
        for key, value in custom_properties.items():
            if value is not None:
                props.setdefault(key, value)
        entity_groups.setdefault(label, []).append({"id": entity["id"], "props": props})

    for label, rows in entity_groups.items():
        await _run_write(
            driver,
            f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{id: row.id}})
            SET n = row.props
            """,
            rows=rows,
        )

    relationship_groups: dict[str, list[dict]] = {}
    for relationship in payload["relationships"]:
        rel_type = relationship["relationship_type"]
        assert rel_type in ALLOWED_RELATIONSHIP_TYPES
        relationship_groups.setdefault(rel_type, []).append(
            {
                "source_id": relationship["source_id"],
                "target_id": relationship["target_id"],
                "props": relationship.get("properties") or {},
            }
        )

    for rel_type, rows in relationship_groups.items():
        await _run_write(
            driver,
            f"""
            UNWIND $rows AS row
            MATCH (source {{id: row.source_id}})
            MATCH (target {{id: row.target_id}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r = row.props
            """,
            rows=rows,
        )

    await _run_write(
        driver,
        """
        MERGE (r:Repository {name: $name})
        SET r.source = $source,
            r.last_indexed_at = $last_indexed_at,
            r.last_commit_sha = $last_commit_sha,
            r.entity_count = $entity_count
        """,
        name=repository,
        source="contract-fixture",
        last_indexed_at="2026-03-12T00:00:00+00:00",
        last_commit_sha="contract-fixture",
        entity_count=len(payload["entities"]),
    )


@pytest.fixture()
async def live_neo4j_driver():
    """Create a real Neo4j driver for live contract tests."""
    _require_integration()
    config = _neo4j_config()
    driver = AsyncGraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )
    await driver.verify_connectivity()
    try:
        yield driver
    finally:
        await driver.close()


@pytest.fixture()
async def live_graph_client():
    """Create a real Telescope Neo4jReadBackend against the local Neo4j instance."""
    _require_integration()
    with patch("telescope.backends.neo4j.get_config", return_value=_neo4j_config()), \
         patch("telescope.backends.neo4j.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        mock_openai.return_value = mock_client
        from telescope.backends.neo4j import Neo4jReadBackend

        client = Neo4jReadBackend()
        await client.connect()
        try:
            yield client
        finally:
            await client.close()


@pytest.fixture()
async def seeded_contract_repository(live_neo4j_driver):
    """Seed a small Constellation-shaped graph into Neo4j and clean it up."""
    _require_integration()
    repository = f"telescope-contract-{uuid4().hex[:8]}"
    payload = _parse_contract_fixture(repository)
    await _seed_repository_graph(live_neo4j_driver, repository, payload)
    try:
        yield repository
    finally:
        await _delete_repository_graph(live_neo4j_driver, repository)


# ── Postgres contract-test fixtures (session-scoped) ─────────────────

# Lazy import: testcontainers is only needed for postgres_integration tests.
# If it's not installed OR Docker is unavailable, the fixture skips.
_postgres_container_class = None
_postgres_import_error: str | None = None
try:
    from testcontainers.postgres import PostgresContainer as _postgres_container_class
except ImportError as exc:
    _postgres_import_error = f"testcontainers not installed: {exc}"


@pytest.fixture(scope="session")
def postgres_container():
    """Session-scoped pgvector container. Starts once, tears down at session end.

    Gracefully skips postgres_integration tests if testcontainers is
    missing OR Docker isn't available.

    Bootstraps the pgvector extension once after the container starts.
    This is database-engine setup (not Constellation-owned schema), and
    mirrors what Constellation's PostgresWriteBackend.connect() does
    before creating its own pool. Without this, Telescope's
    PostgresReadBackend.connect() races against the seeding subprocess
    when pytest resolves pg_read_backend before
    seeded_postgres_contract_repository.
    """
    if _postgres_container_class is None:
        pytest.skip(_postgres_import_error)
    try:
        container = _postgres_container_class(
            image="pgvector/pgvector:pg16",
            username="telescope",
            password="telescope",
            dbname="telescope",
        )
        container.start()
    except Exception as exc:
        pytest.skip(f"Docker unavailable for postgres contract tests: {exc}")

    # Bootstrap pgvector extension via asyncpg (already a Telescope dep).
    # We run a one-off event loop just for this connection.
    try:
        import asyncio
        import asyncpg

        raw_url = container.get_connection_url()
        bootstrap_dsn = raw_url.replace("postgresql+psycopg2://", "postgresql://")

        async def _bootstrap_extension() -> None:
            conn = await asyncpg.connect(bootstrap_dsn)
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await conn.close()

        asyncio.run(_bootstrap_extension())
    except Exception as exc:
        container.stop()
        pytest.skip(f"Failed to bootstrap pgvector extension: {exc}")

    yield container
    container.stop()


@pytest.fixture
def postgres_dsn(postgres_container):
    """asyncpg-compatible DSN for the running session container."""
    raw = postgres_container.get_connection_url()
    # testcontainers returns psycopg2-style URLs; normalize to asyncpg's form
    return raw.replace("postgresql+psycopg2://", "postgresql://")


def _seed_constellation_postgres_fixture(postgres_dsn: str, repository: str) -> None:
    """Parse the contract fixture repo with Constellation and load the
    result into Postgres via Constellation's real PostgresWriteBackend.

    Runs inside Constellation's own Python environment (via
    _constellation_python()) so we get Constellation's real parsers,
    schema DDL, and PostgresWriteBackend without Telescope having to
    import or duplicate any of them.

    Schema is created by Constellation's initialize_schema(); entities
    and relationships are written via apply_indexing_changes(), which
    is the exact code path production indexing uses.
    """
    # Graceful skip if Constellation venv or root isn't available.
    # Matches the plan's "skips gracefully when either Docker or Constellation
    # is missing" contract. The check is at fixture-use time rather than
    # module import so contributors without Constellation can still run the
    # non-integration suite; only the postgres_integration tests skip.
    constellation_python = _constellation_python()
    if not constellation_python.exists():
        pytest.skip(
            f"Constellation Python interpreter not found at {constellation_python}. "
            f"Set CONSTELLATION_ROOT or CONSTELLATION_PYTHON to run postgres contract tests."
        )
    constellation_root = _constellation_root()
    if not (constellation_root / "constellation" / "graph" / "postgres.py").exists():
        pytest.skip(
            f"Constellation source not found at {constellation_root}. "
            f"Set CONSTELLATION_ROOT to the path containing constellation/graph/postgres.py."
        )

    script = f"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, {str(_constellation_root())!r})

from constellation.graph.postgres import PostgresWriteBackend
from constellation.indexer.pipeline import IndexingPipeline
from constellation.models import CodeEntity, EntityType
from constellation.parsers.registry import get_default_registry


async def main() -> None:
    registry = get_default_registry()
    root = Path({str(CONTRACT_FIXTURE_ROOT)!r})
    repository = {repository!r}

    entities = []
    relationships = []
    reindex_preps = {{}}  # relative_path -> set[entity_id]
    errors = []

    for file_path in sorted(path for path in root.rglob('*') if path.is_file()):
        parser = registry.get_parser_for_file(file_path)
        if parser is None:
            continue
        parse_result = parser.parse_file(file_path, repository)
        errors.extend(parse_result.errors)

        # Mirror IndexingPipeline lines 177-270: derive the canonical
        # relative path and file_entity_id, run the production
        # normalization step, then create the canonical File entity the
        # pipeline would have created (the parser's File entity is
        # dropped by _normalize_parse_result).
        relative_path = str(file_path.relative_to(root))
        file_entity_id = f"{{repository}}::{{relative_path}}"

        # Static method — no pipeline instance needed. Returns
        # (normalized_entities, normalized_relationships) with non-File
        # entities having repo-relative file_paths, Python/JavaScript
        # entity IDs remapped to the scoped-ID map, and CALLS target IDs
        # resolved via call_aliases. See
        # constellation/indexer/pipeline.py:325-380.
        normalized_entities, normalized_relationships = (
            IndexingPipeline._normalize_parse_result(
                parse_result=parse_result,
                relative_path=relative_path,
                file_entity_id=file_entity_id,
                language=parser.language,
            )
        )

        # Canonical File entity, mirroring pipeline.py:252-262. We pass
        # content_hash=None because the contract fixture doesn't need
        # hash-based change detection (code_symbols.content_hash is
        # nullable; get_file_hashes filters WHERE content_hash IS NOT
        # NULL, so a NULL hash just means "no cached hash" — not a
        # corrupt row).
        file_entity = CodeEntity(
            id=file_entity_id,
            name=file_path.name,
            entity_type=EntityType.FILE,
            repository=repository,
            file_path=relative_path,
            line_number=1,
            language=parser.language,
            content_hash=None,
        )

        entities.append(file_entity)
        entities.extend(normalized_entities)
        relationships.extend(normalized_relationships)

        reindex_preps[relative_path] = (
            {{file_entity_id}}
            | {{entity.id for entity in normalized_entities}}
        )

    if errors:
        raise SystemExit('\\n'.join(errors))

    backend = PostgresWriteBackend(
        dsn={postgres_dsn!r},
        embedding_dimensions=1536,
        embedding_model='text-embedding-3-small',
    )
    await backend.connect()
    try:
        await backend.initialize_schema()
        # Idempotent: delete any prior data for this repository before
        # loading the fresh parse. delete_repository cascades through
        # code_symbols, code_references, and code_embeddings.
        await backend.delete_repository(repository)
        await backend.apply_indexing_changes(
            repository=repository,
            source=str(root),
            commit_sha=None,
            reindex_preparations=list(reindex_preps.items()),
            entities=entities,
            relationships=relationships,
            stale_file_paths=[],
        )
    finally:
        await backend.close()


asyncio.run(main())
"""
    try:
        subprocess.run(
            [str(_constellation_python()), "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Constellation seeding subprocess failed "
            f"(exit code {exc.returncode}).\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc


@pytest_asyncio.fixture
async def seeded_postgres_contract_repository(postgres_dsn):
    """Seed the session-scoped pgvector container with the contract fixture
    repo (parsed by Constellation's real parsers and loaded via
    Constellation's real PostgresWriteBackend). Yields a unique repository
    name.

    Per-test cleanup is intentionally omitted: each test gets a
    uuid4-suffixed repository name, queries always filter by
    `repository = $1`, and the container is torn down at session end.
    Adding Telescope-side cleanup would require duplicating Constellation's
    schema knowledge (which tables to delete from), creating a drift risk
    if Constellation ever adds new tables that don't cascade from
    code_symbols. Session-end teardown is the schema-ownership-safe
    alternative.

    Mirrors the Neo4j contract pattern at conftest.py:228 but uses
    Constellation's Postgres write path instead of raw Cypher.
    """
    repository = f"telescope-postgres-contract-{uuid4().hex[:8]}"
    _seed_constellation_postgres_fixture(postgres_dsn, repository)
    yield repository
