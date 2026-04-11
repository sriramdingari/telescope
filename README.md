# Telescope

An MCP server that queries [Constellation](https://github.com/sriramdingari/Constellation)'s code knowledge graph. Feed it a natural language question and it searches across embedded entities using vector similarity, then lets you traverse the full persisted graph: files, packages, exports, fields, hooks, references, call graphs, impact, and class hierarchy.

Telescope is the query layer for Constellation. Constellation indexes codebases into Neo4j; Telescope lets AI assistants query that graph.

## Architecture

```mermaid
graph LR
    A[Claude Code / AI Assistant] -->|MCP Protocol| B[Telescope]
    B -->|Cypher Queries| C[(Neo4j)]
    B -->|SQL + pgvector| F[(Postgres)]
    B -->|Embedding API| D[OpenAI / LiteLLM]
    E[Constellation] -->|Indexes code into| C
    E -->|Indexes code into| F

    style B fill:#009688,color:#fff
    style C fill:#4C8BF5,color:#fff
    style F fill:#336791,color:#fff
    style D fill:#412991,color:#fff
    style E fill:#F57C00,color:#fff
```

Telescope speaks to whichever backend Constellation indexed into. Both backends implement the identical 13-tool MCP contract — switch between them with a single environment variable (`STORAGE_BACKEND`).

## Installation

### Prerequisites

- Python 3.12+
- A running [Constellation](https://github.com/sriramdingari/Constellation) deployment (Neo4j with indexed code)
- An OpenAI API key (or LiteLLM proxy) for query-time embedding generation

### Install from source

```bash
git clone https://github.com/sriramdingari/telescope.git
cd telescope
pip install .
```

### Install with uvx (no clone needed)

```bash
uvx --from git+https://github.com/sriramdingari/telescope.git telescope
```

### Postgres Backend (Optional)

By default, Telescope reads from Neo4j. To use a PostgreSQL + pgvector backend instead (e.g. if your Constellation instance is indexing into Postgres), set two environment variables:

```bash
export STORAGE_BACKEND=postgres
export POSTGRES_DSN=postgresql://constellation:secret@localhost:5432/constellation
```

`asyncpg` and `pgvector` are shipped as default dependencies, so no extra install step is needed. The Postgres backend implements the same MCP tool contract as the Neo4j backend — all 13 tools (`search_code`, `get_callers`, `get_impact`, etc.) work identically.

## Setup with Claude Code

Add Telescope as an MCP server in Claude Code:

```bash
claude mcp add-json telescope --scope user '{
  "command": "uvx",
  "args": ["--from", "git+https://github.com/sriramdingari/telescope.git", "telescope"],
  "env": {
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "constellation",
    "OPENAI_API_KEY": "sk-your-key-here"
  }
}'
```

For LiteLLM proxy:

```bash
claude mcp add-json telescope --scope user '{
  "command": "uvx",
  "args": ["--from", "git+https://github.com/sriramdingari/telescope.git", "telescope"],
  "env": {
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "constellation",
    "OPENAI_API_KEY": "sk-your-litellm-key",
    "OPENAI_BASE_URL": "http://localhost:4000"
  }
}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_BACKEND` | `neo4j` | `neo4j` or `postgres` — which backend Telescope reads from |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI (when `STORAGE_BACKEND=neo4j`) |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `constellation` | Neo4j password |
| `POSTGRES_DSN` | — | Postgres + pgvector DSN (required when `STORAGE_BACKEND=postgres`) |
| `OPENAI_API_KEY` | — | API key for embedding generation |
| `OPENAI_BASE_URL` | — | Custom base URL (e.g., LiteLLM proxy) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector dimensions |

### Ollama embedding provider (optional)

Telescope defaults to OpenAI for query-side embeddings. To use a local Ollama
server instead — typically when the corresponding Constellation indexing run
also used Ollama — set:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | Set to `ollama` to enable the Ollama path |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Must match Constellation's indexing model |
| `OLLAMA_EMBEDDING_DIMENSIONS` | `768` | Must match the model's native dim (and Constellation's setting) |

**Critical:** the embedding model, provider, and dimensions MUST be identical
on the indexing side (Constellation) and the query side (telescope). If they
diverge you get either a hard pgvector dimension error or silently broken
search results (vectors from different models live in different semantic
spaces).

## Tools

Telescope exposes 13 tools via the MCP protocol:

### search_code

Semantic code search using vector similarity. Better than grep for conceptual searches like "authentication logic" or "database connection handling". Results now include stable entity ids plus graph metadata such as language, return type, modifiers, stereotypes, content hashes, and custom properties when present. Identifier-like queries such as `useState`, `MyClass`, or `src/App.tsx` automatically blend in exact graph symbol matches.

```
search_code("payment processing", repository="my-app", entity_type="method", code_mode="preview")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language search query |
| `repository` | string | — | Filter by repository name |
| `entity_type` | string | — | `"method"`, `"class"`, `"interface"`, or `"constructor"` |
| `file_pattern` | string | — | Filter by file path pattern |
| `language` | string | — | Filter by persisted language |
| `stereotype` | string | — | Filter by persisted stereotype |
| `limit` | int | 10 | Max results (capped at 20) |
| `code_mode` | string | `"preview"` | `"none"`, `"signature"`, `"preview"` (10 lines), `"full"` |

### find_symbols

Exact/substring graph lookup across all persisted entity types, including files, packages, fields, hooks, and references. Results carry the same metadata-rich shape as `search_code`, and also support `language` and `stereotype` filters.

```
find_symbols("useState", entity_types=["hook"], repository="my-app")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Identifier or path fragment to look up |
| `entity_types` | list[string] | — | Filter by entity type (e.g. `"hook"`, `"class"`, `"method"`, `"field"`, `"file"`, `"package"`, `"reference"`) |
| `repository` | string | — | Filter by repository name |
| `file_pattern` | string | — | Filter by file path pattern |
| `language` | string | — | Filter by persisted language |
| `stereotype` | string | — | Filter by persisted stereotype |
| `limit` | int | 20 | Max results (capped at 50) |
| `exact` | bool | `false` | When true, match the identifier exactly; otherwise substring match |
| `code_mode` | string | `"none"` | `"none"`, `"signature"`, `"preview"` (10 lines), `"full"` |

Unlike `search_code`, `find_symbols` is the identifier-lookup tool, so `code_mode` defaults to `"none"` to keep responses small; agents that need code bodies should pass `"signature"`, `"preview"`, or `"full"` explicitly.

### get_repository_context

Get one repository's source metadata plus aggregate graph statistics.

```
get_repository_context("my-app")
```

### get_callers

Find all functions that call the specified method. Telescope derives interface/implementation families from Constellation's `IMPLEMENTS` and `EXTENDS` graph, instead of relying on method-level `OVERRIDES` edges.

```
get_callers("processPayment", repository="my-app", depth=2)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method_name` | string | required | Method name to find callers for |
| `repository` | string | — | Filter by repository |
| `file_path` | string | — | Disambiguate by file path |
| `depth` | int | 1 | Traversal depth (max 3) |
| `limit` | int | 50 | Max results (capped at 200) |

Each caller result includes `truncated` when additional callers exist beyond the limit.
Caller results also include `depth`, the minimum call distance to the target.

### get_callees

Find all functions, unresolved references, and hooks used by the specified method.

```
get_callees("processPayment", repository="my-app")
```

Same parameters as `get_callers`. Each callee result also includes `truncated` when additional targets exist beyond the limit.
Callee results include `depth`, and can include unresolved `Reference` nodes plus `Hook` usage.

### get_function_context

Get comprehensive context for a function before modifying it: code, signature, docstring, parent class, callers, and callees.

```
get_function_context("processPayment", repository="my-app")
```

`callees` can include real methods/constructors, unresolved `Reference` nodes, and `Hook` nodes.

### get_class_hierarchy

Get inheritance hierarchy for a class or interface, including parents, children, interfaces, implementors, methods, fields, and constructors.

```
get_class_hierarchy("UserService", repository="my-app")
```

### get_package_context

Get package or namespace membership, including files, classes, methods, hooks, references, and direct child packages.

```
get_package_context("src.services", repository="my-app")
```

### get_impact

Analyze blast radius of changing a method. Shows affected tests, endpoints, and other transitive callers.

```
get_impact("processPayment", summary_only=True)       # Quick count
get_impact("processPayment", limit=5)                  # Limited details
get_impact("validateUser")                             # Full analysis
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method_name` | string | required | Method to analyze |
| `repository` | string | — | Filter by repository |
| `file_path` | string | — | Disambiguate by file path |
| `depth` | int | 10 | Max call chain depth |
| `summary_only` | bool | false | Return counts only (no caller details) |
| `limit` | int | — | Max callers per category |

### get_file_context

Get graph context for a single file: package membership, exports, top-level entities, class-scoped constructors and fields, unresolved references, hooks used inside that file, and the file content hash.

```
get_file_context("src/App.tsx", repository="my-app")
```

### get_hook_usage

Find the methods and constructors that use a materialized hook node such as `useState` or `useEffect`. Supports `repository`, `file_pattern`, `language`, `stereotype`, and `limit`. Each result includes `depth` and `truncated`.

```
get_hook_usage("useState", repository="my-app")
```

### list_repositories

List all repositories indexed by Constellation.

```
list_repositories()
```

### get_codebase_overview

High-level codebase statistics: files, classes, interfaces, methods, constructors, fields, packages, hooks, references, exports, languages, entry points, and top-level classes.

```
get_codebase_overview(repository="my-app", include_packages=True)
```

## Language Support

Telescope queries whatever Constellation has indexed. Currently supported languages:

| Language | Extensions |
|----------|-----------|
| Java | `.java` |
| Python | `.py` |
| JavaScript / TypeScript | `.js` `.ts` `.jsx` `.tsx` |
| C# | `.cs` |

## Development

```bash
git clone https://github.com/sriramdingari/telescope.git
cd telescope
pip install -e ".[dev]"
python -m pytest -v
```

### Live Contract Tests

Telescope ships **two** opt-in contract suites that seed each backend with real Constellation parser output (via the production normalization pipeline) and verify the read contract end-to-end. Both suites parse the same fixture repo (Java + JS/TS + C# nested namespace) using Constellation's actual parsers, so a regression in either backend's read path or in Constellation's parser shape surfaces immediately.

**Shared requirement** — a working Constellation checkout with parser deps installed:

- `CONSTELLATION_ROOT=/path/to/Constellation` (the conftest has a developer-machine default; set this env var on any other host)
- `CONSTELLATION_PYTHON` (optional) — Python executable inside Constellation's venv (defaults to `$CONSTELLATION_ROOT/.venv/bin/python`)

Without Constellation reachable, both suites skip cleanly with an actionable message.

#### Neo4j contract suite (`tests/test_contract_integration.py`)

Gated by the `integration` pytest marker. Requires a running Neo4j and `TELESCOPE_RUN_INTEGRATION=1`.

```bash
TELESCOPE_RUN_INTEGRATION=1 \
CONSTELLATION_ROOT=/path/to/Constellation \
TELESCOPE_TEST_NEO4J_URI=bolt://localhost:7687 \
TELESCOPE_TEST_NEO4J_PASSWORD=password \
python -m pytest -v -m integration
```

Override defaults via `TELESCOPE_TEST_NEO4J_URI`, `TELESCOPE_TEST_NEO4J_USER`, `TELESCOPE_TEST_NEO4J_PASSWORD`.

#### Postgres contract suite (`tests/test_postgres_contract_integration.py`)

Gated by the `postgres_integration` pytest marker. Runs **by default** in any normal `pytest` invocation when Docker is available — `testcontainers[postgres]` spins up a `pgvector/pgvector:pg16` container for the session, Constellation seeds it through its real `PostgresWriteBackend`, and the suite tears it all down at the end.

```bash
# Runs automatically with Docker available — no env vars needed:
python -m pytest -v tests/test_postgres_contract_integration.py
```

Without Docker, the suite skips gracefully via the testcontainers fixture.

Both suites validate the same parity contracts: file/package/class/constructor/field/reference/hook/call-graph queries, code-mode handling, overload disambiguation, and nested .NET namespace reconstruction (`Company.Product.Services` should never leak the leaf-only `Services` from any query).

## License

MIT
