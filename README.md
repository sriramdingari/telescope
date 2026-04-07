# Telescope

An MCP server that queries [Constellation](https://github.com/sriramdingari/Constellation)'s code knowledge graph. Feed it a natural language question and it searches across embedded entities using vector similarity, then lets you traverse the full persisted graph: files, packages, exports, fields, hooks, references, call graphs, impact, and class hierarchy.

Telescope is the query layer for Constellation. Constellation indexes codebases into Neo4j; Telescope lets AI assistants query that graph.

## Architecture

```mermaid
graph LR
    A[Claude Code / AI Assistant] -->|MCP Protocol| B[Telescope]
    B -->|Cypher Queries| C[(Neo4j)]
    B -->|Embedding API| D[OpenAI / LiteLLM]
    E[Constellation] -->|Indexes code into| C

    style B fill:#009688,color:#fff
    style C fill:#4C8BF5,color:#fff
    style D fill:#412991,color:#fff
    style E fill:#F57C00,color:#fff
```

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
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `constellation` | Neo4j password |
| `OPENAI_API_KEY` | — | API key for embedding generation |
| `OPENAI_BASE_URL` | — | Custom base URL (e.g., LiteLLM proxy) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector dimensions |

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

Telescope includes an opt-in integration suite that seeds Neo4j with real parser output from Constellation and verifies the query contract end to end.

Requirements:

- local Neo4j reachable through `NEO4J_URI`-style settings
- a working Constellation checkout with its parser dependencies installed

Environment:

- `TELESCOPE_RUN_INTEGRATION=1` enables the live tests
- `CONSTELLATION_ROOT` points at the Constellation checkout
- `CONSTELLATION_PYTHON` optionally points at the Python executable inside Constellation's environment
- `TELESCOPE_TEST_NEO4J_URI`, `TELESCOPE_TEST_NEO4J_USER`, and `TELESCOPE_TEST_NEO4J_PASSWORD` override the default local Neo4j settings

Example:

```bash
TELESCOPE_RUN_INTEGRATION=1 \
CONSTELLATION_ROOT=/path/to/Constellation \
CONSTELLATION_PYTHON=/path/to/Constellation/.venv/bin/python \
python -m pytest -v -m integration
```

## License

MIT
