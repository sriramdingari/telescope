# Ollama Embedding Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ollama as an alternative embedding provider in telescope, so telescope's query-side embeddings can match what Constellation writes when Constellation is configured for Ollama. Without this, semantic search over Ollama-indexed repos is broken (vector dimension mismatch or garbage similarity).

**Architecture:** Introduce an embedding-provider abstraction (`BaseEmbeddingProvider` ABC with `OpenAIEmbeddingProvider` and `OllamaEmbeddingProvider` implementations) under `telescope/src/telescope/embeddings/`. Select the provider via a new `EMBEDDING_PROVIDER` env var and inject the chosen provider into each backend's constructor. Backends stop talking to `AsyncOpenAI` directly and delegate to `self._embedder.embed_batch([query])[0]`.

**Tech Stack:** Python 3.12+, pytest + pytest-asyncio, httpx (for Ollama), openai SDK (existing), asyncpg + pgvector (Postgres), neo4j driver (Neo4j).

---

## Background: Why This Exists

Constellation recently gained Ollama support. Its indexer can write embeddings produced by `nomic-embed-text` (768 dims) into the same Postgres/Neo4j store that telescope queries. Telescope, however, hardcodes OpenAI:

- `telescope/src/telescope/config.py:13-16` — only exposes `openai_api_key`, `openai_base_url`, `embedding_model`, `embedding_dimensions`.
- `telescope/src/telescope/backends/postgres.py:36-39` — constructs `AsyncOpenAI` in `__init__`.
- `telescope/src/telescope/backends/postgres.py:77-83` — `_get_embedding` calls OpenAI directly.
- `telescope/src/telescope/backends/neo4j.py:83-94` — same pattern in `connect()`.
- `telescope/src/telescope/backends/neo4j.py:133-140` — same `_get_embedding` via OpenAI.

If you index with Ollama (768-dim `nomic-embed-text`) and query via telescope (1536-dim `text-embedding-3-small`), pgvector errors on the dimension mismatch. On Neo4j, you'd silently get garbage similarity because vectors from different models live in different semantic spaces.

**Constraint — provider/model/dimensions must match on both sides.** If a deployment uses Ollama for indexing, it must use Ollama for querying, with the same `model` and `dimensions`.

---

## File Structure

### New files
```
src/telescope/embeddings/
├── __init__.py                # Public exports
├── base.py                    # BaseEmbeddingProvider ABC
├── openai_provider.py         # OpenAIEmbeddingProvider
├── ollama_provider.py         # OllamaEmbeddingProvider
└── factory.py                 # create_embedding_provider()

tests/embeddings/
├── __init__.py
├── test_base.py               # ABC contract tests
├── test_openai_provider.py    # OpenAI provider tests
├── test_ollama_provider.py    # Ollama provider tests
└── test_factory.py            # Factory tests
```

> **Naming note:** both provider modules use a `_provider` suffix for symmetry. `openai_provider.py` avoids shadowing the installed `openai` pip package inside `telescope.embeddings`. `ollama_provider.py` guards against the same concern if an `ollama` pip package ever lands as a telescope dependency.

### Modified files
- `pyproject.toml` — add `httpx` as an explicit dependency (it's currently transitive via `openai`, but Ollama provider needs it directly).
- `src/telescope/config.py` — add `embedding_provider`, `ollama_base_url`, `ollama_embedding_model`, `ollama_embedding_dimensions`; add `resolved_embedding_model()` and `resolved_embedding_dimensions()` helpers.
- `src/telescope/backends/postgres.py` — constructor accepts `embedder: BaseEmbeddingProvider`; delete `_openai` field; `_get_embedding` delegates to `self._embedder`.
- `src/telescope/backends/neo4j.py` — constructor accepts `embedder: BaseEmbeddingProvider` (breaking the current no-arg signature); same `_get_embedding` delegation; `connect()` no longer creates `AsyncOpenAI`.
- `src/telescope/backends/factory.py` — builds the embedding provider via `create_embedding_provider(config)` and injects it into both backends.
- `tests/test_config.py` — add tests for new config fields and resolver methods.
- `tests/backends/test_factory.py` — update to assert new factory signature.
- `tests/backends/test_postgres_read.py` — update the `backend` fixture to pass a mock embedder.
- `tests/backends/test_neo4j.py` — update fixtures and drop `AsyncOpenAI` patching.
- `tests/conftest.py:362-365` — the neo4j backend fixture patches `telescope.backends.neo4j.AsyncOpenAI`; this patch target no longer exists. Replace with an embedder mock.

### Out of scope
- Changing any actual query logic (SQL, Cypher, scoring).
- Adding retries to either provider (Constellation has tenacity retries on OpenAI; telescope currently doesn't and we preserve that).
- Changing pgvector schema or column dimensions.
- Adding new MCP tools.
- Supporting multiple embedding providers simultaneously.

---

## Relevant Skills

Invoke these sub-skills when relevant during execution:

- `superpowers:test-driven-development` — strict TDD for each task.
- `superpowers:verification-before-completion` — run full test suite before marking any task complete.
- `superpowers:systematic-debugging` — if a test fails in a surprising way.

---

## Ground Rules

- **Work in the telescope project root** `/Users/d.sriram/Desktop/personal/telescope` throughout. All paths below are relative to that root unless prefixed with `~`.
- **TDD** — write the failing test first, run it to confirm the failure, then implement, then verify.
- **Frequent commits** — one commit per task unless otherwise noted.
- **No OpenAI import inside the Ollama provider.** The whole point is to decouple.
- **No `telescope.config` reach-ins from provider files.** Providers take their config as constructor args.
- **Backward compatibility:** `embedding_provider` defaults to `"openai"` so existing deployments with only OpenAI env vars set continue to work unchanged.
- **Run tests from the project root:** `pytest` (not `python -m pytest`) so pytest config is picked up.

---

## Task 1: Add httpx to dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Inspect current deps**

Run: `grep -n "httpx\|openai" pyproject.toml`
Expected: `openai` listed; `httpx` not listed (transitive via `openai`).

- [ ] **Step 2: Add `httpx` to the `dependencies` array**

Edit `pyproject.toml` so the dependencies list reads:

```toml
dependencies = [
    "mcp>=1.25.0",
    "neo4j>=6.1.0",
    "openai>=2.15.0",
    "asyncpg>=0.29",
    "pgvector>=0.3",
    "httpx>=0.27",
]
```

- [ ] **Step 3: Re-sync the venv**

Run: `uv sync` (or `pip install -e .[dev]` if not using uv)
Expected: no errors; `httpx` shown as already-installed or newly installed.

- [ ] **Step 4: Verify import works**

Run: `python -c "import httpx; print(httpx.__version__)"`
Expected: a version number, no `ModuleNotFoundError`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock 2>/dev/null || git add pyproject.toml
git commit -m "chore: declare httpx as direct dependency for ollama provider"
```

---

## Task 2: Create the embeddings package skeleton

**Files:**
- Create: `src/telescope/embeddings/__init__.py`
- Create: `tests/embeddings/__init__.py`

- [ ] **Step 1: Create both `__init__.py` files as empty stubs**

```bash
mkdir -p src/telescope/embeddings tests/embeddings
printf '"""Embedding providers for Telescope query-side vector generation."""\n' > src/telescope/embeddings/__init__.py
printf '' > tests/embeddings/__init__.py
```

- [ ] **Step 2: Verify the package is importable**

Run: `python -c "import telescope.embeddings; print(telescope.embeddings.__doc__)"`
Expected: `Embedding providers for Telescope query-side vector generation.`

- [ ] **Step 3: Commit**

```bash
git add src/telescope/embeddings/__init__.py tests/embeddings/__init__.py
git commit -m "chore(embeddings): scaffold embeddings package"
```

---

## Task 3: Add BaseEmbeddingProvider ABC

**Files:**
- Create: `src/telescope/embeddings/base.py`
- Create: `tests/embeddings/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/embeddings/test_base.py`:

```python
"""Contract tests for BaseEmbeddingProvider."""

from __future__ import annotations

import pytest

from telescope.embeddings.base import BaseEmbeddingProvider


class TestBaseEmbeddingProviderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseEmbeddingProvider()  # type: ignore[abstract]

    def test_subclass_must_implement_embed_batch(self):
        class Incomplete(BaseEmbeddingProvider):
            @property
            def model_name(self) -> str:
                return "x"

            @property
            def dimensions(self) -> int:
                return 1

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        class Concrete(BaseEmbeddingProvider):
            @property
            def model_name(self) -> str:
                return "concrete-model"

            @property
            def dimensions(self) -> int:
                return 42

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 42 for _ in texts]

        c = Concrete()
        assert c.model_name == "concrete-model"
        assert c.dimensions == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/embeddings/test_base.py -v`
Expected: `ImportError` / `ModuleNotFoundError: No module named 'telescope.embeddings.base'`.

- [ ] **Step 3: Create `src/telescope/embeddings/base.py`**

```python
"""Abstract base class for embedding providers.

All query-side vector generation flows through an instance of a concrete
subclass. Implementations live as sibling modules (``openai_provider``,
``ollama_provider``) and are selected via :mod:`telescope.embeddings.factory`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Interface for query-side embedding generators.

    Subclasses wrap a specific backend (OpenAI API, Ollama HTTP API, etc.)
    and expose a uniform ``embed_batch`` coroutine plus ``model_name`` and
    ``dimensions`` introspection so telescope's read backends can build
    pgvector / Cypher-compatible query vectors without knowing which
    provider produced them.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the provider-specific model identifier in use."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding vector length this provider produces."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text, in order.

        An empty ``texts`` list MUST return an empty list without issuing
        any network calls.
        """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/embeddings/test_base.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/telescope/embeddings/base.py tests/embeddings/test_base.py
git commit -m "feat(embeddings): add BaseEmbeddingProvider ABC"
```

---

## Task 4: Add OpenAIEmbeddingProvider

**Files:**
- Create: `src/telescope/embeddings/openai_provider.py`
- Create: `tests/embeddings/test_openai_provider.py`

- [ ] **Step 1: Write the failing test**

Create `tests/embeddings/test_openai_provider.py`:

```python
"""Tests for OpenAIEmbeddingProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telescope.embeddings.openai_provider import OpenAIEmbeddingProvider


class TestOpenAIEmbeddingProviderProperties:
    def test_returns_configured_model(self):
        p = OpenAIEmbeddingProvider(
            api_key="sk-test",
            model="text-embedding-3-large",
        )
        assert p.model_name == "text-embedding-3-large"

    def test_returns_configured_dimensions(self):
        p = OpenAIEmbeddingProvider(
            api_key="sk-test",
            dimensions=2048,
        )
        assert p.dimensions == 2048

    def test_default_model(self):
        p = OpenAIEmbeddingProvider(api_key="sk-test")
        assert p.model_name == "text-embedding-3-small"

    def test_default_dimensions(self):
        p = OpenAIEmbeddingProvider(api_key="sk-test")
        assert p.dimensions == 1536


class TestOpenAIEmbedBatch:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_without_network_call(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            p = OpenAIEmbeddingProvider(api_key="sk-test")
            result = await p.embed_batch([])
        assert result == []
        mock_cls.return_value.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_single(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            mock_item = MagicMock()
            mock_item.embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [mock_item]
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            p = OpenAIEmbeddingProvider(api_key="sk-test", dimensions=3)
            result = await p.embed_batch(["hello"])

        assert result == [[0.1, 0.2, 0.3]]
        mock_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input=["hello"],
            dimensions=3,
        )

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            items = []
            for vec in [[0.1], [0.2], [0.3]]:
                it = MagicMock()
                it.embedding = vec
                items.append(it)
            mock_response = MagicMock()
            mock_response.data = items
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            p = OpenAIEmbeddingProvider(api_key="sk-test", dimensions=1)
            result = await p.embed_batch(["a", "b", "c"])

        assert result == [[0.1], [0.2], [0.3]]

    @pytest.mark.asyncio
    async def test_passes_base_url_when_provided(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            OpenAIEmbeddingProvider(
                api_key="sk-test",
                base_url="https://proxy.example.com",
            )
        mock_cls.assert_called_once_with(
            api_key="sk-test",
            base_url="https://proxy.example.com",
        )

    @pytest.mark.asyncio
    async def test_omits_base_url_when_none(self):
        with patch("telescope.embeddings.openai_provider.AsyncOpenAI") as mock_cls:
            OpenAIEmbeddingProvider(api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/embeddings/test_openai_provider.py -v`
Expected: `ModuleNotFoundError: No module named 'telescope.embeddings.openai_provider'`.

- [ ] **Step 3: Create `src/telescope/embeddings/openai_provider.py`**

```python
"""OpenAI-backed embedding provider.

Thin wrapper around :class:`openai.AsyncOpenAI` that matches the
:class:`telescope.embeddings.base.BaseEmbeddingProvider` contract.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from telescope.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Generate embeddings via the OpenAI (or OpenAI-compatible) API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/embeddings/test_openai_provider.py -v`
Expected: all tests passed.

- [ ] **Step 5: Commit**

```bash
git add src/telescope/embeddings/openai_provider.py tests/embeddings/test_openai_provider.py
git commit -m "feat(embeddings): add OpenAIEmbeddingProvider"
```

---

## Task 5: Add OllamaEmbeddingProvider

**Files:**
- Create: `src/telescope/embeddings/ollama_provider.py`
- Create: `tests/embeddings/test_ollama_provider.py`

**Design note:** Use Ollama's newer batch endpoint `/api/embed` (singular path, plural `embeddings` field in response). The older `/api/embeddings` endpoint only accepts a single `prompt` and forces one-HTTP-call-per-text, which is dramatically slower. We want one HTTP call per `embed_batch` invocation.

- [ ] **Step 1: Write the failing test**

Create `tests/embeddings/test_ollama_provider.py`:

```python
"""Tests for OllamaEmbeddingProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telescope.embeddings.ollama_provider import OllamaEmbeddingProvider


class TestOllamaEmbeddingProviderProperties:
    def test_returns_configured_model(self):
        p = OllamaEmbeddingProvider(model="mxbai-embed-large")
        assert p.model_name == "mxbai-embed-large"

    def test_default_model(self):
        p = OllamaEmbeddingProvider()
        assert p.model_name == "nomic-embed-text"

    def test_returns_configured_dimensions(self):
        p = OllamaEmbeddingProvider(dimensions=1024)
        assert p.dimensions == 1024

    def test_default_dimensions(self):
        p = OllamaEmbeddingProvider()
        assert p.dimensions == 768

    def test_strips_trailing_slash_from_base_url(self):
        p = OllamaEmbeddingProvider(base_url="http://localhost:11434/")
        assert p._base_url == "http://localhost:11434"


class TestOllamaEmbedBatch:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_without_network_call(self):
        with patch("telescope.embeddings.ollama_provider.httpx.AsyncClient") as mock_cls:
            p = OllamaEmbeddingProvider()
            result = await p.embed_batch([])
        assert result == []
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_single_batch_request(self):
        with patch("telescope.embeddings.ollama_provider.httpx.AsyncClient") as mock_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__.return_value = mock_client

            p = OllamaEmbeddingProvider(
                base_url="http://myhost:11434",
                model="nomic-embed-text",
            )
            result = await p.embed_batch(["alpha", "beta"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.post.assert_awaited_once_with(
            "http://myhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": ["alpha", "beta"]},
        )

    @pytest.mark.asyncio
    async def test_return_shape_for_three_inputs(self):
        with patch("telescope.embeddings.ollama_provider.httpx.AsyncClient") as mock_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "embeddings": [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2],
                ],
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__.return_value = mock_client

            p = OllamaEmbeddingProvider(dimensions=4)
            result = await p.embed_batch(["a", "b", "c"])

        assert len(result) == 3
        assert all(len(vec) == 4 for vec in result)
        assert mock_client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        with patch("telescope.embeddings.ollama_provider.httpx.AsyncClient") as mock_cls:
            import httpx as _httpx

            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock(
                side_effect=_httpx.HTTPStatusError(
                    "500", request=MagicMock(), response=MagicMock()
                )
            )
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__.return_value = mock_client

            p = OllamaEmbeddingProvider()
            with pytest.raises(_httpx.HTTPStatusError):
                await p.embed_batch(["x"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/embeddings/test_ollama_provider.py -v`
Expected: `ModuleNotFoundError: No module named 'telescope.embeddings.ollama_provider'`.

- [ ] **Step 3: Create `src/telescope/embeddings/ollama_provider.py`**

```python
"""Ollama-backed embedding provider.

Uses the ``/api/embed`` (singular) endpoint which accepts ``input`` as a
list and returns all vectors in one HTTP round-trip. The older
``/api/embeddings`` endpoint only accepts one ``prompt`` at a time, which
would force one HTTP request per text — orders of magnitude slower for
batch queries.

Ollama responses for this endpoint look like::

    {"embeddings": [[...], [...], ...]}
"""

from __future__ import annotations

import httpx

from telescope.embeddings.base import BaseEmbeddingProvider

# Give Ollama generous headroom — embedding a large batch with a cold
# model can take tens of seconds on CPU-only hosts.
_DEFAULT_TIMEOUT = httpx.Timeout(300.0)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Generate embeddings via a local (or remote) Ollama server."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimensions: int = 768,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
        return data["embeddings"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/embeddings/test_ollama_provider.py -v`
Expected: all 8 tests passed.

- [ ] **Step 5: Commit**

```bash
git add src/telescope/embeddings/ollama_provider.py tests/embeddings/test_ollama_provider.py
git commit -m "feat(embeddings): add OllamaEmbeddingProvider using /api/embed"
```

---

## Task 6: Extend Config with provider selection fields

**Files:**
- Modify: `src/telescope/config.py`
- Modify: `tests/test_config.py`

**What changes:**
- Add four new fields with safe defaults: `embedding_provider`, `ollama_base_url`, `ollama_embedding_model`, `ollama_embedding_dimensions`.
- Add `resolved_embedding_model()` / `resolved_embedding_dimensions()` helpers that pick the right value based on `embedding_provider`.
- Extend `from_env()` to read `EMBEDDING_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_EMBEDDING_DIMENSIONS`.
- Keep all existing fields and behaviors intact.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py` (place before the trailing `test_config_postgres_backend_*` helpers, grouped into new classes):

```python
class TestConfigEmbeddingProviderDefaults:
    """Tests for embedding-provider-related config defaults."""

    def test_default_embedding_provider_is_openai(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.embedding_provider == "openai"

    def test_default_ollama_base_url(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.ollama_base_url == "http://localhost:11434"

    def test_default_ollama_embedding_model(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.ollama_embedding_model == "nomic-embed-text"

    def test_default_ollama_embedding_dimensions(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.from_env()
        assert cfg.ollama_embedding_dimensions == 768


class TestConfigEmbeddingProviderEnvOverrides:
    def test_embedding_provider_override(self):
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "ollama"}, clear=True):
            cfg = Config.from_env()
        assert cfg.embedding_provider == "ollama"

    def test_ollama_base_url_override(self):
        with patch.dict(
            os.environ, {"OLLAMA_BASE_URL": "http://host.docker.internal:11434"}, clear=True
        ):
            cfg = Config.from_env()
        assert cfg.ollama_base_url == "http://host.docker.internal:11434"

    def test_ollama_embedding_model_override(self):
        with patch.dict(
            os.environ, {"OLLAMA_EMBEDDING_MODEL": "mxbai-embed-large"}, clear=True
        ):
            cfg = Config.from_env()
        assert cfg.ollama_embedding_model == "mxbai-embed-large"

    def test_ollama_embedding_dimensions_override(self):
        with patch.dict(
            os.environ, {"OLLAMA_EMBEDDING_DIMENSIONS": "1024"}, clear=True
        ):
            cfg = Config.from_env()
        assert cfg.ollama_embedding_dimensions == 1024
        assert isinstance(cfg.ollama_embedding_dimensions, int)


class TestResolvedEmbeddingHelpers:
    def _cfg(self, **kw):
        return Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="pw",
            openai_api_key="sk-test",
            **kw,
        )

    def test_resolved_openai_model(self):
        cfg = self._cfg(embedding_provider="openai", embedding_model="text-embedding-3-large")
        assert cfg.resolved_embedding_model() == "text-embedding-3-large"

    def test_resolved_openai_dimensions(self):
        cfg = self._cfg(embedding_provider="openai", embedding_dimensions=3072)
        assert cfg.resolved_embedding_dimensions() == 3072

    def test_resolved_ollama_model(self):
        cfg = self._cfg(
            embedding_provider="ollama",
            ollama_embedding_model="mxbai-embed-large",
        )
        assert cfg.resolved_embedding_model() == "mxbai-embed-large"

    def test_resolved_ollama_dimensions(self):
        cfg = self._cfg(
            embedding_provider="ollama",
            ollama_embedding_dimensions=1024,
        )
        assert cfg.resolved_embedding_dimensions() == 1024

    def test_resolved_model_rejects_unknown_provider(self):
        cfg = self._cfg(embedding_provider="bogus")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            cfg.resolved_embedding_model()

    def test_resolved_dimensions_rejects_unknown_provider(self):
        cfg = self._cfg(embedding_provider="bogus")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            cfg.resolved_embedding_dimensions()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v -k "EmbeddingProvider or Resolved"`
Expected: AttributeError / "unexpected keyword argument" failures on the new fields.

- [ ] **Step 3: Extend `src/telescope/config.py`**

Replace the full `src/telescope/config.py` with:

```python
"""Configuration for Telescope MCP server."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Server configuration loaded from environment variables."""

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    openai_api_key: str
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    storage_backend: str = "neo4j"
    postgres_dsn: str = ""
    # Embedding provider selection -----------------------------------
    # ``embedding_provider`` defaults to "openai" so existing deployments
    # that set only OPENAI_* env vars keep working. Setting it to
    # "ollama" activates the Ollama-specific settings below.
    embedding_provider: str = "openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_embedding_dimensions: int = 768

    def __post_init__(self) -> None:
        """Validate that postgres_dsn is set when storage_backend is postgres.

        Fails fast at config construction instead of deferring to an opaque
        asyncpg connection error when the factory tries to dial an empty DSN.
        """
        if self.storage_backend == "postgres" and not self.postgres_dsn:
            raise ValueError(
                "storage_backend='postgres' requires postgres_dsn to be set. "
                "Example: postgresql://user:pass@host:5432/dbname"
            )

    def resolved_embedding_model(self) -> str:
        """Return the model name for the active ``embedding_provider``."""
        if self.embedding_provider == "openai":
            return self.embedding_model
        if self.embedding_provider == "ollama":
            return self.ollama_embedding_model
        raise ValueError(f"Unknown embedding provider: {self.embedding_provider!r}")

    def resolved_embedding_dimensions(self) -> int:
        """Return the vector length for the active ``embedding_provider``."""
        if self.embedding_provider == "openai":
            return self.embedding_dimensions
        if self.embedding_provider == "ollama":
            return self.ollama_embedding_dimensions
        raise ValueError(f"Unknown embedding provider: {self.embedding_provider!r}")

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "constellation"),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS", "1536")),
            storage_backend=os.environ.get("STORAGE_BACKEND", "neo4j"),
            postgres_dsn=os.environ.get("POSTGRES_DSN", ""),
            embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "openai"),
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            ollama_embedding_model=os.environ.get(
                "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
            ),
            ollama_embedding_dimensions=int(
                os.environ.get("OLLAMA_EMBEDDING_DIMENSIONS", "768")
            ),
        )


_config: Config | None = None


def get_config() -> Config:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
```

- [ ] **Step 4: Run config tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: all existing tests still pass **plus** the new tests pass. No regressions.

- [ ] **Step 5: Commit**

```bash
git add src/telescope/config.py tests/test_config.py
git commit -m "feat(config): add EMBEDDING_PROVIDER + OLLAMA_* settings"
```

---

## Task 7: Add embeddings factory

**Files:**
- Create: `src/telescope/embeddings/factory.py`
- Create: `tests/embeddings/test_factory.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/embeddings/test_factory.py`:

```python
"""Tests for create_embedding_provider factory."""

from __future__ import annotations

import pytest

from telescope.config import Config
from telescope.embeddings.factory import create_embedding_provider
from telescope.embeddings.openai_provider import OpenAIEmbeddingProvider
from telescope.embeddings.ollama_provider import OllamaEmbeddingProvider


def _cfg(**kw) -> Config:
    return Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="pw",
        openai_api_key="sk-test",
        **kw,
    )


class TestCreateEmbeddingProvider:
    def test_returns_openai_when_provider_is_openai(self):
        cfg = _cfg(embedding_provider="openai")
        provider = create_embedding_provider(cfg)
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_openai_provider_uses_configured_model(self):
        cfg = _cfg(
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
        )
        provider = create_embedding_provider(cfg)
        assert provider.model_name == "text-embedding-3-large"
        assert provider.dimensions == 3072

    def test_returns_ollama_when_provider_is_ollama(self):
        cfg = _cfg(embedding_provider="ollama")
        provider = create_embedding_provider(cfg)
        assert isinstance(provider, OllamaEmbeddingProvider)

    def test_ollama_provider_uses_configured_model(self):
        cfg = _cfg(
            embedding_provider="ollama",
            ollama_embedding_model="mxbai-embed-large",
            ollama_embedding_dimensions=1024,
            ollama_base_url="http://host.docker.internal:11434",
        )
        provider = create_embedding_provider(cfg)
        assert provider.model_name == "mxbai-embed-large"
        assert provider.dimensions == 1024

    def test_raises_on_unknown_provider(self):
        cfg = _cfg(embedding_provider="nonsense")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/embeddings/test_factory.py -v`
Expected: `ModuleNotFoundError: No module named 'telescope.embeddings.factory'`.

- [ ] **Step 3: Create `src/telescope/embeddings/factory.py`**

```python
"""Embedding provider factory.

Selects and instantiates the concrete provider based on the active
``Config.embedding_provider`` setting. Providers are imported lazily so
that environments without one of the underlying client libraries
(unlikely in practice, but possible) don't crash on import just to use
the other provider.
"""

from __future__ import annotations

from telescope.config import Config
from telescope.embeddings.base import BaseEmbeddingProvider


def create_embedding_provider(config: Config) -> BaseEmbeddingProvider:
    """Return the concrete embedding provider for the active config."""
    provider = config.embedding_provider
    if provider == "openai":
        from telescope.embeddings.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
    if provider == "ollama":
        from telescope.embeddings.ollama_provider import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            base_url=config.ollama_base_url,
            model=config.ollama_embedding_model,
            dimensions=config.ollama_embedding_dimensions,
        )
    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        f"Expected 'openai' or 'ollama'."
    )
```

- [ ] **Step 4: Run factory tests to verify they pass**

Run: `pytest tests/embeddings/test_factory.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Full embeddings suite sanity check**

Run: `pytest tests/embeddings/ -v`
Expected: ~20 tests pass, all green.

- [ ] **Step 6: Commit**

```bash
git add src/telescope/embeddings/factory.py tests/embeddings/test_factory.py
git commit -m "feat(embeddings): add create_embedding_provider factory"
```

---

## Task 8: Refactor PostgresReadBackend to inject provider

**Files:**
- Modify: `src/telescope/backends/postgres.py` (lines 24-83 and any reference to `self._openai`)
- Modify: `tests/backends/test_postgres_read.py` (the `backend` fixture at lines ~18-25)

**What changes:**
- Constructor signature goes from `(dsn, openai_api_key, openai_base_url, embedding_model, embedding_dimensions)` to `(dsn, embedder)`.
- `_openai`, `_embedding_model`, `_embedding_dimensions` instance fields are removed.
- `_get_embedding` delegates to `self._embedder.embed_batch([text])` and returns `[0]`.
- Import of `AsyncOpenAI` is removed.

- [ ] **Step 1: Write a new failing test for the delegation**

Open `tests/backends/test_postgres_read.py`. At the top (after existing imports), add:

```python
from telescope.embeddings.base import BaseEmbeddingProvider


class _StubEmbedder(BaseEmbeddingProvider):
    """Minimal deterministic embedder used by the postgres backend fixture."""

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0] * self._dimensions for _ in texts]
```

Then modify the existing `backend` fixture (~lines 18-25) to:

```python
@pytest.fixture
def stub_embedder():
    return _StubEmbedder()


@pytest.fixture
def backend(mock_pool, stub_embedder):
    b = PostgresReadBackend(
        dsn="postgresql://test:test@localhost/test",
        embedder=stub_embedder,
    )
    b._pool = mock_pool
    return b
```

Finally, append a new test class at the bottom of the file:

```python
class TestPostgresEmbeddingDelegation:
    @pytest.mark.asyncio
    async def test_get_embedding_delegates_to_provider(self, backend, stub_embedder):
        vec = await backend._get_embedding("find login handler")
        assert vec == [0.0] * stub_embedder.dimensions
        assert stub_embedder.calls == [["find login handler"]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backends/test_postgres_read.py -v`
Expected: fails with `TypeError: __init__() got an unexpected keyword argument 'embedder'` (and/or missing `openai_api_key`).

- [ ] **Step 3: Modify `src/telescope/backends/postgres.py`**

Replace lines 1-83 (the imports block, class docstring, `__init__`, `connect`, `close`, `_require_pool`, and `_get_embedding`) with:

```python
"""PostgreSQL implementation of ReadBackend using asyncpg."""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from telescope.backends.base import ReadBackend
from telescope.embeddings.base import BaseEmbeddingProvider
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
        *,
        embedder: BaseEmbeddingProvider,
    ) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._embedder = embedder

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
        vectors = await self._embedder.embed_batch([text])
        return vectors[0]
```

Leave the rest of the file (from `_resolve_symbol` onward) unchanged.

- [ ] **Step 4: Update the second call site inside `test_postgres_read.py`**

There is a second `PostgresReadBackend(...)` construction in the same file at approximately line 1957, inside `test_connect_registers_pgvector_and_jsonb_codec`. It currently reads:

```python
backend = PostgresReadBackend(
    dsn="postgresql://test:test@localhost/test",
    openai_api_key="sk-test-key",
)
```

Replace it with:

```python
backend = PostgresReadBackend(
    dsn="postgresql://test:test@localhost/test",
    embedder=_StubEmbedder(),
)
```

The `_StubEmbedder` class is already defined at the top of the file from Step 1, so no extra import is needed.

- [ ] **Step 5: Update `tests/test_postgres_contract_integration.py`**

This file constructs the backend around line 37. Current form:

```python
backend = PostgresReadBackend(
    dsn=postgres_dsn,
    openai_api_key="sk-test-contract-key",
    embedding_model="text-embedding-3-small",
    embedding_dimensions=1536,
)
```

Replace with:

```python
from telescope.embeddings.base import BaseEmbeddingProvider


class _StubEmbedder(BaseEmbeddingProvider):
    """Stub embedder for contract tests — semantic search is covered in
    the unit tests; these contract tests exercise the graph read path."""

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def dimensions(self) -> int:
        return 1536

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]


# ... inside the fixture:
backend = PostgresReadBackend(
    dsn=postgres_dsn,
    embedder=_StubEmbedder(),
)
```

Place the `_StubEmbedder` class and its import just above the `pg_read_backend` fixture definition.

- [ ] **Step 6: Run postgres tests to verify everything passes**

Run: `pytest tests/backends/test_postgres_read.py -v`
Expected: all tests pass, including `TestPostgresEmbeddingDelegation` and `test_connect_registers_pgvector_and_jsonb_codec`.

Run: `pytest tests/test_postgres_contract_integration.py -v --co`
Expected: collection succeeds (no ImportError). Don't worry about the actual run results here — this file is gated by `postgres_integration` marker and needs Docker.

- [ ] **Step 7: Grep for any remaining stale call sites**

Run: `grep -rn "openai_api_key" src/telescope/backends/ tests/ || echo "clean"`
Expected: only hits inside `src/telescope/embeddings/openai_provider.py`, `src/telescope/config.py`, `tests/test_config.py`, `tests/embeddings/test_openai_provider.py`, and `tests/backends/test_factory.py`. Nothing in `src/telescope/backends/postgres.py` or tests that construct backends.

- [ ] **Step 8: Commit**

```bash
git add src/telescope/backends/postgres.py tests/backends/test_postgres_read.py tests/test_postgres_contract_integration.py
git commit -m "refactor(postgres): inject embedding provider instead of hardcoding openai"
```

---

## Task 9: Refactor Neo4jReadBackend to inject provider

**Files:**
- Modify: `src/telescope/backends/neo4j.py` (lines 78-140 — `__init__`, `connect`, `close`, `_get_embedding`)
- Modify: `tests/backends/test_neo4j.py` (14 `Neo4jReadBackend()` call sites; delete 6 tests that poke `_openai` or patch `AsyncOpenAI`)
- Modify: `tests/conftest.py:357-373` (the `live_graph_client` fixture that patches `AsyncOpenAI`)

**What changes in `neo4j.py`:**
- Constructor now takes `embedder: BaseEmbeddingProvider` (keyword-only).
- `self._openai` field is removed.
- `connect()` no longer constructs `AsyncOpenAI`.
- `close()` no longer closes `_openai`.
- `_get_embedding` delegates to `self._embedder.embed_batch([text])`.
- `from openai import AsyncOpenAI` import is removed.

**What changes in `test_neo4j.py`:**
The current file has these patterns to remove/update:

1. **Call sites that must be updated from `Neo4jReadBackend()` → `Neo4jReadBackend(embedder=_StubEmbedder())`** at lines 49, 62, 74, 89, 103, 115, 123, 131, 149, 162, 175, 191, 209, 259 (14 sites total).

2. **Tests to DELETE ENTIRELY** (they test OpenAI-coupling that no longer exists):
   - `test_connect_creates_openai_client` (around line 66)
   - `test_connect_openai_client_uses_api_key` (around line 78)
   - `test_connect_passes_base_url_when_set` (around line 80)
   - `test_connect_omits_base_url_when_none` (around line 95)
   - `test_close_calls_openai_close` (around line 120) — reaches into `client._openai = mock_openai_client`
   - `test_get_embedding_calls_openai` (around line 186) — reaches into `client._openai = mock_openai_client`
   - `test_get_embedding_returns_expected_vector` (around line 200) — same pattern with `mock_openai_response`

3. **Tests that poke `client._openai = AsyncMock()`** directly (e.g. at line 261) must be rewritten to pass the embedder through the constructor instead. The cleanest rewrite is to stop pre-setting `_openai` and instead construct with `embedder=_StubEmbedder()`.

4. **`patch("telescope.backends.neo4j.AsyncOpenAI")` lines** at 44, 59, 69, 84, 98 — these are inside `with` blocks, and since the `AsyncOpenAI` import is deleted from the source module, the patches become `AttributeError`s. Delete each `patch("telescope.backends.neo4j.AsyncOpenAI")` line from the `with` chain. (Some of those `with` blocks belong to deleted tests and go away entirely; others need the patch line removed while keeping the `AsyncGraphDatabase` patch.)

- [ ] **Step 1: Add `_StubEmbedder` helper to `tests/backends/test_neo4j.py`**

At the top of the file (after existing imports), add:

```python
from telescope.embeddings.base import BaseEmbeddingProvider


class _StubEmbedder(BaseEmbeddingProvider):
    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0] * self._dimensions for _ in texts]
```

- [ ] **Step 2: Delete the 6 OpenAI-coupled tests**

Delete these test methods in their entirety (including decorators and docstrings):

1. `test_connect_creates_openai_client`
2. `test_connect_openai_client_uses_api_key` (if it exists as a separate method)
3. `test_connect_passes_base_url_when_set`
4. `test_connect_omits_base_url_when_none`
5. `test_close_calls_openai_close`
6. `test_get_embedding_calls_openai`
7. `test_get_embedding_returns_expected_vector`

Verify they're gone:

Run: `grep -n "openai" tests/backends/test_neo4j.py`
Expected: only `openai_api_key="sk-test-key"` in the Config constructor (a test fixture) and the import line of `_StubEmbedder` may mention nothing with openai. No `test_*openai*` methods, no `_openai` attribute access, no `AsyncOpenAI` patches.

- [ ] **Step 3: Update all remaining `Neo4jReadBackend()` call sites**

At the 14 line numbers above (minus any whose containing test was just deleted), change `Neo4jReadBackend()` → `Neo4jReadBackend(embedder=_StubEmbedder())`.

After deletions and updates, re-grep:

Run: `grep -n "Neo4jReadBackend(" tests/backends/test_neo4j.py`
Expected: every hit passes `embedder=_StubEmbedder()` (or a similar embedder argument). No zero-arg constructions remain.

- [ ] **Step 4: Strip remaining `patch("telescope.backends.neo4j.AsyncOpenAI")` lines**

In every surviving `with` block, delete the `patch("telescope.backends.neo4j.AsyncOpenAI")` clause while keeping the other patches (e.g. `patch("telescope.backends.neo4j.AsyncGraphDatabase")`).

Run: `grep -n "AsyncOpenAI" tests/backends/test_neo4j.py`
Expected: zero hits.

- [ ] **Step 5: Update `tests/conftest.py` `live_graph_client` fixture**

The fixture at lines 357-373 is the **live** integration fixture — it runs against a real Neo4j instance (guarded by `_require_integration()`). The plan must NOT mock `AsyncGraphDatabase` here. Drop only the `AsyncOpenAI` patch and inject an embedder stub.

Before:

```python
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
```

After:

```python
@pytest.fixture()
async def live_graph_client():
    """Create a real Telescope Neo4jReadBackend against the local Neo4j instance."""
    _require_integration()

    from telescope.embeddings.base import BaseEmbeddingProvider

    class _StubEmbedder(BaseEmbeddingProvider):
        @property
        def model_name(self) -> str:
            return "stub"

        @property
        def dimensions(self) -> int:
            return 1536

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 1536 for _ in texts]

    with patch("telescope.backends.neo4j.get_config", return_value=_neo4j_config()):
        from telescope.backends.neo4j import Neo4jReadBackend

        client = Neo4jReadBackend(embedder=_StubEmbedder())
        await client.connect()
        try:
            yield client
        finally:
            await client.close()
```

Note: we use a stub embedder because the live contract tests exercise the graph read path, not semantic search. If a contract test does need real semantic search, it can override with a real OpenAI/Ollama provider.

- [ ] **Step 6: Add a delegation test**

Append to `tests/backends/test_neo4j.py`:

```python
class TestNeo4jEmbeddingDelegation:
    @pytest.mark.asyncio
    async def test_get_embedding_delegates_to_provider(self, patched_config):
        embedder = _StubEmbedder()
        from telescope.backends.neo4j import Neo4jReadBackend

        backend = Neo4jReadBackend(embedder=embedder)
        vec = await backend._get_embedding("any query")

        assert vec == [0.0] * embedder.dimensions
        assert embedder.calls == [["any query"]]
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `pytest tests/backends/test_neo4j.py -v`
Expected: the new delegation test fails with a `TypeError` — `Neo4jReadBackend.__init__()` got an unexpected keyword argument 'embedder'. Deleted OpenAI tests are gone from the run.

- [ ] **Step 8: Modify `src/telescope/backends/neo4j.py`**

Replace lines 1-140 (imports through `_get_embedding`) with:

```python
"""Neo4j implementation of ReadBackend for querying Constellation's code knowledge graph."""

import logging
import re
from typing import Any

from neo4j import AsyncGraphDatabase

from telescope.config import get_config
from telescope.embeddings.base import BaseEmbeddingProvider
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

    def __init__(self, *, embedder: BaseEmbeddingProvider) -> None:
        self.config = get_config()
        self._driver = None
        self._embedder = embedder

    async def connect(self):
        """Connect to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )
        await self._driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {self.config.neo4j_uri}")

    async def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()

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
        vectors = await self._embedder.embed_batch([text])
        return vectors[0]
```

Leave the rest of the file (from `_file_pattern_to_regex` onward) unchanged.

- [ ] **Step 9: Run neo4j tests to verify they pass**

Run: `pytest tests/backends/test_neo4j.py -v`
Expected: all remaining tests pass (the deleted OpenAI-coupled tests are gone, the new delegation test passes, all untouched tests continue to pass).

- [ ] **Step 10: Check for the now-dead `mock_openai_client` fixture**

After deleting the `_openai`-reaching tests, the `mock_openai_client` and `mock_openai_response` fixtures in `tests/conftest.py` (around lines 79-97) may have zero consumers. Verify:

Run: `grep -rn "mock_openai_client\|mock_openai_response" tests/`
If only `conftest.py` lights up (no test_* file uses them), delete both fixtures. If any test still references them, leave them alone.

- [ ] **Step 11: Commit**

```bash
git add src/telescope/backends/neo4j.py tests/backends/test_neo4j.py tests/conftest.py
git commit -m "refactor(neo4j): inject embedding provider instead of hardcoding openai"
```

---

## Task 10: Update create_read_backend to build and inject the provider

**Files:**
- Modify: `src/telescope/backends/factory.py`
- Modify: `tests/backends/test_factory.py`

- [ ] **Step 1: Write the failing test**

Open `tests/backends/test_factory.py` and replace the entire file with:

```python
"""Tests for create_read_backend factory."""
import pytest
from unittest.mock import MagicMock, patch

from telescope.config import Config
from telescope.backends.factory import create_read_backend
from telescope.backends.base import ReadBackend
from telescope.embeddings.openai_provider import OpenAIEmbeddingProvider
from telescope.embeddings.ollama_provider import OllamaEmbeddingProvider


def _make_config(
    storage_backend: str = "neo4j",
    postgres_dsn: str = "",
    embedding_provider: str = "openai",
    **kw,
) -> Config:
    return Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test",
        openai_api_key="test-key",
        storage_backend=storage_backend,
        postgres_dsn=postgres_dsn,
        embedding_provider=embedding_provider,
        **kw,
    )


def test_factory_returns_neo4j_by_default():
    from telescope.backends.neo4j import Neo4jReadBackend
    config = _make_config()
    backend = create_read_backend(config)
    assert isinstance(backend, Neo4jReadBackend)
    assert isinstance(backend, ReadBackend)


def test_factory_raises_on_unknown_backend():
    config = _make_config(storage_backend="sqlite")
    with pytest.raises(ValueError, match="Unknown storage backend"):
        create_read_backend(config)


def test_factory_returns_postgres_when_configured():
    # Patch the class directly on the already-imported postgres module.
    # This avoids the brittle `sys.modules` replacement pattern — by the
    # time this test runs, `telescope.backends.postgres` is almost
    # certainly already in `sys.modules` via other test collection paths,
    # which would defeat a module-level stub. An attribute-level patch
    # targets the bound name the factory resolves at call time.
    with patch(
        "telescope.backends.postgres.PostgresReadBackend"
    ) as mock_postgres_cls:
        config = _make_config(
            storage_backend="postgres", postgres_dsn="postgresql://test@localhost/db"
        )
        create_read_backend(config)
        mock_postgres_cls.assert_called_once()
        kwargs = mock_postgres_cls.call_args.kwargs
        assert kwargs["dsn"] == "postgresql://test@localhost/db"
        assert "embedder" in kwargs
        assert isinstance(kwargs["embedder"], OpenAIEmbeddingProvider)


def test_factory_passes_ollama_embedder_when_provider_is_ollama():
    from telescope.backends.neo4j import Neo4jReadBackend

    config = _make_config(embedding_provider="ollama")
    backend = create_read_backend(config)
    assert isinstance(backend, Neo4jReadBackend)
    assert isinstance(backend._embedder, OllamaEmbeddingProvider)


def test_factory_passes_openai_embedder_when_provider_is_openai():
    from telescope.backends.neo4j import Neo4jReadBackend

    config = _make_config(embedding_provider="openai")
    backend = create_read_backend(config)
    assert isinstance(backend, Neo4jReadBackend)
    assert isinstance(backend._embedder, OpenAIEmbeddingProvider)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backends/test_factory.py -v`
Expected: failures because `create_read_backend` still uses the old signature and doesn't build an embedder.

- [ ] **Step 3: Replace `src/telescope/backends/factory.py`**

```python
"""Factory for creating ReadBackend instances."""

from __future__ import annotations

from telescope.config import Config
from telescope.backends.base import ReadBackend
from telescope.embeddings.factory import create_embedding_provider


def create_read_backend(config: Config) -> ReadBackend:
    """Return the configured read backend.

    Reads ``config.storage_backend`` to select the implementation and
    ``config.embedding_provider`` to pick the embedding provider that
    the backend uses for query-side vector generation.
    """
    embedder = create_embedding_provider(config)

    if config.storage_backend == "postgres":
        from telescope.backends.postgres import PostgresReadBackend
        return PostgresReadBackend(
            dsn=config.postgres_dsn,
            embedder=embedder,
        )
    elif config.storage_backend == "neo4j":
        from telescope.backends.neo4j import Neo4jReadBackend
        return Neo4jReadBackend(embedder=embedder)
    else:
        raise ValueError(
            f"Unknown storage backend: {config.storage_backend!r}. "
            f"Expected 'neo4j' or 'postgres'."
        )
```

- [ ] **Step 4: Run factory tests to verify they pass**

Run: `pytest tests/backends/test_factory.py -v`
Expected: all tests pass.

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: **zero failures**. Every unit test green. Any remaining failures point to a call site that still uses the old backend signature — fix and re-run.

- [ ] **Step 6: Commit**

```bash
git add src/telescope/backends/factory.py tests/backends/test_factory.py
git commit -m "refactor(factory): build embedding provider and inject into backends"
```

---

## Task 11: Smoke test against a real Ollama server (manual)

This task has no failing test — it's a manual verification that query-time embeddings produced by telescope match what Constellation wrote during indexing. Mark the checkboxes as you confirm each step.

**Prerequisites:** Constellation must have already indexed a repository with `EMBEDDING_PROVIDER=ollama` (so the Postgres `code_symbol_embeddings.embedding` column holds 768-dim vectors from `nomic-embed-text`).

- [ ] **Step 1: Verify Ollama is running and reachable**

Run: `curl -s http://localhost:11434/api/tags | head -5`
Expected: JSON listing installed models, including `nomic-embed-text`.

If the model is missing: `ollama pull nomic-embed-text`

- [ ] **Step 2: Test telescope's Ollama provider end-to-end from a REPL**

Run:
```bash
cd /Users/d.sriram/Desktop/personal/telescope
python -c "
import asyncio
from telescope.embeddings.ollama_provider import OllamaEmbeddingProvider

async def main():
    p = OllamaEmbeddingProvider()
    vecs = await p.embed_batch(['authentication middleware', 'database connection pool'])
    print('vecs:', len(vecs))
    print('dim 0:', len(vecs[0]))
    print('dim 1:', len(vecs[1]))

asyncio.run(main())
"
```
Expected output:
```
vecs: 2
dim 0: 768
dim 1: 768
```

- [ ] **Step 3: Update the telescope MCP config in `~/.claude.json`**

Open `~/.claude.json`, find the `"telescope"` entry under `mcpServers`, and add the Ollama env vars:

```json
"telescope": {
  "command": "/Users/d.sriram/Desktop/personal/telescope/.venv/bin/telescope",
  "args": [],
  "env": {
    "STORAGE_BACKEND": "postgres",
    "POSTGRES_DSN": "postgresql://constellation:secret@127.0.0.1:5433/constellation",
    "EMBEDDING_PROVIDER": "ollama",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
    "OLLAMA_EMBEDDING_DIMENSIONS": "768"
  }
}
```

You can leave `OPENAI_API_KEY` and `OPENAI_BASE_URL` in the env block — telescope's `Config` still reads `OPENAI_API_KEY` unconditionally, so removing it causes the field to default to `""` which is fine, but an explicit empty-string value (or the old key) is the safer choice. When `EMBEDDING_PROVIDER=ollama`, the OpenAI credentials are simply not used at request time.

**Watch out:** the MCP server runs *outside* a Docker container — `localhost` on the host machine. If you later move telescope into Docker, switch to `host.docker.internal`.

- [ ] **Step 4: Restart Claude Code / MCP client**

Fully quit and relaunch so the new env vars are picked up.

- [ ] **Step 5: Issue a semantic search from the MCP client**

Invoke the `search_code` MCP tool with a natural-language query against a repository that Constellation indexed with Ollama. Expected: non-empty results with plausible ranking.

If you get a pgvector "expected N dimensions, got M" error, the vectors in the DB and the query vectors are different sizes — this means Constellation and telescope disagree on the `OLLAMA_EMBEDDING_DIMENSIONS`. Re-check both .env files.

- [ ] **Step 6: Check nothing to commit**

Run: `git status`
Expected: clean working tree (this task was manual verification only).

---

## Task 12: Document the new env vars in README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Inspect current README**

Run: `grep -n "OPENAI_API_KEY\|EMBEDDING_" README.md | head`
Identify the section that lists environment variables.

- [ ] **Step 2: Add an Ollama section**

Under the existing env vars section, append:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document EMBEDDING_PROVIDER + OLLAMA_* env vars"
```

---

## Final verification

- [ ] **Full test suite clean**

Run: `pytest -v`
Expected: 0 failures, 0 errors. Count the total to make sure it matches or exceeds the pre-change total.

- [ ] **No stale imports left behind**

Run: `grep -rn "AsyncOpenAI" src/telescope/`
Expected: only appears in `src/telescope/embeddings/openai_provider.py`. Nothing in `backends/`.

- [ ] **Lint / type check (if configured)**

Run: `ruff check src tests 2>/dev/null || echo "no ruff configured"`
Run: `mypy src 2>/dev/null || echo "no mypy configured"`
Resolve anything that's not a pre-existing issue.

- [ ] **Git log is clean**

Run: `git log --oneline -20`
Expected: 11 new commits (or 10 if README is part of an earlier commit), each with a scoped conventional-commit message, in logical order.

---

## Rollback plan

If anything goes wrong at the MCP-client smoke test (Task 11) and you need to revert:

```bash
git revert HEAD~10..HEAD  # adjust number to match how many commits were added
```

Then restart the MCP client. Telescope returns to the OpenAI-only path.

The code changes are fully test-covered and additive — no existing public interface breaks as long as you keep `EMBEDDING_PROVIDER` unset (defaults to `openai`).
