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
