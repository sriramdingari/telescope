"""Tests that ReadBackend ABC enforces the correct interface."""
import pytest
from telescope.backends.base import ReadBackend


def test_read_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        ReadBackend()


def test_concrete_read_backend_all_methods_implemented():
    class FakeBackend(ReadBackend):
        async def connect(self): pass
        async def close(self): pass
        async def search_code(self, query, **kw): return []
        async def find_symbols(self, query, **kw): return []
        async def get_callers(self, method_name, **kw): return []
        async def get_callees(self, method_name, **kw): return []
        async def get_function_context(self, method_name, **kw): return None
        async def get_class_hierarchy(self, class_name, **kw): return None
        async def get_package_context(self, package_name, **kw): return None
        async def get_file_context(self, file_path, **kw): return None
        async def get_hook_usage(self, hook_name, **kw): return []
        async def get_impact(self, method_name, **kw): return None
        async def list_repositories(self): return []
        async def get_repository_context(self, repository): return None
        async def get_codebase_overview(self, repository=None, include_packages=False): ...

    backend = FakeBackend()
    assert backend is not None
