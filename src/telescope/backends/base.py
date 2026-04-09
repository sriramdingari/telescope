"""Abstract base class for Telescope read backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

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


class ReadBackend(ABC):
    """Interface for all Telescope storage read backends.

    Implementations: Neo4jReadBackend, PostgresReadBackend.
    All public methods use keyword-only arguments after the primary positional arg.
    """

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def search_code(
        self, query: str, *,
        limit: int = 10,
        entity_type: str | None = None,
        file_pattern: str | None = None,
        repository: str | None = None,
        code_mode: str = "preview",
        language: str | None = None,
        stereotype: str | None = None,
    ) -> list[CodeEntity]: ...

    @abstractmethod
    async def find_symbols(
        self, query: str, *,
        entity_types: list[str] | None = None,
        repository: str | None = None,
        file_pattern: str | None = None,
        limit: int = 20,
        exact: bool = False,
        language: str | None = None,
        stereotype: str | None = None,
        code_mode: str = "none",
    ) -> list[CodeEntity]: ...

    @abstractmethod
    async def get_callers(
        self, method_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
        entity_id: str | None = None,
        depth: int = 1,
        limit: int = 50,
    ) -> list[CallGraphNode]: ...

    @abstractmethod
    async def get_callees(
        self, method_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
        entity_id: str | None = None,
        depth: int = 1,
        limit: int = 50,
    ) -> list[CallGraphNode]: ...

    @abstractmethod
    async def get_function_context(
        self, method_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
    ) -> FunctionContext | None: ...

    @abstractmethod
    async def get_class_hierarchy(
        self, class_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
    ) -> ClassHierarchy | None: ...

    @abstractmethod
    async def get_package_context(
        self, package_name: str, *,
        repository: str | None = None,
    ) -> PackageContext | None: ...

    @abstractmethod
    async def get_file_context(
        self, file_path: str, *,
        repository: str | None = None,
    ) -> FileContext | None: ...

    @abstractmethod
    async def get_hook_usage(
        self, hook_name: str, *,
        repository: str | None = None,
        file_pattern: str | None = None,
        language: str | None = None,
        stereotype: str | None = None,
        limit: int = 50,
    ) -> list[CallGraphNode]: ...

    @abstractmethod
    async def get_impact(
        self, method_name: str, *,
        repository: str | None = None,
        file_path: str | None = None,
        depth: int = 10,
        summary_only: bool = False,
        limit: int | None = None,
    ) -> ImpactResult | None: ...

    @abstractmethod
    async def list_repositories(self) -> list[dict]: ...

    @abstractmethod
    async def get_repository_context(
        self, repository: str,
    ) -> RepositoryContext | None: ...

    @abstractmethod
    async def get_codebase_overview(
        self,
        repository: str | None = None,
        include_packages: bool = False,
    ) -> CodebaseOverview: ...
