"""Data models for Telescope query results."""

from dataclasses import dataclass, field


@dataclass
class CodeEntity:
    """A code entity found via search."""
    name: str
    file_path: str
    repository: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    code: str | None = None
    signature: str | None = None
    docstring: str | None = None
    score: float = 0.0
    entity_type: str = "method"


@dataclass
class CallGraphNode:
    """A node in a call graph traversal."""
    name: str
    file_path: str
    repository: str | None = None
    signature: str | None = None
    line_start: int | None = None
    depth: int = 1
    is_test: bool = False
    is_endpoint: bool = False
    entity_type: str = "method"
    relationship_type: str = "CALLS"


@dataclass
class FunctionContext:
    """Full context for a function/method."""
    name: str
    full_name: str
    file_path: str
    repository: str | None = None
    code: str | None = None
    signature: str | None = None
    docstring: str | None = None
    class_name: str | None = None
    callers: list[CallGraphNode] = field(default_factory=list)
    callees: list[CallGraphNode] = field(default_factory=list)


@dataclass
class ClassHierarchy:
    """Class or interface inheritance information."""
    name: str
    full_name: str
    file_path: str
    repository: str | None = None
    is_interface: bool = False
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    implementors: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    constructors: list[str] = field(default_factory=list)


@dataclass
class CodebaseOverview:
    """High-level codebase statistics."""
    total_files: int = 0
    total_classes: int = 0
    total_interfaces: int = 0
    total_methods: int = 0
    total_constructors: int = 0
    total_fields: int = 0
    total_packages: int = 0
    total_hooks: int = 0
    total_references: int = 0
    total_exports: int = 0
    languages: list[str] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)
    top_level_classes: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)


@dataclass
class FileContext:
    """High-level graph context for a file."""
    name: str
    file_path: str
    repository: str | None = None
    language: str | None = None
    packages: list[str] = field(default_factory=list)
    exports: list[CodeEntity] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    top_level_methods: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)


@dataclass
class ImpactResult:
    """Result of blast radius / impact analysis."""
    target_name: str
    target_file: str
    target_repository: str | None = None
    total_callers: int = 0
    test_count: int = 0
    endpoint_count: int = 0
    affected_tests: list[CallGraphNode] = field(default_factory=list)
    affected_endpoints: list[CallGraphNode] = field(default_factory=list)
    other_callers: list[CallGraphNode] = field(default_factory=list)
    truncated: bool = False
