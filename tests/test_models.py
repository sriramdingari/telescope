"""Tests for Telescope data models."""

import pytest
from telescope.models import (
    CallGraphNode,
    ClassHierarchy,
    CodebaseOverview,
    CodeEntity,
    FileContext,
    FunctionContext,
    ImpactResult,
)


class TestCodeEntity:
    """Tests for CodeEntity dataclass."""

    def test_required_fields_only(self):
        entity = CodeEntity(name="my_func", file_path="/src/foo.py")
        assert entity.name == "my_func"
        assert entity.file_path == "/src/foo.py"

    def test_all_fields(self):
        entity = CodeEntity(
            name="my_func",
            file_path="/src/foo.py",
            repository="my-repo",
            line_start=10,
            line_end=20,
            code="def my_func(): pass",
            signature="my_func() -> None",
            docstring="Does stuff.",
            score=0.95,
            entity_type="function",
        )
        assert entity.name == "my_func"
        assert entity.file_path == "/src/foo.py"
        assert entity.repository == "my-repo"
        assert entity.line_start == 10
        assert entity.line_end == 20
        assert entity.code == "def my_func(): pass"
        assert entity.signature == "my_func() -> None"
        assert entity.docstring == "Does stuff."
        assert entity.score == 0.95
        assert entity.entity_type == "function"

    def test_default_repository_is_none(self):
        entity = CodeEntity(name="f", file_path="/a.py")
        assert entity.repository is None

    def test_default_line_start_is_none(self):
        entity = CodeEntity(name="f", file_path="/a.py")
        assert entity.line_start is None

    def test_default_line_end_is_none(self):
        entity = CodeEntity(name="f", file_path="/a.py")
        assert entity.line_end is None

    def test_default_code_is_none(self):
        entity = CodeEntity(name="f", file_path="/a.py")
        assert entity.code is None

    def test_default_signature_is_none(self):
        entity = CodeEntity(name="f", file_path="/a.py")
        assert entity.signature is None

    def test_default_docstring_is_none(self):
        entity = CodeEntity(name="f", file_path="/a.py")
        assert entity.docstring is None

    def test_default_score_is_zero(self):
        entity = CodeEntity(name="f", file_path="/a.py")
        assert entity.score == 0.0

    def test_default_entity_type_is_method(self):
        entity = CodeEntity(name="f", file_path="/a.py")
        assert entity.entity_type == "method"


class TestCallGraphNode:
    """Tests for CallGraphNode dataclass."""

    def test_required_fields_only(self):
        node = CallGraphNode(name="caller", file_path="/src/bar.py")
        assert node.name == "caller"
        assert node.file_path == "/src/bar.py"

    def test_all_fields(self):
        node = CallGraphNode(
            name="caller",
            file_path="/src/bar.py",
            repository="repo-x",
            signature="caller() -> None",
            line_start=5,
            depth=3,
            is_test=True,
            is_endpoint=True,
            entity_type="reference",
            relationship_type="CALLS",
        )
        assert node.name == "caller"
        assert node.file_path == "/src/bar.py"
        assert node.repository == "repo-x"
        assert node.signature == "caller() -> None"
        assert node.line_start == 5
        assert node.depth == 3
        assert node.is_test is True
        assert node.is_endpoint is True
        assert node.entity_type == "reference"
        assert node.relationship_type == "CALLS"

    def test_default_repository_is_none(self):
        node = CallGraphNode(name="n", file_path="/a.py")
        assert node.repository is None

    def test_default_signature_is_none(self):
        node = CallGraphNode(name="n", file_path="/a.py")
        assert node.signature is None

    def test_default_line_start_is_none(self):
        node = CallGraphNode(name="n", file_path="/a.py")
        assert node.line_start is None

    def test_default_depth_is_one(self):
        node = CallGraphNode(name="n", file_path="/a.py")
        assert node.depth == 1

    def test_default_is_test_is_false(self):
        node = CallGraphNode(name="n", file_path="/a.py")
        assert node.is_test is False

    def test_default_is_endpoint_is_false(self):
        node = CallGraphNode(name="n", file_path="/a.py")
        assert node.is_endpoint is False

    def test_default_entity_type_is_method(self):
        node = CallGraphNode(name="n", file_path="/a.py")
        assert node.entity_type == "method"

    def test_default_relationship_type_is_call(self):
        node = CallGraphNode(name="n", file_path="/a.py")
        assert node.relationship_type == "CALLS"


class TestFunctionContext:
    """Tests for FunctionContext dataclass."""

    def test_required_fields_only(self):
        ctx = FunctionContext(
            name="do_thing",
            full_name="module.do_thing",
            file_path="/src/mod.py",
        )
        assert ctx.name == "do_thing"
        assert ctx.full_name == "module.do_thing"
        assert ctx.file_path == "/src/mod.py"

    def test_all_fields(self):
        caller = CallGraphNode(name="caller", file_path="/a.py")
        callee = CallGraphNode(name="callee", file_path="/b.py")
        ctx = FunctionContext(
            name="do_thing",
            full_name="module.do_thing",
            file_path="/src/mod.py",
            repository="my-repo",
            code="def do_thing(): ...",
            signature="do_thing() -> None",
            docstring="Does a thing.",
            class_name="MyClass",
            callers=[caller],
            callees=[callee],
        )
        assert ctx.repository == "my-repo"
        assert ctx.code == "def do_thing(): ..."
        assert ctx.signature == "do_thing() -> None"
        assert ctx.docstring == "Does a thing."
        assert ctx.class_name == "MyClass"
        assert ctx.callers == [caller]
        assert ctx.callees == [callee]

    def test_default_repository_is_none(self):
        ctx = FunctionContext(name="f", full_name="m.f", file_path="/a.py")
        assert ctx.repository is None

    def test_default_code_is_none(self):
        ctx = FunctionContext(name="f", full_name="m.f", file_path="/a.py")
        assert ctx.code is None

    def test_default_signature_is_none(self):
        ctx = FunctionContext(name="f", full_name="m.f", file_path="/a.py")
        assert ctx.signature is None

    def test_default_docstring_is_none(self):
        ctx = FunctionContext(name="f", full_name="m.f", file_path="/a.py")
        assert ctx.docstring is None

    def test_default_class_name_is_none(self):
        ctx = FunctionContext(name="f", full_name="m.f", file_path="/a.py")
        assert ctx.class_name is None

    def test_default_callers_is_empty_list(self):
        ctx = FunctionContext(name="f", full_name="m.f", file_path="/a.py")
        assert ctx.callers == []

    def test_default_callees_is_empty_list(self):
        ctx = FunctionContext(name="f", full_name="m.f", file_path="/a.py")
        assert ctx.callees == []

    def test_callers_list_independence(self):
        ctx1 = FunctionContext(name="f", full_name="m.f", file_path="/a.py")
        ctx2 = FunctionContext(name="g", full_name="m.g", file_path="/b.py")
        ctx1.callers.append(CallGraphNode(name="x", file_path="/x.py"))
        assert ctx2.callers == []

    def test_callees_list_independence(self):
        ctx1 = FunctionContext(name="f", full_name="m.f", file_path="/a.py")
        ctx2 = FunctionContext(name="g", full_name="m.g", file_path="/b.py")
        ctx1.callees.append(CallGraphNode(name="y", file_path="/y.py"))
        assert ctx2.callees == []


class TestClassHierarchy:
    """Tests for ClassHierarchy dataclass."""

    def test_required_fields_only(self):
        hier = ClassHierarchy(
            name="MyClass",
            full_name="com.example.MyClass",
            file_path="/src/MyClass.java",
        )
        assert hier.name == "MyClass"
        assert hier.full_name == "com.example.MyClass"
        assert hier.file_path == "/src/MyClass.java"

    def test_all_fields(self):
        hier = ClassHierarchy(
            name="MyClass",
            full_name="com.example.MyClass",
            file_path="/src/MyClass.java",
            repository="java-repo",
            is_interface=True,
            parents=["BaseClass"],
            children=["SubClass"],
            interfaces=["Runnable"],
            implementors=["ConcreteImpl"],
            methods=["doWork", "init"],
            fields=["count"],
            constructors=["MyClass(int)"],
        )
        assert hier.repository == "java-repo"
        assert hier.is_interface is True
        assert hier.parents == ["BaseClass"]
        assert hier.children == ["SubClass"]
        assert hier.interfaces == ["Runnable"]
        assert hier.implementors == ["ConcreteImpl"]
        assert hier.methods == ["doWork", "init"]
        assert hier.fields == ["count"]
        assert hier.constructors == ["MyClass(int)"]

    def test_default_repository_is_none(self):
        hier = ClassHierarchy(name="C", full_name="p.C", file_path="/C.java")
        assert hier.repository is None

    def test_default_is_interface_is_false(self):
        hier = ClassHierarchy(name="C", full_name="p.C", file_path="/C.java")
        assert hier.is_interface is False

    def test_default_parents_is_empty_list(self):
        hier = ClassHierarchy(name="C", full_name="p.C", file_path="/C.java")
        assert hier.parents == []

    def test_default_children_is_empty_list(self):
        hier = ClassHierarchy(name="C", full_name="p.C", file_path="/C.java")
        assert hier.children == []

    def test_default_interfaces_is_empty_list(self):
        hier = ClassHierarchy(name="C", full_name="p.C", file_path="/C.java")
        assert hier.interfaces == []

    def test_default_implementors_is_empty_list(self):
        hier = ClassHierarchy(name="C", full_name="p.C", file_path="/C.java")
        assert hier.implementors == []

    def test_default_methods_is_empty_list(self):
        hier = ClassHierarchy(name="C", full_name="p.C", file_path="/C.java")
        assert hier.methods == []

    def test_default_fields_is_empty_list(self):
        hier = ClassHierarchy(name="C", full_name="p.C", file_path="/C.java")
        assert hier.fields == []

    def test_default_constructors_is_empty_list(self):
        hier = ClassHierarchy(name="C", full_name="p.C", file_path="/C.java")
        assert hier.constructors == []

    def test_list_independence(self):
        h1 = ClassHierarchy(name="A", full_name="p.A", file_path="/A.java")
        h2 = ClassHierarchy(name="B", full_name="p.B", file_path="/B.java")
        h1.parents.append("Base")
        h1.children.append("Sub")
        h1.methods.append("run")
        h1.constructors.append("A()")
        assert h2.parents == []
        assert h2.children == []
        assert h2.methods == []
        assert h2.constructors == []


class TestCodebaseOverview:
    """Tests for CodebaseOverview dataclass."""

    def test_no_required_fields(self):
        overview = CodebaseOverview()
        assert overview is not None

    def test_all_fields(self):
        overview = CodebaseOverview(
            total_files=100,
            total_classes=50,
            total_interfaces=10,
            total_methods=300,
            total_constructors=25,
            total_fields=80,
            total_packages=12,
            total_hooks=6,
            total_references=14,
            total_exports=19,
            languages=["Java", "Python"],
            packages=["com.example"],
            top_level_classes=["Main"],
            entry_points=["main()"],
        )
        assert overview.total_files == 100
        assert overview.total_classes == 50
        assert overview.total_interfaces == 10
        assert overview.total_methods == 300
        assert overview.total_constructors == 25
        assert overview.total_fields == 80
        assert overview.total_packages == 12
        assert overview.total_hooks == 6
        assert overview.total_references == 14
        assert overview.total_exports == 19
        assert overview.languages == ["Java", "Python"]
        assert overview.packages == ["com.example"]
        assert overview.top_level_classes == ["Main"]
        assert overview.entry_points == ["main()"]

    def test_default_total_files_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_files == 0

    def test_default_total_classes_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_classes == 0

    def test_default_total_methods_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_methods == 0

    def test_default_total_interfaces_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_interfaces == 0

    def test_default_total_constructors_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_constructors == 0

    def test_default_total_fields_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_fields == 0

    def test_default_total_packages_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_packages == 0

    def test_default_total_hooks_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_hooks == 0

    def test_default_total_references_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_references == 0

    def test_default_total_exports_is_zero(self):
        overview = CodebaseOverview()
        assert overview.total_exports == 0

    def test_default_languages_is_empty_list(self):
        overview = CodebaseOverview()
        assert overview.languages == []

    def test_default_packages_is_empty_list(self):
        overview = CodebaseOverview()
        assert overview.packages == []

    def test_default_top_level_classes_is_empty_list(self):
        overview = CodebaseOverview()
        assert overview.top_level_classes == []

    def test_default_entry_points_is_empty_list(self):
        overview = CodebaseOverview()
        assert overview.entry_points == []

    def test_list_independence(self):
        o1 = CodebaseOverview()
        o2 = CodebaseOverview()
        o1.languages.append("Java")
        o1.packages.append("com.example")
        assert o2.languages == []
        assert o2.packages == []


class TestFileContext:
    """Tests for FileContext dataclass."""

    def test_required_fields_only(self):
        context = FileContext(name="foo.py", file_path="src/foo.py")
        assert context.name == "foo.py"
        assert context.file_path == "src/foo.py"

    def test_all_fields(self):
        export = CodeEntity(name="foo", file_path="src/foo.py", entity_type="method")
        context = FileContext(
            name="foo.py",
            file_path="src/foo.py",
            repository="repo",
            language="Python",
            packages=["pkg.core"],
            exports=[export],
            classes=["Foo"],
            interfaces=["Bar"],
            top_level_methods=["foo"],
            hooks=["useState"],
        )
        assert context.repository == "repo"
        assert context.language == "Python"
        assert context.packages == ["pkg.core"]
        assert context.exports == [export]
        assert context.classes == ["Foo"]
        assert context.interfaces == ["Bar"]
        assert context.top_level_methods == ["foo"]
        assert context.hooks == ["useState"]

    def test_default_list_fields_are_empty(self):
        context = FileContext(name="foo.py", file_path="src/foo.py")
        assert context.packages == []
        assert context.exports == []
        assert context.classes == []
        assert context.interfaces == []
        assert context.top_level_methods == []
        assert context.hooks == []


class TestImpactResult:
    """Tests for ImpactResult dataclass."""

    def test_required_fields_only(self):
        result = ImpactResult(target_name="my_func", target_file="/src/mod.py")
        assert result.target_name == "my_func"
        assert result.target_file == "/src/mod.py"

    def test_all_fields(self):
        test_node = CallGraphNode(name="test_it", file_path="/tests/test_mod.py", is_test=True)
        endpoint_node = CallGraphNode(name="api_view", file_path="/api/views.py", is_endpoint=True)
        other_node = CallGraphNode(name="helper", file_path="/src/helpers.py")
        result = ImpactResult(
            target_name="my_func",
            target_file="/src/mod.py",
            target_repository="my-repo",
            total_callers=10,
            test_count=3,
            endpoint_count=2,
            affected_tests=[test_node],
            affected_endpoints=[endpoint_node],
            other_callers=[other_node],
            truncated=True,
        )
        assert result.target_repository == "my-repo"
        assert result.total_callers == 10
        assert result.test_count == 3
        assert result.endpoint_count == 2
        assert result.affected_tests == [test_node]
        assert result.affected_endpoints == [endpoint_node]
        assert result.other_callers == [other_node]
        assert result.truncated is True

    def test_default_target_repository_is_none(self):
        result = ImpactResult(target_name="f", target_file="/a.py")
        assert result.target_repository is None

    def test_default_total_callers_is_zero(self):
        result = ImpactResult(target_name="f", target_file="/a.py")
        assert result.total_callers == 0

    def test_default_test_count_is_zero(self):
        result = ImpactResult(target_name="f", target_file="/a.py")
        assert result.test_count == 0

    def test_default_endpoint_count_is_zero(self):
        result = ImpactResult(target_name="f", target_file="/a.py")
        assert result.endpoint_count == 0

    def test_default_affected_tests_is_empty_list(self):
        result = ImpactResult(target_name="f", target_file="/a.py")
        assert result.affected_tests == []

    def test_default_affected_endpoints_is_empty_list(self):
        result = ImpactResult(target_name="f", target_file="/a.py")
        assert result.affected_endpoints == []

    def test_default_other_callers_is_empty_list(self):
        result = ImpactResult(target_name="f", target_file="/a.py")
        assert result.other_callers == []

    def test_default_truncated_is_false(self):
        result = ImpactResult(target_name="f", target_file="/a.py")
        assert result.truncated is False

    def test_list_independence(self):
        r1 = ImpactResult(target_name="f", target_file="/a.py")
        r2 = ImpactResult(target_name="g", target_file="/b.py")
        node = CallGraphNode(name="x", file_path="/x.py")
        r1.affected_tests.append(node)
        r1.affected_endpoints.append(node)
        r1.other_callers.append(node)
        assert r2.affected_tests == []
        assert r2.affected_endpoints == []
        assert r2.other_callers == []
