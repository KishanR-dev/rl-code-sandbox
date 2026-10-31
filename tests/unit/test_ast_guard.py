"""Unit tests for the AST static security guard."""

from __future__ import annotations

import pytest

from rl_sandbox.domain.errors import SecurityViolationError
from rl_sandbox.security.ast_guard import ASTGuard


@pytest.fixture
def guard() -> ASTGuard:
    return ASTGuard()


def test_valid_code(guard: ASTGuard) -> None:
    source = """
def add(a: int, b: int) -> int:
    x = [1, 2, 3]
    return a + b + sum(x)
"""
    guard.check(source)


@pytest.mark.parametrize(
    "module_name",
    [
        "os",
        "sys",
        "subprocess",
        "socket",
        "importlib",
        "pty",
        "shutil",
        "os.path",
        "sys.version",
        "subprocess.Popen",
    ],
)
def test_forbidden_import(guard: ASTGuard, module_name: str) -> None:
    source = f"import {module_name}"
    with pytest.raises(
        SecurityViolationError, match=f"Forbidden import: '{module_name}'"
    ):
        guard.check(source)


@pytest.mark.parametrize(
    "module_name",
    [
        "os",
        "sys",
        "subprocess",
        "socket",
        "importlib",
        "pty",
        "shutil",
    ],
)
def test_forbidden_import_from(guard: ASTGuard, module_name: str) -> None:
    source = f"from {module_name} import path"
    with pytest.raises(
        SecurityViolationError, match=f"Forbidden import: '{module_name}'"
    ):
        guard.check(source)


def test_import_from_safe(guard: ASTGuard) -> None:
    source = "from math import sqrt"
    guard.check(source)


def test_import_from_no_module(guard: ASTGuard) -> None:
    # edge case: from . import foo (module is None)
    source = "from . import foo"
    # Should not raise SecurityViolationError
    try:
        guard.check(source)
    except SecurityViolationError:
        pytest.fail("Safe relative import raised SecurityViolationError")
    except SyntaxError:
        pass


@pytest.mark.parametrize(
    "call_expr",
    [
        "eval('1 + 1')",
        "exec('x = 1')",
        "compile('x = 1', '', 'exec')",
        "__import__('os')",
    ],
)
def test_forbidden_calls(guard: ASTGuard, call_expr: str) -> None:
    with pytest.raises(
        SecurityViolationError, match="Forbidden dynamic execution call"
    ):
        guard.check(call_expr)


@pytest.mark.parametrize(
    "attr_name",
    [
        "__subclasses__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__mro__",
    ],
)
def test_forbidden_attributes(guard: ASTGuard, attr_name: str) -> None:
    source = f"x = obj.{attr_name}"
    with pytest.raises(
        SecurityViolationError,
        match=f"Forbidden private attribute access: '{attr_name}'",
    ):
        guard.check(source)


def test_syntax_error(guard: ASTGuard) -> None:
    source = "def broken(:"
    with pytest.raises(SyntaxError):
        guard.check(source)


def test_resolve_func_name_edge_cases(guard: ASTGuard) -> None:
    # Test call via attribute or non-name function
    source = "obj.eval('1')"
    with pytest.raises(
        SecurityViolationError, match="Forbidden dynamic execution call"
    ):
        guard.check(source)

    # Call with non-name/non-attribute func, e.g. indexing
    source2 = "[print][0]('hello')"
    # Should not raise security violation (func_name is empty)
    guard.check(source2)
