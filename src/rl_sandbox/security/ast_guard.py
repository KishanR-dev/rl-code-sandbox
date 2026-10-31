"""Static AST security guard for pre-execution candidate code inspection.

This module is defense-in-depth only (D-001). The Docker container is the
authoritative security perimeter; this guard fast-fails known prohibited
constructs before dispatching to the sandbox provider.
"""

from __future__ import annotations

import ast
from typing import Final

from rl_sandbox.domain.errors import SecurityViolationError

# Forbidden top-level module imports (D-001, PLAN.md §1).
FORBIDDEN_MODULES: Final[frozenset[str]] = frozenset(
    {"os", "sys", "subprocess", "socket", "importlib", "pty", "shutil"}
)

# Forbidden dynamic execution primitives.
FORBIDDEN_CALLS: Final[frozenset[str]] = frozenset(
    {"eval", "exec", "compile", "__import__"}
)

# Private attribute traversal attempts that can escape sandboxing.
FORBIDDEN_ATTRS: Final[frozenset[str]] = frozenset(
    {"__subclasses__", "__globals__", "__code__", "__builtins__", "__mro__"}
)


class ASTGuard:
    """Static AST visitor that inspects candidate code strings

    for security violations.
    """

    def __init__(self) -> None:
        self._forbidden_modules = FORBIDDEN_MODULES
        self._forbidden_calls = FORBIDDEN_CALLS
        self._forbidden_attrs = FORBIDDEN_ATTRS

    def check(self, source: str) -> None:
        """Inspect `source` and raise `SecurityViolationError` on any violation.

        Args:
            source: Python source code string to inspect.

        Raises:
            SecurityViolationError: If the code contains forbidden imports,
                dynamic execution calls, or private attribute tampering.
            SyntaxError: If the source cannot be parsed as valid Python.
        """
        tree = ast.parse(source)
        visitor = _SecurityVisitor(
            self._forbidden_modules, self._forbidden_calls, self._forbidden_attrs
        )
        visitor.visit(tree)
        if visitor.violations:
            raise SecurityViolationError("; ".join(visitor.violations))


class _SecurityVisitor(ast.NodeVisitor):
    """AST visitor that collects security violations."""

    def __init__(
        self,
        forbidden_modules: frozenset[str],
        forbidden_calls: frozenset[str],
        forbidden_attrs: frozenset[str],
    ) -> None:
        self._forbidden_modules = forbidden_modules
        self._forbidden_calls = forbidden_calls
        self._forbidden_attrs = forbidden_attrs
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in self._forbidden_modules:
                self.violations.append(f"Forbidden import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self.generic_visit(node)
            return
        top = node.module.split(".")[0]
        if top in self._forbidden_modules:
            self.violations.append(f"Forbidden import: '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._resolve_func_name(node.func)
        if func_name in self._forbidden_calls:
            self.violations.append(f"Forbidden dynamic execution call: '{func_name}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self._forbidden_attrs:
            self.violations.append(f"Forbidden private attribute access: '{node.attr}'")
        self.generic_visit(node)

    def _resolve_func_name(self, func: ast.expr) -> str:
        """Resolve the simple name of a call target (handles dotted chains)."""
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""
