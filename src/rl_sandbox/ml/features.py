"""AST-based structural feature extractor for candidate code submissions."""

from __future__ import annotations

import ast
import math


def _ast_depth(node: ast.AST) -> int:
    """Return the maximum depth of an AST tree rooted at *node*."""
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(child) for child in children)


def _cyclomatic_complexity(tree: ast.AST) -> int:
    """Approximate McCabe cyclomatic complexity (decision points + 1)."""
    decision_nodes = (
        ast.If,
        ast.For,
        ast.While,
        ast.ExceptHandler,
        ast.With,
        ast.Assert,
        ast.comprehension,
        ast.BoolOp,
    )
    count = sum(1 for node in ast.walk(tree) if isinstance(node, decision_nodes))
    return count + 1


def _branch_count(tree: ast.AST) -> int:
    """Count explicit branching nodes (if/elif/else, for, while, try)."""
    branch_nodes = (ast.If, ast.For, ast.While, ast.Try)
    return sum(1 for node in ast.walk(tree) if isinstance(node, branch_nodes))


def _exception_handler_count(tree: ast.AST) -> int:
    """Count except-handler clauses."""
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler))


def extract_features(code: str) -> list[float]:
    """Extract structural features from *code*.

    Returns a fixed-length feature vector:
      [ast_depth, node_count, cyclomatic_complexity, branch_count,
       exception_density, token_length_log1p]

    Returns a zero vector when *code* cannot be parsed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    node_count = sum(1 for _ in ast.walk(tree))
    depth = _ast_depth(tree)
    cc = _cyclomatic_complexity(tree)
    branches = _branch_count(tree)
    handlers = _exception_handler_count(tree)
    exception_density = handlers / node_count if node_count > 0 else 0.0
    token_length = math.log1p(len(code))

    return [
        float(depth),
        float(node_count),
        float(cc),
        float(branches),
        exception_density,
        token_length,
    ]


FEATURE_NAMES: list[str] = [
    "ast_depth",
    "node_count",
    "cyclomatic_complexity",
    "branch_count",
    "exception_density",
    "token_length_log1p",
]

FEATURE_DIM: int = len(FEATURE_NAMES)
