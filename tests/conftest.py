"""Shared pytest fixtures for rl-code-sandbox test suite."""

from __future__ import annotations

import pytest

from rl_sandbox.domain.models import CodeSubmission, TestHarness


@pytest.fixture()
def simple_submission() -> CodeSubmission:
    return CodeSubmission(
        code="def add(a, b):\n    return a + b\n",
        task_id="task-add",
        candidate_id="cand-001",
        timeout_seconds=10.0,
    )


@pytest.fixture()
def simple_harness() -> TestHarness:
    return TestHarness(assertions="assert add(1, 2) == 3\nassert add(0, 0) == 0\n")
