"""Unit tests for sandbox provider interface and context helper."""

from __future__ import annotations

import pytest

from rl_sandbox.domain.models import CodeSubmission, TestHarness
from rl_sandbox.sandbox import SandboxProvider, sandbox_context
from rl_sandbox.sandbox.mock_provider import MockSandboxProvider


@pytest.fixture
def submission() -> CodeSubmission:
    return CodeSubmission(
        code="print('hello')",
        task_id="task-1",
        candidate_id="cand-1",
    )


@pytest.fixture
def harness() -> TestHarness:
    return TestHarness(assertions="assert True")


async def test_sandbox_context_helper(
    submission: CodeSubmission, harness: TestHarness
) -> None:
    provider = MockSandboxProvider()
    async with sandbox_context(provider) as p:
        assert p is provider
        result = await p.run(submission, harness)
        assert result.status == "PASSED"


def test_sandbox_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        SandboxProvider()  # type: ignore[abstract]
