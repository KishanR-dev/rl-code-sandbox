"""Unit tests for MockSandboxProvider (D-004, D-013)."""

from __future__ import annotations

import pytest

from rl_sandbox.domain.models import CodeSubmission, TestHarness
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


@pytest.mark.parametrize(
    "outcome,expected_status,expected_exit_code",
    [
        ("pass", "PASSED", 0),
        ("fail", "FAILED", 1),
        ("runtime_error", "FAILED", 1),
        ("oom", "OOM", 137),
        ("timeout", "TIMEOUT", 124),
        ("pid_exhaustion", "VIOLATION", 1),
    ],
)
async def test_outcome_mapping(
    submission: CodeSubmission,
    harness: TestHarness,
    outcome: str,
    expected_status: str,
    expected_exit_code: int,
) -> None:
    provider = MockSandboxProvider(outcome=outcome, exit_code=expected_exit_code)
    result = await provider.run(submission, harness)
    assert result.status == expected_status
    assert result.stats.exit_code == expected_exit_code
    assert result.stats.provisioning_time_ms >= 0
    assert result.stats.execution_time_ms >= 0


async def test_context_manager() -> None:
    provider = MockSandboxProvider()
    async with provider as p:
        assert p is provider


async def test_custom_metrics(submission: CodeSubmission, harness: TestHarness) -> None:
    provider = MockSandboxProvider(
        provisioning_time_ms=42.5,
        execution_time_ms=99.0,
        peak_memory_bytes=1024,
        memory_best_effort=False,
        stdout="output",
        stderr="error",
    )
    result = await provider.run(submission, harness)
    assert result.stats.provisioning_time_ms == 42.5
    assert result.stats.execution_time_ms == 99.0
    assert result.stats.peak_memory_bytes == 1024
    assert result.stats.memory_best_effort is False
    assert result.stdout == "output"
    assert result.stderr == "error"
