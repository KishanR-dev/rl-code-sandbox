"""Unit tests for RewardEngine and deterministic reward mapping

(D-003, D-019, D-020).
"""

from __future__ import annotations

import pytest

from rl_sandbox.domain.models import CodeSubmission, TestHarness
from rl_sandbox.evaluator.reward import RewardEngine
from rl_sandbox.sandbox.mock_provider import MockSandboxProvider


@pytest.fixture
def submission() -> CodeSubmission:
    return CodeSubmission(
        code="x = 1 + 1",
        task_id="task-eval-1",
        candidate_id="cand-1",
    )


@pytest.fixture
def harness() -> TestHarness:
    return TestHarness(assertions="assert x == 2")


@pytest.mark.asyncio
async def test_reward_engine_passed(
    submission: CodeSubmission, harness: TestHarness
) -> None:
    provider = MockSandboxProvider(outcome="pass", exit_code=0)
    engine = RewardEngine(provider)
    result = await engine.evaluate(submission, harness)
    assert result.reward == 1.0
    assert result.passed is True
    assert result.execution.status == "PASSED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,expected_status,expected_reward",
    [
        ("fail", "FAILED", 0.0),
        ("runtime_error", "FAILED", 0.0),
        ("timeout", "TIMEOUT", 0.0),
        ("oom", "OOM", 0.0),
        ("pid_exhaustion", "VIOLATION", -1.0),
    ],
)
async def test_reward_engine_outcomes(
    submission: CodeSubmission,
    harness: TestHarness,
    outcome: str,
    expected_status: str,
    expected_reward: float,
) -> None:
    provider = MockSandboxProvider(
        outcome=outcome, exit_code=1 if outcome != "timeout" else 124
    )
    engine = RewardEngine(provider)
    result = await engine.evaluate(submission, harness)
    assert result.reward == expected_reward
    assert result.passed is (expected_reward == 1.0)
    assert result.execution.status == expected_status


@pytest.mark.asyncio
async def test_reward_engine_ast_violation(harness: TestHarness) -> None:
    bad_submission = CodeSubmission(
        code="import os; os.system('echo hacked')",
        task_id="task-ast-1",
        candidate_id="cand-1",
    )
    provider = MockSandboxProvider(outcome="pass")
    engine = RewardEngine(provider)
    result = await engine.evaluate(bad_submission, harness)
    assert result.reward == -1.0
    assert result.passed is False
    assert result.execution.status == "VIOLATION"
    assert "AST Security Violation" in result.execution.stderr
