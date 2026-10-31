"""Deterministic evaluator and reward engine (D-003, D-019, D-020)."""

from __future__ import annotations

from typing import assert_never

from rl_sandbox.domain.errors import SecurityViolationError
from rl_sandbox.domain.models import (
    CodeSubmission,
    EvaluationResult,
    ExecutionResult,
    ExecutionStats,
    StatusLiteral,
    TestHarness,
)
from rl_sandbox.sandbox.base import SandboxProvider
from rl_sandbox.security.ast_guard import ASTGuard


class RewardEngine:
    """Coordinates AST inspection, sandbox execution,

    and deterministic reward mapping.
    """

    def __init__(self, provider: SandboxProvider) -> None:
        self._provider = provider
        self._ast_guard = ASTGuard()

    async def evaluate(
        self, submission: CodeSubmission, harness: TestHarness
    ) -> EvaluationResult:
        """Evaluate submission against test harness using sandbox provider."""
        # 1. Static AST Security Inspection (defense-in-depth)
        try:
            self._ast_guard.check(submission.code)
        except SecurityViolationError as e:
            stats = ExecutionStats(
                provisioning_time_ms=0.0,
                execution_time_ms=0.0,
                exit_code=-1,
                peak_memory_bytes=None,
                memory_best_effort=True,
            )
            exec_result = ExecutionResult(
                status="VIOLATION",
                stdout="",
                stderr=f"AST Security Violation: {e}",
                stats=stats,
            )
            return EvaluationResult(
                submission=submission,
                execution=exec_result,
                reward=-1.0,
                passed=False,
            )

        # 2. Sandbox Execution Dispatch
        exec_result = await self._provider.run(submission, harness)

        # 3. Deterministic Reward Mapping
        reward, passed = self._map_status_to_reward(exec_result.status)

        return EvaluationResult(
            submission=submission,
            execution=exec_result,
            reward=reward,
            passed=passed,
        )

    @staticmethod
    def _map_status_to_reward(status: StatusLiteral) -> tuple[float, bool]:
        """Map execution status to scalar reward and boolean pass indicator."""
        if status == "PASSED":
            return 1.0, True
        if status == "FAILED":
            return 0.0, False
        if status == "TIMEOUT":
            return 0.0, False
        if status == "OOM":
            return 0.0, False
        if status == "VIOLATION":
            return -1.0, False
        assert_never(status)
