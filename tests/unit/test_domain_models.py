"""Unit tests for domain models (CHECKLIST 1.4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rl_sandbox.domain.errors import (
    REWARD_FAILED,
    REWARD_PASSED,
    REWARD_VIOLATION,
    VALID_REWARDS,
    ConfigurationError,
    ExecutionError,
    ProviderUnavailableError,
    RLSandboxError,
    SecurityViolationError,
)
from rl_sandbox.domain.models import (
    CodeSubmission,
    EvaluationRequest,
    EvaluationResult,
    ExecutionResult,
    ExecutionStats,
    TestHarness,
)

# ---------------------------------------------------------------------------
# CodeSubmission
# ---------------------------------------------------------------------------


class TestCodeSubmission:
    def test_valid_submission(self) -> None:
        s = CodeSubmission(
            code="x = 1",
            task_id="t1",
            candidate_id="c1",
            timeout_seconds=5.0,
        )
        assert s.code == "x = 1"
        assert s.timeout_seconds == 5.0

    def test_default_timeout(self) -> None:
        s = CodeSubmission(code="pass", task_id="t1", candidate_id="c1")
        assert s.timeout_seconds == 10.0

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            CodeSubmission(
                code="pass", task_id="t1", candidate_id="c1", timeout_seconds=0.0
            )

    def test_timeout_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            CodeSubmission(
                code="pass", task_id="t1", candidate_id="c1", timeout_seconds=301.0
            )

    def test_timeout_at_upper_bound_is_valid(self) -> None:
        s = CodeSubmission(
            code="pass", task_id="t1", candidate_id="c1", timeout_seconds=300.0
        )
        assert s.timeout_seconds == 300.0

    def test_code_exceeds_max_length(self) -> None:
        long_code = "x = 1\n" * 8000  # >40,000 chars
        with pytest.raises(ValidationError):
            CodeSubmission(code=long_code, task_id="t1", candidate_id="c1")


# ---------------------------------------------------------------------------
# TestHarness
# ---------------------------------------------------------------------------


class TestTestHarness:
    def test_valid_harness(self) -> None:
        h = TestHarness(assertions="assert 1 == 1")
        assert "assert" in h.assertions

    def test_empty_assertions_allowed(self) -> None:
        h = TestHarness(assertions="")
        assert h.assertions == ""

    def test_assertions_exceeds_max_length(self) -> None:
        long_assertions = "assert True\n" * 5000  # >40,000 chars
        with pytest.raises(ValidationError):
            TestHarness(assertions=long_assertions)


# ---------------------------------------------------------------------------
# ExecutionStats
# ---------------------------------------------------------------------------


class TestExecutionStats:
    def test_valid_stats(self) -> None:
        s = ExecutionStats(
            provisioning_time_ms=50.0,
            execution_time_ms=120.0,
            exit_code=0,
            peak_memory_bytes=1024 * 1024,
            memory_best_effort=False,
        )
        assert s.exit_code == 0
        assert s.peak_memory_bytes == 1024 * 1024

    def test_peak_memory_optional(self) -> None:
        s = ExecutionStats(
            provisioning_time_ms=10.0,
            execution_time_ms=20.0,
            exit_code=1,
        )
        assert s.peak_memory_bytes is None
        assert s.memory_best_effort is False

    def test_negative_provisioning_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionStats(
                provisioning_time_ms=-1.0,
                execution_time_ms=10.0,
                exit_code=0,
            )


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def _stats(self) -> ExecutionStats:
        return ExecutionStats(
            provisioning_time_ms=10.0,
            execution_time_ms=50.0,
            exit_code=0,
        )

    def test_passed_status(self) -> None:
        r = ExecutionResult(status="PASSED", stats=self._stats())
        assert r.status == "PASSED"
        assert r.stdout == ""
        assert r.stderr == ""

    def test_all_statuses_valid(self) -> None:
        for status in ("PASSED", "FAILED", "TIMEOUT", "OOM", "VIOLATION"):
            r = ExecutionResult(status=status, stats=self._stats())  # type: ignore[arg-type]
            assert r.status == status

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionResult(status="UNKNOWN", stats=self._stats())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def _submission(self) -> CodeSubmission:
        return CodeSubmission(code="pass", task_id="t1", candidate_id="c1")

    def _stats(self) -> ExecutionStats:
        return ExecutionStats(
            provisioning_time_ms=5.0,
            execution_time_ms=15.0,
            exit_code=0,
        )

    def _execution(self, status: str = "PASSED") -> ExecutionResult:
        return ExecutionResult(status=status, stats=self._stats())  # type: ignore[arg-type]

    def test_passed_result(self) -> None:
        r = EvaluationResult(
            submission=self._submission(),
            execution=self._execution("PASSED"),
            reward=1.0,
            passed=True,
        )
        assert r.reward == 1.0
        assert r.passed is True

    def test_failed_result(self) -> None:
        r = EvaluationResult(
            submission=self._submission(),
            execution=self._execution("FAILED"),
            reward=0.0,
            passed=False,
        )
        assert r.reward == 0.0
        assert r.passed is False

    def test_violation_result(self) -> None:
        r = EvaluationResult(
            submission=self._submission(),
            execution=self._execution("VIOLATION"),
            reward=-1.0,
            passed=False,
        )
        assert r.reward == -1.0
        assert r.passed is False

    def test_inconsistent_passed_reward_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(
                submission=self._submission(),
                execution=self._execution("PASSED"),
                reward=1.0,
                passed=False,  # inconsistent
            )

    def test_inconsistent_failed_reward_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(
                submission=self._submission(),
                execution=self._execution("FAILED"),
                reward=0.0,
                passed=True,  # inconsistent
            )

    def test_invalid_reward_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(
                submission=self._submission(),
                execution=self._execution("PASSED"),
                reward=0.5,  # type: ignore[arg-type]
                passed=True,
            )


# ---------------------------------------------------------------------------
# EvaluationRequest
# ---------------------------------------------------------------------------


class TestEvaluationRequest:
    def test_default_use_mock_false(self) -> None:
        req = EvaluationRequest(
            submission=CodeSubmission(code="pass", task_id="t1", candidate_id="c1"),
            harness=TestHarness(assertions="assert True"),
        )
        assert req.use_mock is False

    def test_use_mock_true(self) -> None:
        req = EvaluationRequest(
            submission=CodeSubmission(code="pass", task_id="t1", candidate_id="c1"),
            harness=TestHarness(assertions="assert True"),
            use_mock=True,
        )
        assert req.use_mock is True


# ---------------------------------------------------------------------------
# Domain errors (CHECKLIST 1.5)
# ---------------------------------------------------------------------------


class TestRewardConstants:
    def test_reward_values(self) -> None:
        assert REWARD_PASSED == 1.0
        assert REWARD_FAILED == 0.0
        assert REWARD_VIOLATION == -1.0

    def test_valid_rewards_set(self) -> None:
        assert VALID_REWARDS == frozenset({1.0, 0.0, -1.0})

    def test_no_other_values_in_valid_set(self) -> None:
        assert 0.5 not in VALID_REWARDS
        assert 2.0 not in VALID_REWARDS


class TestDomainExceptions:
    def test_rl_sandbox_error_is_base(self) -> None:
        err = RLSandboxError("base error")
        assert isinstance(err, Exception)

    def test_configuration_error(self) -> None:
        err = ConfigurationError("bad config")
        assert isinstance(err, RLSandboxError)
        assert "bad config" in str(err)

    def test_security_violation_error(self) -> None:
        err = SecurityViolationError("forbidden import: os")
        assert isinstance(err, RLSandboxError)
        assert err.reason == "forbidden import: os"
        assert "Security violation" in str(err)

    def test_execution_error_with_exit_code(self) -> None:
        err = ExecutionError("container crashed", exit_code=137)
        assert isinstance(err, RLSandboxError)
        assert err.exit_code == 137
        assert "137" in str(err)

    def test_execution_error_without_exit_code(self) -> None:
        err = ExecutionError("unknown failure")
        assert err.exit_code is None

    def test_provider_unavailable_error(self) -> None:
        err = ProviderUnavailableError("docker", "daemon not running")
        assert isinstance(err, RLSandboxError)
        assert err.provider == "docker"
        assert "daemon not running" in str(err)
