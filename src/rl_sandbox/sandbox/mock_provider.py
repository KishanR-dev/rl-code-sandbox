"""Mock sandbox provider for 100% offline testing (D-004)."""

from __future__ import annotations

import asyncio
import types
from typing import Optional

from rl_sandbox.domain.models import (
    CodeSubmission,
    ExecutionResult,
    ExecutionStats,
    StatusLiteral,
    TestHarness,
)
from rl_sandbox.sandbox import SandboxProvider


class MockSandboxProvider(SandboxProvider):
    """Simulates pass, assertion failure, runtime error, OOM, timeout, PID exhaust."""

    def __init__(
        self,
        provisioning_time_ms: float = 5.0,
        execution_time_ms: float = 10.0,
        outcome: str = "pass",
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
        peak_memory_bytes: Optional[int] = None,
        memory_best_effort: bool = True,
    ) -> None:
        self._provisioning_time_ms = provisioning_time_ms
        self._execution_time_ms = execution_time_ms
        self._outcome = outcome
        self._exit_code = exit_code
        self._stdout = stdout
        self._stderr = stderr
        self._peak_memory_bytes = peak_memory_bytes
        self._memory_best_effort = memory_best_effort

    async def run(
        self, submission: CodeSubmission, harness: TestHarness
    ) -> ExecutionResult:
        await asyncio.sleep(0)  # yield to event loop
        status_map: dict[str, StatusLiteral] = {
            "pass": "PASSED",
            "fail": "FAILED",
            "runtime_error": "FAILED",
            "oom": "OOM",
            "timeout": "TIMEOUT",
            "pid_exhaustion": "VIOLATION",
        }
        status: StatusLiteral = status_map.get(self._outcome, "FAILED")
        stats = ExecutionStats(
            provisioning_time_ms=self._provisioning_time_ms,
            execution_time_ms=self._execution_time_ms,
            exit_code=self._exit_code,
            peak_memory_bytes=self._peak_memory_bytes,
            memory_best_effort=self._memory_best_effort,
        )
        return ExecutionResult(
            status=status,
            stdout=self._stdout,
            stderr=self._stderr,
            stats=stats,
        )

    async def __aenter__(self) -> "MockSandboxProvider":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        return None
