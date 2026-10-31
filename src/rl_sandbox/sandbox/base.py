"""Abstract sandbox provider interface (D-001, D-004, D-012)."""

from __future__ import annotations

import types
from abc import ABC, abstractmethod

from rl_sandbox.domain.models import (
    CodeSubmission,
    ExecutionResult,
    TestHarness,
)


class SandboxProvider(ABC):
    """Abstract interface for executing code submissions in isolated sandboxes."""

    @abstractmethod
    async def run(
        self, submission: CodeSubmission, harness: TestHarness
    ) -> ExecutionResult:
        """Execute a code submission with a test harness and return the raw result."""
        ...

    @abstractmethod
    async def __aenter__(self) -> "SandboxProvider":
        """Enter the async context manager."""
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the async context manager and release resources."""
        ...
