"""Domain exceptions and reward value constants for rl-code-sandbox."""

from __future__ import annotations

from typing import Final

# Reward value constants — the only three valid scalar rewards (D-003, D-019).
REWARD_PASSED: Final[float] = 1.0
REWARD_FAILED: Final[float] = 0.0
REWARD_VIOLATION: Final[float] = -1.0

VALID_REWARDS: Final[frozenset[float]] = frozenset(
    {REWARD_PASSED, REWARD_FAILED, REWARD_VIOLATION}
)


class RLSandboxError(Exception):
    """Base exception for all rl-code-sandbox errors."""


class ConfigurationError(RLSandboxError):
    """Raised when the sandbox or evaluator is misconfigured."""


class SecurityViolationError(RLSandboxError):
    """Raised when a submission is rejected by the AST security guard."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Security violation: {reason}")


class ExecutionError(RLSandboxError):
    """Raised when the sandbox encounters an unrecoverable execution error."""

    def __init__(self, reason: str, exit_code: int | None = None) -> None:
        self.reason = reason
        self.exit_code = exit_code
        super().__init__(f"Execution error (exit={exit_code}): {reason}")


class ProviderUnavailableError(RLSandboxError):
    """Raised when the sandbox provider (e.g. Docker daemon) is not reachable."""

    def __init__(self, provider: str, detail: str) -> None:
        self.provider = provider
        self.detail = detail
        super().__init__(f"Provider '{provider}' unavailable: {detail}")
