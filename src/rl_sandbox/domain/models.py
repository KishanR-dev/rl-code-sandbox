"""Domain models for rl-code-sandbox."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator


class CodeSubmission(BaseModel):
    """A candidate code submission to be executed in the sandbox."""

    code: str = Field(
        ...,
        max_length=40_000,
        description="Python source code to execute.",
    )
    task_id: str = Field(..., description="Unique identifier for the task.")
    candidate_id: str = Field(..., description="Unique identifier for this candidate.")
    timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=300.0,
        description="Wall-clock execution timeout in seconds.",
    )


class TestHarness(BaseModel):
    """Test assertions to run against the submitted code."""

    __test__ = False  # prevent pytest from collecting this model as a test class

    assertions: str = Field(
        ...,
        max_length=40_000,
        description=(
            "Python source containing assert statements to validate the submission."
        ),
    )


class ExecutionStats(BaseModel):
    """Decomposed latency and resource metrics for a single execution."""

    provisioning_time_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Time to provision the container or mock environment (milliseconds)."
        ),
    )
    execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Time for user code and harness to run inside the container (milliseconds)."
        ),
    )
    exit_code: int = Field(..., description="Process exit code from the container.")
    peak_memory_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Peak RSS/cgroup memory in bytes; None when unavailable.",
    )
    memory_best_effort: bool = Field(
        default=False,
        description=(
            "True when peak_memory_bytes is a best-effort estimate,"
            " not cgroup v2 exact."
        ),
    )


StatusLiteral: TypeAlias = Literal["PASSED", "FAILED", "TIMEOUT", "OOM", "VIOLATION"]


class ExecutionResult(BaseModel):
    """Raw output from a sandbox execution, before reward mapping."""

    status: StatusLiteral = Field(..., description="Execution outcome classification.")
    stdout: str = Field(default="", description="Captured standard output.")
    stderr: str = Field(default="", description="Captured standard error.")
    stats: ExecutionStats = Field(..., description="Latency and resource metrics.")


# Reward is one of exactly three float values: 1.0, 0.0, -1.0.
# mypy does not support float literals in Literal[], so we use Annotated[float, ...]
# and enforce the constraint with a field validator.
RewardValue: TypeAlias = Annotated[float, Field(ge=-1.0, le=1.0)]

_VALID_REWARDS: frozenset[float] = frozenset({1.0, 0.0, -1.0})


class EvaluationResult(BaseModel):
    """Final evaluation output including deterministic reward."""

    submission: CodeSubmission = Field(..., description="The evaluated submission.")
    execution: ExecutionResult = Field(..., description="Raw execution result.")
    reward: RewardValue = Field(
        ...,
        description=(
            "Deterministic scalar reward: 1.0 passed, 0.0 failed/timeout/OOM,"
            " -1.0 violation."
        ),
    )
    passed: bool = Field(..., description="True when reward == 1.0.")

    @field_validator("reward")
    @classmethod
    def reward_must_be_valid(cls, v: float) -> float:
        if v not in _VALID_REWARDS:
            raise ValueError(f"reward must be one of {sorted(_VALID_REWARDS)}, got {v}")
        return v

    @model_validator(mode="after")
    def validate_passed_reward_consistency(self) -> EvaluationResult:
        if self.passed != (self.reward == 1.0):
            raise ValueError(
                f"passed={self.passed} is inconsistent with reward={self.reward}"
            )
        return self


class EvaluationRequest(BaseModel):
    """Unified API request schema for a single evaluation (D-017)."""

    submission: CodeSubmission = Field(..., description="Code submission to evaluate.")
    harness: TestHarness = Field(..., description="Test assertions to run.")
    use_mock: bool = Field(
        default=False,
        description=(
            "When True, use MockSandboxProvider instead of DockerSandboxProvider."
        ),
    )
