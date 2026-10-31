"""Offline integration tests for the Typer CLI using MockSandboxProvider."""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest.mock import AsyncMock, patch

import typer
from typer.testing import CliRunner

from rl_sandbox.cli.main import app
from rl_sandbox.domain.models import (
    CodeSubmission,
    EvaluationResult,
    ExecutionResult,
    ExecutionStats,
)
from rl_sandbox.evaluator.reward import RewardEngine

runner = CliRunner()


def _make_eval_result(
    status: str = "PASSED",
    reward: float = 1.0,
    code: str = "x = 1",
) -> EvaluationResult:
    stats = ExecutionStats(
        provisioning_time_ms=1.0,
        execution_time_ms=2.0,
        exit_code=0,
        peak_memory_bytes=None,
        memory_best_effort=True,
    )
    exec_result = ExecutionResult(
        status=status,  # type: ignore[arg-type]
        stdout="",
        stderr="",
        stats=stats,
    )
    submission = CodeSubmission(code=code, task_id="t1", candidate_id="c1")
    return EvaluationResult(
        submission=submission,
        execution=exec_result,
        reward=reward,
        passed=(reward == 1.0),
    )


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_run_command_passed() -> None:
    mock_result = _make_eval_result("PASSED", 1.0)
    with patch.object(
        RewardEngine, "evaluate", new=AsyncMock(return_value=mock_result)
    ):
        result = runner.invoke(
            app,
            ["run", "--code", "x = 1", "--test", "assert x == 1", "--mock"],
        )
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["status"] == "PASSED"
    assert output["reward"] == 1.0
    assert output["passed"] is True


def test_run_command_failed_exits_1() -> None:
    mock_result = _make_eval_result("FAILED", 0.0)
    with patch.object(
        RewardEngine, "evaluate", new=AsyncMock(return_value=mock_result)
    ):
        result = runner.invoke(
            app,
            ["run", "--code", "x = 0", "--test", "assert x == 1", "--mock"],
        )
    assert result.exit_code == 1
    output = json.loads(result.output)
    assert output["status"] == "FAILED"
    assert output["reward"] == 0.0


def test_run_command_violation_exits_1() -> None:
    mock_result = _make_eval_result("VIOLATION", -1.0)
    with patch.object(
        RewardEngine, "evaluate", new=AsyncMock(return_value=mock_result)
    ):
        result = runner.invoke(
            app,
            ["run", "--code", "import os", "--test", "assert True", "--mock"],
        )
    assert result.exit_code == 1
    output = json.loads(result.output)
    assert output["reward"] == -1.0


def test_run_command_docker_unavailable() -> None:
    with patch(
        "rl_sandbox.cli.main._build_provider",
        side_effect=typer.Exit(code=1),
    ):
        result = runner.invoke(
            app,
            ["run", "--code", "x = 1", "--test", "assert x == 1"],
        )
    assert result.exit_code == 1


def test_benchmark_command_missing_dir() -> None:
    result = runner.invoke(app, ["benchmark", "--tasks-dir", "/nonexistent/path"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_benchmark_command_empty_dir(tmp_path: pathlib.Path) -> None:
    result = runner.invoke(app, ["benchmark", "--tasks-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No JSONL fixtures" in result.output


def test_benchmark_command_mock(tmp_path: pathlib.Path) -> None:
    fixture: dict[str, Any] = {
        "submission": {
            "code": "x = 1 + 1",
            "task_id": "bench-task-1",
            "candidate_id": "cand-1",
        },
        "harness": {"assertions": "assert x == 2"},
    }
    fixture_file = tmp_path / "tasks.jsonl"
    fixture_file.write_text(json.dumps(fixture) + "\n", encoding="utf-8")

    output_file = tmp_path / "results.jsonl"
    mock_result = _make_eval_result("PASSED", 1.0)
    with patch.object(
        RewardEngine, "evaluate", new=AsyncMock(return_value=mock_result)
    ):
        result = runner.invoke(
            app,
            [
                "benchmark",
                "--tasks-dir",
                str(tmp_path),
                "--output",
                str(output_file),
                "--mock",
            ],
        )
    assert result.exit_code == 0
    assert "Evaluated 1 candidates" in result.output
    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task_id"] == "bench-task-1"
    assert record["reward"] == 1.0


def test_build_provider_docker_error_exits() -> None:
    """_build_provider raises typer.Exit when DockerSandboxProvider fails."""
    with patch(
        "rl_sandbox.sandbox.docker_provider.DockerSandboxProvider",
        side_effect=RuntimeError("no docker"),
    ):
        result = runner.invoke(
            app,
            ["run", "--code", "x = 1", "--test", "assert x == 1"],
        )
    assert result.exit_code == 1
