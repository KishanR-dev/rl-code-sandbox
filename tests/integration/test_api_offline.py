"""Offline integration tests for the FastAPI service using MockSandboxProvider."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rl_sandbox.api.app import create_app
from rl_sandbox.domain.models import (
    CodeSubmission,
    EvaluationResult,
    ExecutionResult,
    ExecutionStats,
    TestHarness,
)
from rl_sandbox.evaluator.reward import RewardEngine
from rl_sandbox.sandbox.mock_provider import MockSandboxProvider


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def _make_eval_result(status: str = "PASSED", reward: float = 1.0) -> EvaluationResult:
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
    submission = CodeSubmission(code="x = 1", task_id="t1", candidate_id="c1")
    return EvaluationResult(
        submission=submission,
        execution=exec_result,
        reward=reward,
        passed=(reward == 1.0),
    )


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert isinstance(body["docker_available"], bool)


def test_evaluate_mock_passed(client: TestClient) -> None:
    payload: dict[str, Any] = {
        "submission": {
            "code": "x = 1 + 1",
            "task_id": "task-1",
            "candidate_id": "cand-1",
        },
        "harness": {"assertions": "assert x == 2"},
        "use_mock": True,
    }
    mock_result = _make_eval_result("PASSED", 1.0)
    with patch.object(
        RewardEngine, "evaluate", new=AsyncMock(return_value=mock_result)
    ):
        response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["reward"] == 1.0
    assert body["result"]["passed"] is True
    assert body["result"]["execution"]["status"] == "PASSED"


def test_evaluate_mock_failed(client: TestClient) -> None:
    payload: dict[str, Any] = {
        "submission": {
            "code": "x = 0",
            "task_id": "task-2",
            "candidate_id": "cand-2",
        },
        "harness": {"assertions": "assert x == 2"},
        "use_mock": True,
    }
    mock_result = _make_eval_result("FAILED", 0.0)
    with patch.object(
        RewardEngine, "evaluate", new=AsyncMock(return_value=mock_result)
    ):
        response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["reward"] == 0.0
    assert body["result"]["passed"] is False
    assert body["result"]["execution"]["status"] == "FAILED"


def test_evaluate_mock_violation(client: TestClient) -> None:
    payload: dict[str, Any] = {
        "submission": {
            "code": "import os",
            "task_id": "task-3",
            "candidate_id": "cand-3",
        },
        "harness": {"assertions": "assert True"},
        "use_mock": True,
    }
    mock_result = _make_eval_result("VIOLATION", -1.0)
    with patch.object(
        RewardEngine, "evaluate", new=AsyncMock(return_value=mock_result)
    ):
        response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["reward"] == -1.0
    assert body["result"]["passed"] is False
    assert body["result"]["execution"]["status"] == "VIOLATION"


def test_evaluate_docker_unavailable(client: TestClient) -> None:
    """When use_mock=False and Docker is unavailable, expect 503."""
    payload: dict[str, Any] = {
        "submission": {
            "code": "x = 1",
            "task_id": "task-4",
            "candidate_id": "cand-4",
        },
        "harness": {"assertions": "assert x == 1"},
        "use_mock": False,
    }
    with patch(
        "rl_sandbox.api.routes.DockerSandboxProvider",
        side_effect=RuntimeError("Docker not available"),
    ):
        response = client.post("/evaluate", json=payload)
    assert response.status_code == 503
    assert "Docker provider unavailable" in response.json()["detail"]


def test_evaluate_real_mock_provider() -> None:
    """End-to-end offline test using real MockSandboxProvider (no patching)."""
    import asyncio

    provider = MockSandboxProvider(outcome="pass", exit_code=0)
    engine = RewardEngine(provider)
    submission = CodeSubmission(code="x = 1", task_id="t1", candidate_id="c1")
    harness = TestHarness(assertions="assert x == 1")
    result = asyncio.run(engine.evaluate(submission, harness))
    assert result.reward == 1.0
    assert result.passed is True


def test_health_docker_available(client: TestClient) -> None:
    """Health endpoint reports docker_available=True when ping succeeds."""
    mock_client = MagicMock()
    with patch("rl_sandbox.api.routes.docker.from_env", return_value=mock_client):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["docker_available"] is True


def test_health_docker_unavailable(client: TestClient) -> None:
    """Health endpoint reports docker_available=False when ping raises."""
    with patch(
        "rl_sandbox.api.routes.docker.from_env",
        side_effect=Exception("no docker"),
    ):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["docker_available"] is False
