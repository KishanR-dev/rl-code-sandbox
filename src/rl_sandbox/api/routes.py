"""FastAPI route definitions for rl-sandbox service."""

from __future__ import annotations

import contextlib

import docker
import docker.errors
from fastapi import APIRouter, Depends, HTTPException

from rl_sandbox.api.schemas import EvaluateResponse, HealthResponse
from rl_sandbox.domain.models import EvaluationRequest
from rl_sandbox.evaluator.reward import RewardEngine
from rl_sandbox.sandbox.docker_provider import DockerSandboxProvider
from rl_sandbox.sandbox.mock_provider import MockSandboxProvider

router = APIRouter()

_VERSION = "0.1.0"


def _docker_available() -> bool:
    with contextlib.suppress(Exception):
        client = docker.from_env()
        client.ping()
        return True
    return False


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service liveness and Docker connectivity status."""
    return HealthResponse(
        status="ok",
        version=_VERSION,
        docker_available=_docker_available(),
    )


async def _get_engine(request: EvaluationRequest) -> RewardEngine:
    if request.use_mock:
        return RewardEngine(MockSandboxProvider(outcome="pass"))
    try:
        provider = DockerSandboxProvider()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Docker provider unavailable: {exc}",
        ) from exc
    return RewardEngine(provider)


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    request: EvaluationRequest,
    engine: RewardEngine = Depends(_get_engine),
) -> EvaluateResponse:
    """Evaluate a code submission against a test harness."""
    result = await engine.evaluate(request.submission, request.harness)
    return EvaluateResponse(result=result)
