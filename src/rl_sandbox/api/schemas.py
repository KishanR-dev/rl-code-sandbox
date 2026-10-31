"""Pydantic v2 request/response schemas for the FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rl_sandbox.domain.models import EvaluationResult


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(..., description="Service status string.")
    version: str = Field(..., description="Package version.")
    docker_available: bool = Field(
        ..., description="True when Docker daemon is reachable."
    )


class EvaluateResponse(BaseModel):
    """Response schema for POST /evaluate."""

    result: EvaluationResult = Field(..., description="Full evaluation result.")
