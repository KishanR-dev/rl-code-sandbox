"""FastAPI application factory for rl-sandbox."""

from __future__ import annotations

from fastapi import FastAPI

from rl_sandbox.api.routes import router

_VERSION = "0.1.0"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="rl-code-sandbox",
        version=_VERSION,
        description=(
            "Secure code execution sandbox with deterministic RL reward evaluation."
        ),
    )
    app.include_router(router)
    return app


app = create_app()
