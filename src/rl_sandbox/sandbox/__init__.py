"""Sandbox provider package: abstract interface and context helper."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from rl_sandbox.sandbox.base import SandboxProvider

__all__ = ["SandboxProvider", "sandbox_context"]


@asynccontextmanager
async def sandbox_context(
    provider: SandboxProvider,
) -> AsyncIterator[SandboxProvider]:
    """Context manager helper for providers that implement async context management."""
    async with provider:
        yield provider
