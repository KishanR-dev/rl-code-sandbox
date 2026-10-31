"""Hardened Docker sandbox provider (D-001, D-002, D-012, D-014, D-023)."""

from __future__ import annotations

import asyncio
import base64
import time
import types
from typing import Any, Optional

import docker
from docker.models.containers import Container

from rl_sandbox.domain.models import (
    CodeSubmission,
    ExecutionResult,
    ExecutionStats,
    StatusLiteral,
    TestHarness,
)
from rl_sandbox.sandbox.base import SandboxProvider

# Pinned digest resolved 2026-10-31 from docker.io/library/python:3.11-slim (amd64).
# Update by running: docker pull python:3.11-slim && docker inspect python:3.11-slim
BASE_IMAGE = (
    "python:3.11-slim"
    "@sha256:6d10e720318bd84cc3d88af92bb5a47189222e17620a2fd214da6445e4afb679"
)

# Exit code returned by asyncio.wait_for when the wall-clock timer fires.
_TIMEOUT_EXIT_CODE = 124

# errno 11 (EAGAIN/EWOULDBLOCK) is raised by the kernel when pids_limit is hit.
_PID_EXHAUSTION_ERRNO = 11

# errno 101 (ENETUNREACH) is raised when network_mode="none" blocks a connection.
_NETWORK_UNREACHABLE_ERRNO = 101


class DockerSandboxProvider(SandboxProvider):
    """Executes code inside a hardened OCI container with no network, no privileges."""

    def __init__(
        self,
        image: str = BASE_IMAGE,
        mem_limit: str = "256m",
        memswap_limit: str = "256m",
        nano_cpus: int = 1_000_000_000,
        pids_limit: int = 64,
    ) -> None:
        self._image = image
        self._mem_limit = mem_limit
        self._memswap_limit = memswap_limit
        self._nano_cpus = nano_cpus
        self._pids_limit = pids_limit
        self._client: Optional[docker.DockerClient] = None
        self._container: Optional[Container] = None

    async def run(
        self, submission: CodeSubmission, harness: TestHarness
    ) -> ExecutionResult:
        provision_start = time.perf_counter()
        await self._provision_container()
        provision_time_ms = (time.perf_counter() - provision_start) * 1000.0

        exec_start = time.perf_counter()
        exit_code, stdout, stderr = await self._execute(submission, harness)
        exec_time_ms = (time.perf_counter() - exec_start) * 1000.0

        peak_memory, best_effort = await self._collect_memory_metrics()
        oom_killed = await self._inspect_oom()

        await self._terminate_container()

        status = self._classify_status(exit_code, stdout, stderr, oom_killed)

        stats = ExecutionStats(
            provisioning_time_ms=provision_time_ms,
            execution_time_ms=exec_time_ms,
            exit_code=exit_code,
            peak_memory_bytes=peak_memory,
            memory_best_effort=best_effort,
        )

        return ExecutionResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            stats=stats,
        )

    async def _provision_container(self) -> None:
        loop = asyncio.get_event_loop()
        self._client = await loop.run_in_executor(None, docker.from_env)
        self._container = await loop.run_in_executor(
            None,
            self._create_container,
        )
        await loop.run_in_executor(None, self._container.start)

    def _create_container(self) -> Container:
        if self._client is None:
            raise RuntimeError("Docker client not initialized")
        return self._client.containers.create(
            self._image,
            command=["sleep", "infinity"],
            user="1000:1000",
            tty=False,
            detach=True,
            network_mode="none",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            read_only=True,
            pids_limit=self._pids_limit,
            mem_limit=self._mem_limit,
            memswap_limit=self._memswap_limit,
            nano_cpus=self._nano_cpus,
            tmpfs={"/tmp": "rw,noexec,nosuid,nodev,size=32m"},
        )

    async def _execute(
        self, submission: CodeSubmission, harness: TestHarness
    ) -> tuple[int, str, str]:
        full_code = submission.code + "\n" + harness.assertions
        loop = asyncio.get_event_loop()
        try:
            result: tuple[int, str, str] = await asyncio.wait_for(
                loop.run_in_executor(None, self._run_sync, full_code),
                timeout=submission.timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            return _TIMEOUT_EXIT_CODE, "", "TIMEOUT: wall-clock expiration"

    def _run_sync(self, code: str) -> tuple[int, str, str]:
        if self._container is None or self._client is None:
            return 1, "", "Container not provisioned"

        encoded_bytes = code.encode("utf-8")
        if len(encoded_bytes) > 80_000:
            return 1, "", "ERROR: Payload exceeds 80KB limit"

        encoded_code = base64.b64encode(encoded_bytes).decode("ascii")
        exec_resp: dict[str, Any] = self._client.api.exec_create(
            self._container.id,
            [
                "python",
                "-c",
                (
                    "import os, base64; "
                    "exec(base64.b64decode(os.environ['B64_CODE']).decode('utf-8'))"
                ),
            ],
            environment={"B64_CODE": encoded_code},
            user="1000:1000",
        )
        exec_id: str = exec_resp["Id"]

        demux_result = self._client.api.exec_start(
            exec_id,
            detach=False,
            tty=False,
            stream=False,
            demux=True,
        )
        stdout_bytes, stderr_bytes = (
            demux_result if isinstance(demux_result, tuple) else (demux_result, None)
        )
        stdout_text: str = (
            stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        )
        stderr_text: str = (
            stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        )
        exit_info = self._client.api.exec_inspect(exec_id)
        raw_code = exit_info["ExitCode"]
        exit_code: int = int(raw_code) if raw_code is not None else 1
        return exit_code, stdout_text, stderr_text

    async def _collect_memory_metrics(self) -> tuple[Optional[int], bool]:
        if self._container is None or self._client is None:
            return None, True
        loop = asyncio.get_event_loop()
        try:
            raw_stats: dict[str, Any] = await loop.run_in_executor(
                None,
                lambda: self._container.stats(stream=False),  # type: ignore[arg-type, return-value, union-attr]
            )
            mem_stats: dict[str, Any] = raw_stats.get("memory_stats", {})
            # cgroup v2 exposes memory.peak via max_usage; cgroup v1 uses max_usage too.
            peak = mem_stats.get("max_usage", 0)
            if isinstance(peak, int) and peak > 0:
                return peak, False
            return None, True
        except Exception:
            return None, True

    async def _inspect_oom(self) -> bool:
        """Return True if the container was killed by the OOM killer."""
        if self._container is None or self._client is None:
            return False
        container_id: str | None = getattr(self._container, "id", None)
        if container_id is None:
            return False
        loop = asyncio.get_event_loop()
        try:
            info: dict[str, Any] = await loop.run_in_executor(
                None,
                lambda: self._client.api.inspect_container(container_id),  # type: ignore[union-attr]
            )
            state: dict[str, Any] = info.get("State", {})
            return bool(state.get("OOMKilled", False))
        except Exception:
            return False

    async def _terminate_container(self) -> None:
        if self._container is not None:
            loop = asyncio.get_event_loop()
            container = self._container
            self._container = None
            try:
                await loop.run_in_executor(None, container.kill)
            except Exception:
                pass
            try:
                await loop.run_in_executor(None, lambda: container.remove(force=True))
            except Exception:
                pass

    @staticmethod
    def _classify_status(
        exit_code: int,
        stdout: str,
        stderr: str,
        oom_killed: bool,
    ) -> StatusLiteral:
        # Wall-clock timeout (harness timer fired before container exited).
        if exit_code == _TIMEOUT_EXIT_CODE and "TIMEOUT" in stderr:
            return "TIMEOUT"

        # OOM: verified via container inspect, not exit code alone.
        if oom_killed:
            return "OOM"

        # PID exhaustion / fork-bomb: kernel raises EAGAIN (errno 11).
        if exit_code == _PID_EXHAUSTION_ERRNO or (
            f"[Errno {_PID_EXHAUSTION_ERRNO}]" in stdout
            or f"[Errno {_PID_EXHAUSTION_ERRNO}]" in stderr
        ):
            return "VIOLATION"

        # Network access attempt: kernel raises ENETUNREACH (errno 101) with
        # network_mode="none".  Treat as a violation (reward -1.0).
        if (
            f"[Errno {_NETWORK_UNREACHABLE_ERRNO}]" in stdout
            or f"[Errno {_NETWORK_UNREACHABLE_ERRNO}]" in stderr
        ):
            return "VIOLATION"

        if exit_code == 0:
            return "PASSED"

        return "FAILED"

    async def __aenter__(self) -> "DockerSandboxProvider":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self._terminate_container()
