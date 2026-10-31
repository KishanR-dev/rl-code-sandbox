"""Integration tests for DockerSandboxProvider. Require a live Docker daemon."""

import pytest

from rl_sandbox.domain.models import CodeSubmission, TestHarness
from rl_sandbox.sandbox.docker_provider import DockerSandboxProvider


@pytest.mark.docker
class TestDockerSandboxProvider:
    """Live Docker integration tests — skipped in offline CI."""

    @pytest.fixture()
    def provider(self) -> DockerSandboxProvider:
        return DockerSandboxProvider()

    @pytest.fixture()
    def passing_submission(self) -> CodeSubmission:
        return CodeSubmission(
            code="x = 1 + 1",
            task_id="task-1",
            candidate_id="cand-1",
        )

    @pytest.fixture()
    def passing_harness(self) -> TestHarness:
        return TestHarness(assertions="assert x == 2")

    @pytest.fixture()
    def failing_submission(self) -> CodeSubmission:
        return CodeSubmission(
            code="x = 1 + 1",
            task_id="task-1",
            candidate_id="cand-1",
        )

    @pytest.fixture()
    def failing_harness(self) -> TestHarness:
        return TestHarness(assertions="assert x == 99")

    @pytest.fixture()
    def timeout_submission(self) -> CodeSubmission:
        return CodeSubmission(
            code="import time; time.sleep(60)",
            task_id="task-1",
            candidate_id="cand-1",
            timeout_seconds=2.0,
        )

    @pytest.fixture()
    def timeout_harness(self) -> TestHarness:
        return TestHarness(assertions="")

    @pytest.mark.asyncio
    async def test_passing_code_returns_passed(
        self,
        provider: DockerSandboxProvider,
        passing_submission: CodeSubmission,
        passing_harness: TestHarness,
    ) -> None:
        result = await provider.run(passing_submission, passing_harness)
        assert result.status == "PASSED"
        assert result.stats.exit_code == 0

    @pytest.mark.asyncio
    async def test_failing_assertion_returns_failed(
        self,
        provider: DockerSandboxProvider,
        failing_submission: CodeSubmission,
        failing_harness: TestHarness,
    ) -> None:
        result = await provider.run(failing_submission, failing_harness)
        assert result.status == "FAILED"
        assert result.stats.exit_code != 0

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout(
        self,
        provider: DockerSandboxProvider,
        timeout_submission: CodeSubmission,
        timeout_harness: TestHarness,
    ) -> None:
        result = await provider.run(timeout_submission, timeout_harness)
        assert result.status == "TIMEOUT"

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up(
        self,
        passing_submission: CodeSubmission,
        passing_harness: TestHarness,
    ) -> None:
        async with DockerSandboxProvider() as provider:
            result = await provider.run(passing_submission, passing_harness)
        assert result.status == "PASSED"
        assert provider._container is None

    @pytest.mark.asyncio
    async def test_network_is_disabled(
        self,
        provider: DockerSandboxProvider,
    ) -> None:
        submission = CodeSubmission(
            code="import socket; socket.create_connection(('8.8.8.8', 53), timeout=1)",
            task_id="task-net",
            candidate_id="cand-net",
        )
        harness = TestHarness(assertions="")
        result = await provider.run(submission, harness)
        assert result.status in {"FAILED", "TIMEOUT", "VIOLATION"}

    @pytest.mark.asyncio
    async def test_stats_populated(
        self,
        provider: DockerSandboxProvider,
        passing_submission: CodeSubmission,
        passing_harness: TestHarness,
    ) -> None:
        result = await provider.run(passing_submission, passing_harness)
        assert result.stats.provisioning_time_ms >= 0.0
        assert result.stats.execution_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_user_is_unprivileged(
        self,
        provider: DockerSandboxProvider,
    ) -> None:
        submission = CodeSubmission(
            code="import os; assert os.getuid() == 1000",
            task_id="task-uid",
            candidate_id="cand-uid",
        )
        harness = TestHarness(assertions="")
        result = await provider.run(submission, harness)
        assert result.status == "PASSED"
        assert result.stats.exit_code == 0
