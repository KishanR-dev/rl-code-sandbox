"""Command-line interface for rl-sandbox."""

from __future__ import annotations

import asyncio
import json

import typer

from rl_sandbox.domain.models import CodeSubmission, EvaluationRequest, TestHarness
from rl_sandbox.evaluator.reward import RewardEngine
from rl_sandbox.sandbox.base import SandboxProvider
from rl_sandbox.sandbox.mock_provider import MockSandboxProvider

app = typer.Typer(
    help="rl-code-sandbox: secure code execution sandbox for reinforcement learning."
)


def _build_provider(mock: bool) -> SandboxProvider:
    if mock:
        return MockSandboxProvider(outcome="pass")
    try:
        from rl_sandbox.sandbox.docker_provider import DockerSandboxProvider

        return DockerSandboxProvider()
    except Exception as exc:
        typer.echo(f"Docker provider unavailable: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def version() -> None:
    """Print version information."""
    typer.echo("rl-sandbox 0.1.0")


@app.command()
def run(
    code: str = typer.Option(
        ..., "--code", "-c", help="Python source code to execute."
    ),
    test: str = typer.Option(
        ..., "--test", "-t", help="Python assert statements to validate the code."
    ),
    task_id: str = typer.Option("cli-task", "--task-id", help="Task identifier."),
    candidate_id: str = typer.Option(
        "cli-candidate", "--candidate-id", help="Candidate identifier."
    ),
    timeout: float = typer.Option(
        10.0, "--timeout", help="Wall-clock timeout seconds."
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Use MockSandboxProvider instead of Docker."
    ),
) -> None:
    """Evaluate a code submission against test assertions."""
    request = EvaluationRequest(
        submission=CodeSubmission(
            code=code,
            task_id=task_id,
            candidate_id=candidate_id,
            timeout_seconds=timeout,
        ),
        harness=TestHarness(assertions=test),
        use_mock=mock,
    )
    engine = RewardEngine(_build_provider(mock))
    result = asyncio.run(engine.evaluate(request.submission, request.harness))
    output = {
        "status": result.execution.status,
        "reward": result.reward,
        "passed": result.passed,
        "stdout": result.execution.stdout,
        "stderr": result.execution.stderr,
        "provisioning_ms": result.execution.stats.provisioning_time_ms,
        "execution_ms": result.execution.stats.execution_time_ms,
    }
    typer.echo(json.dumps(output, indent=2))
    if not result.passed:
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    tasks_dir: str = typer.Option(
        ..., "--tasks-dir", help="Directory containing JSONL task fixture files."
    ),
    output: str = typer.Option(
        "results.jsonl", "--output", "-o", help="Output JSONL file path."
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Use MockSandboxProvider instead of Docker."
    ),
) -> None:
    """Run benchmark evaluation over a directory of task fixtures."""
    import pathlib

    tasks_path = pathlib.Path(tasks_dir)
    if not tasks_path.is_dir():
        typer.echo(f"tasks-dir not found: {tasks_dir}", err=True)
        raise typer.Exit(code=1)

    fixtures = sorted(tasks_path.glob("*.jsonl"))
    if not fixtures:
        typer.echo(f"No JSONL fixtures found in {tasks_dir}", err=True)
        raise typer.Exit(code=1)

    engine = RewardEngine(_build_provider(mock))
    output_path = pathlib.Path(output)
    count = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for fixture_path in fixtures:
            with fixture_path.open(encoding="utf-8") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    record = json.loads(raw_line)
                    submission = CodeSubmission(**record["submission"])
                    harness = TestHarness(**record["harness"])
                    result = asyncio.run(engine.evaluate(submission, harness))
                    out_fh.write(
                        json.dumps(
                            {
                                "task_id": submission.task_id,
                                "candidate_id": submission.candidate_id,
                                "status": result.execution.status,
                                "reward": result.reward,
                                "passed": result.passed,
                                "provisioning_ms": (
                                    result.execution.stats.provisioning_time_ms
                                ),
                                "execution_ms": (
                                    result.execution.stats.execution_time_ms
                                ),
                            }
                        )
                        + "\n"
                    )
                    count += 1

    typer.echo(f"Evaluated {count} candidates. Results written to {output}")


if __name__ == "__main__":
    app()
