"""Benchmark runner for rl-sandbox evaluating static task fixtures."""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any

from rl_sandbox.domain.models import CodeSubmission, TestHarness
from rl_sandbox.evaluator.reward import RewardEngine
from rl_sandbox.sandbox.base import SandboxProvider


class BenchmarkRunner:
    """Runner to execute benchmark task fixtures against a SandboxProvider."""

    def __init__(self, provider: SandboxProvider) -> None:
        self._engine = RewardEngine(provider)

    async def run_directory(
        self, tasks_dir: pathlib.Path, output_path: pathlib.Path
    ) -> dict[str, Any]:
        """Run evaluation over all jsonl task fixtures in tasks_dir."""
        if not tasks_dir.is_dir():
            raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

        fixtures = sorted(tasks_dir.glob("*.jsonl"))
        if not fixtures:
            raise ValueError(f"No JSONL task fixtures found in {tasks_dir}")

        total_evaluated = 0
        total_passed = 0
        total_prov_ms = 0.0
        total_exec_ms = 0.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as out_fh:
            for fixture_path in fixtures:
                with fixture_path.open(encoding="utf-8") as in_fh:
                    for raw_line in in_fh:
                        line = raw_line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        submission = CodeSubmission(**record["submission"])
                        harness = TestHarness(**record["harness"])

                        result = await self._engine.evaluate(submission, harness)

                        total_evaluated += 1
                        if result.passed:
                            total_passed += 1
                        total_prov_ms += result.execution.stats.provisioning_time_ms
                        total_exec_ms += result.execution.stats.execution_time_ms

                        stats = result.execution.stats
                        out_record = {
                            "task_id": submission.task_id,
                            "candidate_id": submission.candidate_id,
                            "status": result.execution.status,
                            "reward": result.reward,
                            "passed": result.passed,
                            "provisioning_ms": stats.provisioning_time_ms,
                            "execution_ms": stats.execution_time_ms,
                            "peak_memory_bytes": stats.peak_memory_bytes,
                        }
                        out_fh.write(json.dumps(out_record) + "\n")

        pass_rate = (total_passed / total_evaluated) if total_evaluated > 0 else 0.0
        avg_prov = (total_prov_ms / total_evaluated) if total_evaluated > 0 else 0.0
        avg_exec = (total_exec_ms / total_evaluated) if total_evaluated > 0 else 0.0

        summary = {
            "total_evaluated": total_evaluated,
            "total_passed": total_passed,
            "pass_rate": pass_rate,
            "avg_provisioning_ms": avg_prov,
            "avg_execution_ms": avg_exec,
        }
        return summary


def run_benchmark(
    tasks_dir: str | pathlib.Path,
    output_path: str | pathlib.Path,
    provider: SandboxProvider,
) -> dict[str, Any]:
    """Synchronous helper wrapper for BenchmarkRunner."""
    runner = BenchmarkRunner(provider)
    return asyncio.run(
        runner.run_directory(pathlib.Path(tasks_dir), pathlib.Path(output_path))
    )
