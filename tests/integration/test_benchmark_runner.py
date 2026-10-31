"""Integration tests for BenchmarkRunner using MockSandboxProvider."""

from __future__ import annotations

import asyncio
import pathlib

import pytest

from rl_sandbox.benchmarks.runner import BenchmarkRunner, run_benchmark
from rl_sandbox.sandbox.mock_provider import MockSandboxProvider


@pytest.mark.asyncio
async def test_benchmark_runner_directory(tmp_path: pathlib.Path) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    fixture_file = tasks_dir / "task-01.jsonl"
    sub_json = '{"code": "x = 1", "task_id": "t1", "candidate_id": "c1"}'
    harn_json = '{"assertions": "assert x == 1"}'
    fixture_file.write_text(
        f'{{"submission": {sub_json}, "harness": {harn_json}}}\n',
        encoding="utf-8",
    )

    output_file = tmp_path / "results.jsonl"
    provider = MockSandboxProvider(outcome="pass")
    runner = BenchmarkRunner(provider)
    summary = await runner.run_directory(tasks_dir, output_file)

    assert summary["total_evaluated"] == 1
    assert summary["total_passed"] == 1
    assert summary["pass_rate"] == 1.0
    assert output_file.exists()


def test_run_benchmark_sync_helper(tmp_path: pathlib.Path) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    fixture_file = tasks_dir / "task-01.jsonl"
    sub_json = '{"code": "x = 1", "task_id": "t1", "candidate_id": "c1"}'
    harn_json = '{"assertions": "assert x == 1"}'
    fixture_file.write_text(
        f'{{"submission": {sub_json}, "harness": {harn_json}}}\n',
        encoding="utf-8",
    )

    output_file = tmp_path / "results.jsonl"
    provider = MockSandboxProvider(outcome="pass")
    summary = run_benchmark(tasks_dir, output_file, provider)

    assert summary["total_evaluated"] == 1
    assert summary["pass_rate"] == 1.0


def test_benchmark_runner_errors(tmp_path: pathlib.Path) -> None:
    provider = MockSandboxProvider(outcome="pass")
    runner = BenchmarkRunner(provider)
    output_file = tmp_path / "out.jsonl"

    with pytest.raises(FileNotFoundError):
        asyncio.run(runner.run_directory(tmp_path / "nonexistent", output_file))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        asyncio.run(runner.run_directory(empty_dir, output_file))
