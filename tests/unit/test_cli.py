"""Unit tests for the CLI entrypoint."""

from __future__ import annotations

from typer.testing import CliRunner

from rl_sandbox.cli.main import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "rl-code-sandbox" in result.output


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "rl-sandbox 0.1.0" in result.output
