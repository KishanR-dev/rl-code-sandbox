# CHECKLIST.md

# rl-code-sandbox: Implementation Checklist

## Phase 1: Tooling, Packaging & Domain Models
- [x] 1.1: Initialize `pyproject.toml` with Python 3.11, Hatchling build backend, console script `rl-sandbox = "rl_sandbox.cli.main:app"`, and core dependencies (`fastapi`, `uvicorn`, `pydantic`, `typer`, `docker`, `torch`, `scikit-learn`).
  * *Acceptance Criteria:* `pip install -e .` succeeds in a clean virtual environment; running `rl-sandbox --help` outputs CLI usage without error.
- [x] 1.2: Configure toolchain in `pyproject.toml` for `ruff`, `mypy --strict`, and `pytest`.
  * *Acceptance Criteria:* Running `ruff check .`, `ruff format --check .`, and `mypy --strict src` passes with zero errors or warnings.
- [x] 1.3: Add `types-docker` and required type stubs; configure per-file security ignores for tests and fixtures in `ruff`.
  * *Acceptance Criteria:* Security testing fixtures containing intentionally unsafe strings do not trigger `ruff` S-rule failures.
- [x] 1.4: Implement core domain models in `src/rl_sandbox/domain/models.py`.
  * *Acceptance Criteria:* Pydantic v2 models defined for `CodeSubmission`, `TestHarness`, `ExecutionStats`, `ExecutionResult`, and `EvaluationResult` with strict type validation.
- [x] 1.5: Define deterministic reward enumeration and domain exceptions in `src/rl_sandbox/domain/`.
  * *Acceptance Criteria:* `RewardValue` literal/enum strictly covers `1.0`, `0.0`, `-1.0`; custom exceptions defined for configuration, security, and execution errors.

## Phase 2: Static Security Boundary (AST Guard)
- [x] 2.1: Implement `ASTGuard` in `src/rl_sandbox/security/ast_guard.py`.
  * *Acceptance Criteria:* Static AST visitor inspects code strings and blocks forbidden modules (`os`, `sys`, `subprocess`, `socket`, `importlib`), builtin tampering, and dynamic execution (`eval`, `exec`).
- [x] 2.2: Unit test suite for AST Guard in `tests/unit/test_ast_guard.py`.
  * *Acceptance Criteria:* Suite tests direct and obfuscated import attempts; achieves 100% test coverage on `ast_guard.py`.

## Phase 3: Hardened Container Infrastructure
- [x] 3.1: Define abstract base provider `SandboxProvider` in `src/rl_sandbox/sandbox/base.py`.
  * *Acceptance Criteria:* Abstract interface declares `async def run(self, submission: CodeSubmission, harness: TestHarness) -> ExecutionResult` and lifecycle context management.
- [x] 3.2: Implement `MockSandboxProvider` in `src/rl_sandbox/sandbox/mock_provider.py`.
  * *Acceptance Criteria:* Provider simulates pass, assertion failure, runtime error, OOM, timeout, and PID exhaustion 100% offline without Docker daemon.
- [x] 3.3: Implement `DockerSandboxProvider` in `src/rl_sandbox/sandbox/docker_provider.py`.
  * *Acceptance Criteria:* Configures OCI container with pinned image `python:3.11-slim@sha256:...`, `network_mode="none"`, `cap_drop=["ALL"]`, `security_opt=["no-new-privileges:true"]`, `read_only=True`, `pids_limit=64`, `mem_limit="256m"`, `memswap_limit="256m"`, and `nano_cpus=1000000000`.
- [x] 3.4: Configure tmpfs mount with security restrictions.
  * *Acceptance Criteria:* `/tmp` mounted with `rw,noexec,nosuid,nodev,size=32m`.
- [x] 3.5: Implement code delivery mechanism.
  * *Acceptance Criteria:* Untrusted code delivered via stdin or ephemeral memory mount; zero persistent host bind mounts created.
- [x] 3.6: Implement wall-clock execution timeout with deterministic container termination.
  * *Acceptance Criteria:* Process exceeding `timeout_seconds` triggers harness-detected wall-clock expiration, forceful container SIGKILL, and records timeout status (`0.0` reward).
- [x] 3.7: Implement latency decomposition and resource metrics collection.
  * *Acceptance Criteria:* `ExecutionStats` measures and reports `provisioning_time_ms` and `execution_time_ms` separately; retrieves `cgroup v2` `memory.peak` or sets `best_effort=True`.
- [x] 3.8: Implement container violation rules & OOM/Timeout detection.
  * *Acceptance Criteria:* PID exhaustion (exit code / errno 11) yields `-1.0`; test harness intercept of `OSError: [Errno 101]` yields `-1.0`; OOM verified strictly via container inspect `OOMKilled=true` yielding `0.0`; wall-clock timeout yielding `0.0`.
- [x] 3.9: Integration tests for Docker provider in `tests/docker/test_docker_sandbox.py` marked `@pytest.mark.docker`.
  * *Acceptance Criteria:* Integration tests pass on Docker-enabled hosts (verifying network isolation, resource limits, and timeout kills).

## Phase 4: Deterministic Evaluator & Reward Engine
- [x] 4.1: Implement `RewardEngine` in `src/rl_sandbox/evaluator/reward.py`.
  * *Acceptance Criteria:* Computes deterministic rewards: `1.0` for all tests passing, `0.0` for assertion failures, runtime errors, wall-clock timeouts, and container OOMs (`OOMKilled=true`), and `-1.0` for AST rejections, PID limit exhaustion, and network access attempts.
- [x] 4.2: Unit test suite for Evaluator in `tests/unit/test_reward_engine.py`.
  * *Acceptance Criteria:* Offline tests verify reward and status mappings across all execution permutations; achieves 100% coverage on `reward.py`.

## Phase 5: Async API Service & CLI
- [x] 5.1: Implement FastAPI application and schemas in `src/rl_sandbox/api/`.
  * *Acceptance Criteria:* `GET /health` returns service status; `POST /evaluate` accepts `EvaluationRequest` and returns `EvaluationResult` with complete execution trace.
- [x] 5.2: Implement Typer CLI in `src/rl_sandbox/cli/main.py`.
  * *Acceptance Criteria:* Console script `rl-sandbox` executes `run` and `benchmark` commands with formatted console output.
- [x] 5.3: Offline integration tests for API and CLI in `tests/integration/`.
  * *Acceptance Criteria:* API and CLI test suites execute 100% offline using `MockSandboxProvider`, validating endpoints and command-line exit codes.

## Phase 6: Benchmark Suite & Data Collection
- [x] 6.1: Curate static task fixtures in `src/rl_sandbox/benchmarks/tasks/`.
  * *Acceptance Criteria:* 30 distinct tasks with 8–12 completions each (~300 samples total) stored in offline JSONL format covering passes, logic bugs, timeouts, OOMs, and security violations.
- [x] 6.2: Implement benchmark runner in `src/rl_sandbox/benchmarks/runner.py`.
  * *Acceptance Criteria:* Runner iterates over benchmark tasks, evaluates candidates, captures separated latency stats, and records structured JSONL output.
- [x] 6.3: Validate benchmark dataset artifact.
  * *Acceptance Criteria:* Emitted JSONL contains valid evaluations, correct reward distributions, and non-null execution statistics.

## Phase 7: PyTorch Pass/Fail Predictor
- [x] 7.1: Implement candidate AST feature extractor in `src/rl_sandbox/ml/features.py`.
  * *Acceptance Criteria:* Extracts candidate-level structural features (AST depth, cyclomatic complexity, token length, branch counts) that vary across candidates for the same task.
- [x] 7.2: Implement dataset loader with GroupKFold split & AST filtering in `src/rl_sandbox/ml/dataset.py`.
  * *Acceptance Criteria:* Filters out AST-blocked runs (since they never execute); uses `scikit-learn` `GroupKFold` split grouped by `task_id`; zero overlap of tasks between training and test folds.
- [x] 7.3: Implement lightweight PyTorch MLP model in `src/rl_sandbox/ml/predictor.py`.
  * *Acceptance Criteria:* 2-layer MLP classifier with dropout and weight decay predicting binary outcome (`label = 1` if `reward == 1.0` else `0`).
- [x] 7.4: Implement training loop and baseline comparison evaluation.
  * *Acceptance Criteria:* Trains PyTorch MLP and evaluates against `DummyClassifier(strategy="most_frequent")` baseline across folds; records Accuracy, Balanced Accuracy, ROC-AUC, and Brier Score.
- [x] 7.5: Offline integration test for ML pipeline in `tests/integration/test_ml_pipeline.py`.
  * *Acceptance Criteria:* End-to-end test trains predictor on small synthetic grouped dataset, verifying tensor shapes, training steps, and metric computations.

## Phase 8: CI/CD Pipeline & End-to-End Verification
- [x] 8.1: Create GitHub Actions workflow `.github/workflows/ci.yml`.
  * *Acceptance Criteria:* Implements two isolated jobs:
    1. `offline-gate`: Runs on `ubuntu-latest` (<2m target); runs `ruff`, `ruff format`, `mypy --strict`, and `pytest -m "not docker" --cov=rl_sandbox --cov-fail-under=90`.
    2. `docker-integration`: Runs on `ubuntu-latest` (<5m target); pulls pinned container image digest, runs `pytest -m "docker"`.
- [x] 8.2: Run full verification suite on local Windows PowerShell + Docker Desktop environment.
  * *Acceptance Criteria:* Offline tests pass locally; Docker integration tests pass against local Docker Desktop daemon.

## Phase 9: Documentation & Portfolio Polish
- [x] 9.1: Produce comprehensive `README.md`.
  * *Acceptance Criteria:* Documents problem statement, architecture diagrams, security boundary specifications, API documentation, and empirical benchmark tables with commit hash and hardware environment.
- [x] 9.2: Verify absolute compliance with Rule D-011.
  * *Acceptance Criteria:* Codebase, commit messages, comments, docstrings, and documentation contain zero conversational references or AI assistant attribution.
