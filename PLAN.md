# PLAN.md

# rl-code-sandbox: Architecture and Implementation Plan

## System Overview
`rl-code-sandbox` is an asynchronous execution engine, API service, and evaluation pipeline that executes untrusted, model-generated Python code inside hardened OCI Docker containers. It grades candidates with deterministic test harnesses into scalar reinforcement learning rewards, logs telemetry traces, benchmarks multi-candidate task distributions, and trains a PyTorch multi-layer perceptron (MLP) pass/fail predictor evaluated against a scikit-learn baseline.

### Core Security Perimeter
The Docker container is the authoritative security boundary. Static AST inspection serves strictly as a pre-execution heuristic filter (defense-in-depth) to fast-fail known prohibited constructs.

```
                           +----------------------------------------+
                           |     FastAPI Service / CLI Engine       |
                           +----------------------------------------+
                                              |
                                              v
                           +----------------------------------------+
                           |     Security Filter (AST Analyzer)     |
                           +----------------------------------------+
                                              |
                     +------------------------+------------------------+
                     | (Passed AST Filter)                             | (Violated AST Filter)
                     v                                                 v
+------------------------------------------+       +------------------------------------+
|         SandboxProvider (Interface)      |       | Fast Rejection Trace               |
|  - MockSandboxProvider (Offline Test)    |       | Reward = -1.0 (VIOLATION)          |
|  - DockerSandboxProvider (Hardened OCI)  |       +------------------------------------+
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|       Hardened Container Instance        |
|  - network_mode: "none"                  |
|  - cap_drop: ["ALL"]                     |
|  - security_opt: ["no-new-privileges"]   |
|  - read_only: true                       |
|  - tmpfs: /tmp (noexec,nosuid,nodev,32m) |
|  - pids_limit: 64                        |
|  - mem_limit: 256m, memswap_limit: 256m  |
|  - nano_cpus: 1000000000                 |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|       Deterministic Reward Engine        |
|  - Pass all assertions:     Reward  1.0  |
|  - Assertion / Run Error:   Reward  0.0  |
|  - Timeout (Wall-clock):    Reward  0.0  |
|  - OOM (Inspect OOMKilled): Reward  0.0  |
|  - PID exhaustion / Bomb:   Reward -1.0  |
|  - Network attempt (E101):  Reward -1.0  |
|  - AST Check Block:         Reward -1.0  |
+------------------------------------------+
                     |
                     v
+---------------------------------------------------------------------------------+
| Benchmark Suite & Dataset Logger -> PyTorch Pass/Fail Predictor (GroupKFold CV) |
+---------------------------------------------------------------------------------+
```

---

## Reward and Violation Taxonomy

Rewards are deterministic and mapped as follows:

| Outcome | Exit / Signal Indicator | Status Classification | Reward |
| :--- | :--- | :--- | :--- |
| **All Test Assertions Pass** | Exit code `0`, assertions verified | `PASSED` | `1.0` |
| **Test Assertion Failure** | Exit code != 0, test assertion failed | `FAILED` | `0.0` |
| **Standard Runtime Exception** | `ZeroDivisionError`, `IndexError`, `TypeError`, etc. | `FAILED` | `0.0` |
| **Execution Timeout** | Harness wall-clock timer expired, container killed | `TIMEOUT` | `0.0` |
| **Out-Of-Memory (OOM)** | Container inspection `OOMKilled: true` | `OOM` | `0.0` |
| **PID Limit Exhaustion** | Fork bomb / `BlockingIOError: [Errno 11]` | `VIOLATION` | `-1.0` |
| **Network Access Attempt** | Harness intercepts `OSError: [Errno 101] Network is unreachable` (heuristic) | `VIOLATION` | `-1.0` |
| **AST Security Rejection** | Forbidden import, eval/exec, or private attribute tampering | `VIOLATION` | `-1.0` |

---

## Module and Directory Layout

```
rl-code-sandbox/
├── .github/
│   └── workflows/
│       └── ci.yml               # Offline gate + Ubuntu Docker integration gate
├── src/
│   └── rl_sandbox/
│       ├── __init__.py
│       ├── py.typed
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── models.py        # CodeSubmission, EvaluationResult, ExecutionStats
│       │   └── errors.py        # Domain exceptions
│       ├── security/
│       │   ├── __init__.py
│       │   └── ast_guard.py     # AST-based static syntax/import security rules
│       ├── sandbox/
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract SandboxProvider
│       │   ├── mock_provider.py # Mock provider for 100% offline unit/API testing
│       │   └── docker_provider.py # Hardened OCI Docker implementation
│       ├── evaluator/
│       │   ├── __init__.py
│       │   └── reward.py        # Deterministic scoring and reward mapping
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py           # FastAPI application
│       │   ├── routes.py        # POST /evaluate, GET /health
│       │   └── schemas.py       # Pydantic v2 request/response schemas
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py          # Typer CLI application (rl-sandbox)
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   ├── runner.py        # Benchmark harness (~30 tasks x candidates)
│       │   └── tasks/           # Vendored offline task fixtures
│       └── ml/
│           ├── __init__.py
│           ├── dataset.py       # GroupKFold dataset extraction
│           ├── features.py      # Candidate AST feature extractor
│           └── predictor.py     # PyTorch MLP vs scikit-learn DummyClassifier baseline
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_ast_guard.py
│   │   ├── test_domain_models.py
│   │   └── test_reward_engine.py
│   ├── integration/
│   │   ├── test_api_offline.py
│   │   ├── test_cli_offline.py
│   │   └── test_ml_pipeline.py
│   └── docker/
│       └── test_docker_sandbox.py # Marked with @pytest.mark.docker
├── pyproject.toml
├── README.md
├── PLAN.md
├── CHECKLIST.md
└── DECISIONS.md
```

---

## Detailed Component Specifications

### 1. Static Security Guard (`src/rl_sandbox/security/ast_guard.py`)
- Analyzes candidate code prior to container dispatch using Python's standard `ast` module.
- Rejects prohibited modules: `os`, `sys`, `subprocess`, `socket`, `importlib`, `pty`, `shutil`.
- Rejects dynamic execution primitives: `eval`, `exec`, `compile`, `__import__`.
- Flags attempts to traverse private object trees (e.g., `__subclasses__`, `__globals__`, `__code__`).
- Produces an immediate evaluation trace with `status="VIOLATION"` and `reward=-1.0`.

### 2. Container Sandbox Provider (`src/rl_sandbox/sandbox/docker_provider.py`)
- Image: Pinned `python:3.11-slim@sha256:...` digest (resolved at implementation time).
- Host Compatibility: Windows 11 Docker Desktop (WSL2 backend) and Ubuntu native Linux.
- Isolation Flags:
  - `network_mode="none"`
  - `cap_drop=["ALL"]`
  - `security_opt=["no-new-privileges:true"]`
  - `read_only=True`
  - `pids_limit=64`
  - `mem_limit="256m"`, `memswap_limit="256m"`
  - `nano_cpus=1000000000` (1.0 CPU equivalent for `docker-py`)
  - `tmpfs={"/tmp": "rw,noexec,nosuid,nodev,size=32m"}`
- Code Ingestion: Transferred via stdin to `python -` or written to ephemeral non-executable mount; no persistent host volume mounts.
- Metrics & Telemetry:
  - Container provisioning time (`provisioning_time_ms`) timed and reported separately from script execution time (`execution_time_ms`).
  - Memory usage read via cgroup v2 (`memory.peak`) when available, falling back gracefully to Docker stats with `best_effort: true`.
- Timeout & OOM Detection:
  - Wall-clock timeout is enforced and detected via the harness timer.
  - OOM conditions are explicitly inspected via container state (`OOMKilled: true`), avoiding false classification from exit code 137 alone.

### 3. Reward and Evaluation Engine (`src/rl_sandbox/evaluator/reward.py`)
- Coordinates the evaluation pipeline:
  1. AST inspection.
  2. Sandbox execution dispatch.
  3. Output parsing and exception analysis.
- Maps outcome into strict `EvaluationResult`:
  - `passed`: Boolean indicator.
  - `reward`: Floating-point scalar (`1.0`, `0.0`, `-1.0`).
  - `status`: String literal (`PASSED`, `FAILED`, `TIMEOUT`, `OOM`, `VIOLATION`).
  - `stats`: `ExecutionStats` (timings, exit code, peak memory).
  - `stdout`, `stderr`: Captured outputs.

### 4. API & CLI Interfaces
- **FastAPI (`src/rl_sandbox/api/`)**:
  - `POST /evaluate`: Accepts `EvaluationRequest` (code, test harness, resource limits), returns `EvaluationResult`.
  - `GET /health`: Reports service status and Docker provider connectivity.
- **Typer CLI (`src/rl_sandbox/cli/`)**:
  - Registered as `rl-sandbox`.
  - Commands:
    - `rl-sandbox run --code "..." --test "..."`
    - `rl-sandbox benchmark --tasks-dir ... --output results.jsonl`

### 5. Benchmark Suite & Predictor Pipeline
- **Dataset**: ~30 distinct algorithmic tasks with 8–12 completions each (~300 runs total) vendored as static offline fixtures.
- **Feature Extraction (`src/rl_sandbox/ml/features.py`)**: Computes structural code metrics per candidate (AST depth, node count, cyclomatic complexity, control flow branch count, exception handling density).
- **Dataset Splitting (`src/rl_sandbox/ml/dataset.py`)**: Strict `GroupKFold` split on `task_id` using `scikit-learn`. No task appears in both training and test folds. Note: AST-blocked runs are filtered out and excluded from the training dataset because they never execute.
- **Model Architecture (`src/rl_sandbox/ml/predictor.py`)**:
  - Lightweight PyTorch MLP (Input Layer -> Linear(32) -> ReLU -> Dropout(0.2) -> Linear(16) -> ReLU -> Linear(1) -> Sigmoid).
  - Binary classification target: `1` if `reward == 1.0` else `0`.
- **Baseline Comparison**:
  - Evaluated against `DummyClassifier(strategy="most_frequent")`.
  - Metrics tracked: Accuracy, Balanced Accuracy, ROC-AUC, Brier Score.

### 6. Continuous Integration (GitHub Actions)
- **Job 1 (Offline Fast Gate, <2m target)**:
  - `ruff check .`
  - `ruff format --check .`
  - `mypy --strict src`
  - `pytest -m "not docker" --cov=rl_sandbox --cov-fail-under=90` (excluding docker provider from offline gate).
- **Job 2 (Integration Docker Gate, <5m target)**:
  - Runs on `ubuntu-latest`.
  - Pre-pulls pinned container image digest.
  - Runs `pytest -m "docker"`.