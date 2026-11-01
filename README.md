# rl-code-sandbox

An async Python 3.11 service and CLI that executes model-generated code in isolated Docker sandboxes with CPU, memory, time, and network constraints, scores each execution using deterministic test suites, and logs evaluation traces. Includes an embedded PyTorch MLP to predict test pass/fail outcomes from code execution features.

Repository: [https://github.com/KishanR-dev/rl-code-sandbox](https://github.com/KishanR-dev/rl-code-sandbox)

---

## Architecture

- **Isolation Provider**: Ephemeral Docker containers configured with Docker's default seccomp profile, zero network egress (`--network none`), read-only root filesystems, unprivileged user execution (`1000:1000`), and hardware resource quotas (`nano_cpus`, `mem_limit`).
- **Offline Mock Provider**: Pure in-memory sandbox simulator for deterministic offline unit testing without external dependencies or subprocesses.
- **Static Guardrails**: AST pre-execution filter intercepting disallowed imports (network, raw sockets, process manipulation).
- **Deterministic Scorer**: Evaluates execution status, assertion pass rates, and runtime constraints into normalized scalar rewards.
- **PyTorch Predictor**: Lightweight two-hidden-layer MLP predicting pass/fail probabilities from static and dynamic execution features, benchmarked against a majority-class baseline.
- **Service & CLI**: Single-endpoint FastAPI service (`POST /evaluate`) accompanied by a unified command-line tool.

---

## Benchmark & Performance Metrics

| Metric | Measured Value |
| :--- | :--- |
| **Total Tasks** | 30 |
| **Candidates Evaluated** | 300 |
| **Passed** | 180 |
| **Failed** | 60 |
| **Timeout** | 30 |
| **Violation** | 30 |
| **Reward 1.0 Count** | 180 |
| **Reward 0.0 Count** | 90 |
| **Reward -1.0 Count** | 30 |
| **MLP Accuracy (Mean)** | 0.6667 |
| **Baseline Accuracy (Mean)** | 0.6667 |
| **MLP Brier Score (Mean)** | 0.2283 |
| **Baseline Brier Score (Mean)** | 0.3333 |

Predictor result: the MLP matches the majority-class baseline on accuracy (0.6667) and has chance-level ROC AUC (0.5000) under 5-fold GroupKFold by task_id, so it does not separate passing from failing candidates on this benchmark. Its mean Brier score is lower (0.2283 vs 0.3333). Every task has the same candidate mix (6 pass, 2 fail, 1 timeout, 1 violation), which likely leaves the current features with no task-independent signal.

---

## Project Structure

```
rl-code-sandbox/
├── CHECKLIST.md
├── DECISIONS.md
├── PLAN.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .github/
│   ├── copilot-instructions.md
│   └── workflows/
│       └── ci.yml
├── scripts/
│   └── generate_fixtures.py
├── src/
│   └── rl_sandbox/
│       ├── __init__.py
│       ├── py.typed
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── routes.py
│       │   └── schemas.py
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   ├── runner.py
│       │   └── tasks/
│       │       └── task-01 through task-30 .jsonl
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── errors.py
│       │   └── models.py
│       ├── evaluator/
│       │   ├── __init__.py
│       │   └── reward.py
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── features.py
│       │   └── predictor.py
│       ├── sandbox/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── docker_provider.py
│       │   └── mock_provider.py
│       └── security/
│           ├── __init__.py
│           └── ast_guard.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── docker/
    │   ├── __init__.py
    │   └── test_docker_sandbox.py
    ├── integration/
    │   ├── __init__.py
    │   ├── test_api_offline.py
    │   ├── test_benchmark_runner.py
    │   ├── test_cli_offline.py
    │   └── test_ml_pipeline.py
    └── unit/
        ├── __init__.py
        ├── test_ast_guard.py
        ├── test_cli.py
        ├── test_domain_models.py
        ├── test_mock_provider.py
        ├── test_reward_engine.py
        └── test_sandbox_base.py
```

---

## Setup & Development

### Prerequisites
- Python 3.11+
- `uv`
- Docker (optional; required only for `@pytest.mark.docker` integration tests)

### Installation
```bash
git clone https://github.com/KishanR-dev/rl-code-sandbox.git
cd rl-code-sandbox
uv sync
```

### Running Tests
```bash
# Offline test suite (pure in-memory mock, no Docker required)
uv run pytest -m "not docker" --cov=src

# Full test suite including Docker integration
uv run pytest --cov=src
```

### Code Quality
```bash
# Linter and formatter checks
uv run ruff check .
uv run ruff format --check .

# Static type checking
uv run --python 3.11 mypy --strict src
```

---

## Usage

### CLI
```bash
# Print version information
uv run --python 3.11 rl-sandbox version

# Evaluate a code submission against test assertions
uv run --python 3.11 rl-sandbox run --code "def solve(x): return x * 2" --test "assert solve(3) == 6"

# Run the 30-task benchmark suite
uv run --python 3.11 rl-sandbox benchmark --tasks-dir src/rl_sandbox/benchmarks/tasks
```

### HTTP Service
```bash
# Start the FastAPI service
uv run --python 3.11 uvicorn rl_sandbox.api.app:app --host 0.0.0.0 --port 8000

# Submit an evaluation request
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "submission": {
      "code": "def add(a, b): return a + b",
      "task_id": "task-1",
      "candidate_id": "cand-1"
    },
    "harness": {
      "assertions": "assert add(2, 3) == 5"
    },
    "use_mock": true
  }'
```

---

## Definition of Done
1. All unit and mock tests pass offline with zero network calls and >=90% code coverage.
2. Docker integration tests are marked and skipped gracefully when Docker is unavailable.
3. Strict static typing (`mypy`) and linting (`ruff`) succeed with 0 warnings or errors.
4. Benchmark metrics and PyTorch predictor comparisons reflect actual measured run outputs.
5. No assistant attribution appears in any project files or commits (Rule D-011).
