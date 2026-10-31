# rl-code-sandbox

An async Python 3.11 service and CLI that executes model-generated code in isolated Docker sandboxes with CPU, memory, time, and network constraints, scores each execution using deterministic test suites, and logs evaluation traces. Includes an embedded PyTorch MLP to predict test pass/fail outcomes from code execution features.

Repository: [https://github.com/KishanR-dev/rl-code-sandbox](https://github.com/KishanR-dev/rl-code-sandbox)

---

## Architecture

- **Isolation Provider**: Ephemeral Docker containers configured with Docker's default seccomp profile, zero network egress (`--network none`), read-only root filesystems, unprivileged user execution (`1000:1000`), and hardware resource quotas (`nano_cpus`, `mem_limit`).
- **Offline Mock Provider**: Pure in-memory sandbox simulator for deterministic offline unit testing without external dependencies or subprocesses.
- **Static Guardrails**: AST pre-execution filter intercepting disallowed imports (network, raw sockets, process manipulation).
- **Deterministic Scorer**: Evaluates execution status, assertion pass rates, and runtime constraints into normalized scalar rewards.
- **PyTorch Predictor**: Lightweight 2-layer MLP predicting pass/fail probabilities from static and dynamic execution features, benchmarked against a majority-class baseline.
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
├── PLAN.md
├── CHECKLIST.md
├── README.md
├── pyproject.toml
├── ruff.toml
├── mypy.ini
├── pytest.ini
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py            # FastAPI service exposing POST /evaluate
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── runner.py         # 30-task benchmark runner
│   │   └── tasks.py          # Task definitions (algorithmic + adversarial)
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py           # CLI interface
│   ├── domain/
│   │   ├── __init__.py
│   │   └── models.py         # Pydantic v2 schemas: Task, ExecutionRequest, ExecutionResult, Reward
│   ├── evaluator/
│   │   ├── __init__.py
│   │   └── scorer.py         # Deterministic test and reward computation
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset generation & train/test splits
│   │   ├── features.py       # Code & execution feature extraction
│   │   ├── model.py          # PyTorch MLP classifier
│   │   └── train.py          # Training and baseline evaluation
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract SandboxProvider protocol
│   │   ├── docker.py         # DockerSandboxProvider (default seccomp, limits)
│   │   └── mock.py           # MockSandboxProvider (pure in-memory)
│   └── security/
│       ├── __init__.py
│       └── guardrail.py      # AST static security checks
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_benchmarks.py
    ├── test_guardrails.py
    ├── test_models.py
    ├── test_predictor.py
    ├── test_sandbox_docker.py # Marked @pytest.mark.docker
    ├── test_sandbox_mock.py
    └── test_scorer.py
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
uv run mypy src tests
```

---

## Usage

### CLI
```bash
# Evaluate code against test assertions
uv run rl-sandbox evaluate --code "def solve(x): return x * 2" --test "assert solve(3) == 6"

# Run the 30-task benchmark suite
uv run rl-sandbox benchmark

# Train the PyTorch pass/fail predictor and compare against baseline
uv run rl-sandbox train-predictor
```

### HTTP Service
```bash
# Start the FastAPI service
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Submit an evaluation request
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b): return a + b",
    "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
    "timeout_seconds": 2.0,
    "memory_mb": 64
  }'
```

---

## Definition of Done
1. All unit and mock tests pass offline with zero network calls and >=90% code coverage.
2. Docker integration tests are marked and skipped gracefully when Docker is unavailable.
3. Strict static typing (`mypy`) and linting (`ruff`) succeed with 0 warnings or errors.
4. Benchmark metrics and PyTorch predictor comparisons reflect actual measured run outputs.
5. No assistant attribution appears in any project files or commits (Rule D-011).
