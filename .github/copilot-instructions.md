# rl-code-sandbox — Copilot Workspace Instructions

## Project Overview

`rl-code-sandbox` is an async Python 3.11 service and CLI that executes model-generated code
inside hardened Docker containers, scores executions with deterministic test harnesses into
scalar RL rewards, and trains a PyTorch MLP pass/fail predictor evaluated against a
scikit-learn baseline.

---

## Conventions

### Language and Runtime
- Python 3.11 strictly; no syntax or features from 3.12+.
- Async-first: all I/O-bound operations use `async`/`await`.
- Type annotations on every function, method, and module-level variable; `mypy --strict` must pass.

### Package Layout (D-010)
```
src/rl_sandbox/{domain,security,sandbox,evaluator,api,cli,benchmarks,ml}
```
- All source lives under `src/rl_sandbox/`.
- Tests live under `tests/{unit,integration,docker}/`.
- Docker-dependent tests are marked `@pytest.mark.docker` and live in `tests/docker/`.

### Code Style
- Line length: 88 characters (`ruff`).
- Imports sorted by `ruff` (isort-compatible, `I` rule set).
- No `eval`, `exec`, `compile`, or `__import__` in production code.
- Comments only where logic needs clarification; no narrative or attribution comments.
- Zero assistant attribution in any file, comment, docstring, or commit message (D-011).

### Commits
- Conventional commits: `feat:`, `fix:`, `test:`, `chore:`, `docs:`, `refactor:`.
- No assistant attribution in commit messages (D-011).
- One logical change per commit.

### Security Perimeter (D-001, D-002, D-014)
- Docker container is the authoritative security boundary.
- `network_mode="none"`, `cap_drop=["ALL"]`, `read_only=True`, `pids_limit=64`.
- `mem_limit="256m"`, `memswap_limit="256m"`, `nano_cpus=1000000000`.
- `/tmp` mounted as `tmpfs` with `rw,noexec,nosuid,nodev,size=32m`.
- AST guard is defense-in-depth only; never the sole security control.

### Reward Values (D-003, D-019, D-020)
| Outcome | Reward |
|---------|--------|
| All assertions pass | `1.0` |
| Assertion failure / runtime error | `0.0` |
| Wall-clock timeout | `0.0` |
| OOM (`OOMKilled=true`) | `0.0` |
| PID exhaustion / fork bomb | `-1.0` |
| Network attempt (`errno 101`) | `-1.0` |
| AST rejection | `-1.0` |

### Testing (D-004, D-013)
- 100% offline unit/API tests via `MockSandboxProvider`; no Docker daemon required.
- Docker tests segregated with `@pytest.mark.docker`.
- Offline coverage gate: ≥90% on `src/rl_sandbox/` (excluding docker provider).
- Never mock the `DockerSandboxProvider` in offline tests; use `MockSandboxProvider`.

### Metrics Policy (D-007)
- Zero fabricated or estimated numbers in documentation or code comments.
- README benchmark tables populated only from executed runs; include commit hash and date.

---

## Toolchain Commands (PowerShell — Windows + uv)

### Install / Sync Dependencies
```powershell
uv sync --all-groups
```

### Lint
```powershell
uv run ruff check .
```

### Format Check
```powershell
uv run ruff format --check .
```

### Type Check
```powershell
uv run mypy --strict src/rl_sandbox
```

### Run Offline Tests (no Docker required)
```powershell
uv run pytest -m "not docker" --cov=rl_sandbox --cov-fail-under=90
```

### Run Docker Integration Tests (requires Docker Desktop)
```powershell
uv run pytest -m "docker"
```

### Run Full Suite
```powershell
uv run pytest --cov=rl_sandbox
```

### All Checks in One Pass
```powershell
uv run ruff check .; uv run ruff format --check .; uv run mypy --strict src/rl_sandbox; uv run pytest -m "not docker" --cov=rl_sandbox --cov-fail-under=90
```

---

## Key Decisions (settled — do not re-open without updating DECISIONS.md)

| ID | Summary |
|----|---------|
| D-001 | Docker container is the authoritative security perimeter |
| D-002 | `network_mode: "none"` always |
| D-004 | Two-tier tests: offline via MockSandboxProvider + docker-marked integration |
| D-005 | Windows 11 + PowerShell + Docker Desktop (WSL2) local; CI on ubuntu-latest |
| D-006 | Async Python 3.11, FastAPI, Typer CLI |
| D-009 | `ruff` + `mypy --strict`, zero warnings |
| D-010 | `src/rl_sandbox/{domain,security,sandbox,evaluator,api,cli,benchmarks,ml}` layout |
| D-011 | Zero assistant attribution anywhere |
| D-012 | `sandbox/docker_provider.py` (not `docker.py`) to avoid SDK import collision |
| D-013 | Segregated CI: fast offline job (<2m) + Docker integration job (<5m) |
| D-023 | Base image pinned by digest: `python:3.11-slim@sha256:...` |
| D-025 | `copilot-instructions.md` at `.github/copilot-instructions.md` |

Full decision log: [DECISIONS.md](../DECISIONS.md)
