# DECISIONS.md

# Architecture and Engineering Decisions Log

- **D-001:** Docker container process boundary is the single authoritative security perimeter.
- **D-002:** Container execution runs strictly under `network_mode: "none"`.
- **D-003:** Deterministic v1 reward baseline: 1.0 for passed tests, 0.0 for failed tests/runtime errors.
- **D-004:** Two-tier test architecture: 100% offline unit/API tests via `MockSandboxProvider`; real Docker tests segregated via `@pytest.mark.docker`.
- **D-005:** Full local compatibility with Windows 11 + PowerShell + Docker Desktop (WSL2), CI on `ubuntu-latest`.
- **D-006:** Async Python 3.11 service with FastAPI and CLI with Typer.
- **D-007:** Strict measured-only metrics policy: Zero fabricated or estimated numbers in documentation. Container start-up and execution latencies must be measured empirically.
- **D-008:** Grouped cross-validation: ML evaluation grouped strictly by `task_id` using scikit-learn `GroupKFold` to prevent task leakage.
- **D-009:** Strict static analysis: `ruff` and `mypy --strict` with zero warnings allowed.
- **D-010:** Modular package layout under `src/rl_sandbox/{domain,security,sandbox,evaluator,api,cli,benchmarks,ml}`.
- **D-011:** Zero assistant attribution in all codebase artifacts, documentation, commits, and comments.
- **D-012:** Rename `sandbox/docker.py` to `sandbox/docker_provider.py` to prevent import collision with the upstream `docker` Python SDK.
- **D-013:** Segregated CI pipeline: Fast offline job (<2m target) with >=90% offline coverage gate + separate Ubuntu Docker integration job (<5m target).
- **D-014:** Container hardening profile: Drop all Linux capabilities (`cap_drop=["ALL"]`), enforce `no-new-privileges:true`, read-only root, 64 PID limit, and tmpfs mounted at `/tmp` with `rw,noexec,nosuid,nodev,size=32m`.
- **D-015:** Latency decomposition: Separate container provisioning latency from user code execution latency in execution traces.
- **D-016:** Graceful memory accounting: Attempt `cgroup v2` `memory.peak` retrieval with best-effort fallback flag for heterogeneous hosts.
- **D-017:** Ergonomic evaluation API: Single unified `EvaluationRequest` schema with cross-field Pydantic validation over complex discriminated unions.
- **D-018:** Dependency addition: Add `scikit-learn` for `GroupKFold` splitting, `DummyClassifier` baseline comparison, and evaluation metrics.
- **D-019:** Reward calibration for resource limits: Wall-clock timeouts and out-of-memory events (`OOMKilled=true`) yield `reward = 0.0` (unsolved/failed task), whereas PID limit exhaustion (fork bombs) and AST blocks yield `reward = -1.0` (malicious/violation).
- **D-020:** Network attempt detection heuristic: Under `network_mode: "none"`, the test harness catches `OSError` with `errno == 101` (`ENETUNREACH`) and classifies it as a violation (`reward = -1.0`), documented explicitly as a heuristic.
- **D-021:** Memory limit headrooms: `mem_limit` configured to `256m` and `nano_cpus` to `1000000000` (1.0 CPU) because the Python interpreter and test harness require sufficient headroom; lower values like `128m` risk false OOMs on correct code.
- **D-022:** Precise failure mode separation: Wall-clock timeouts are detected via the harness timer, and OOMs are detected exclusively via container inspect `OOMKilled=true`, avoiding ambiguous conflations from exit code 137 alone.
- **D-023:** Pinned base image digest: Base image `python:3.11-slim` is pinned by digest (`@sha256:...`), resolved at implementation time to guarantee immutable reproducibility.
- **D-024:** Predictor dataset filtration: AST-blocked runs are excluded from the training set for the PyTorch pass/fail predictor because static rejections bypass dynamic execution entirely.
- **D-025:** Workspace AI instructions are placed at `.github/copilot-instructions.md`. This file documents project conventions, toolchain commands, and settled decisions for AI-assisted development sessions. It contains no assistant attribution per D-011.
- **D-026:** `pyproject.toml` is the single configuration file for `ruff`, `mypy`, `pytest`, and `coverage`. No separate `ruff.toml`, `mypy.ini`, or `pytest.ini` files are created. The `[tool.ruff.lint.per-file-ignores]` section suppresses S-rule false positives in `tests/**` (S101 assert, S105/S106 hardcoded passwords in fixtures, S108 temp file paths).
- **D-027:** `mypy_path = "src"` and `explicit_package_bases = true` are set in `[tool.mypy]` so that `mypy --strict src/rl_sandbox` resolves the `src/` layout correctly without requiring `MYPYPATH` environment variable manipulation.
- **D-028:** `[tool.coverage.run]` omits `src/rl_sandbox/sandbox/docker_provider.py` from the offline coverage gate because it requires a live Docker daemon; it is covered exclusively by `@pytest.mark.docker` tests in the separate CI integration job.
- **D-029:** Docker container command configured with idle command `["sleep", "infinity"]` (without `stdin_open`), code delivered via base64 encoded environment variable (`B64_CODE`) to `exec_create` executed via `python -c`, and `user="1000:1000"` explicitly enforced in both `containers.create` and `exec_create` for unprivileged execution.