"""Offline integration test for the ML pipeline (Phase 7)."""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import torch

from rl_sandbox.ml.dataset import build_dataset, group_kfold_splits
from rl_sandbox.ml.features import FEATURE_DIM, FEATURE_NAMES, extract_features
from rl_sandbox.ml.predictor import (
    CVResult,
    FoldMetrics,
    PassFailMLP,
    evaluate_baseline,
    evaluate_mlp,
    run_cross_validation,
    train_mlp,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jsonl(records: list[dict[object, object]], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _synthetic_records(
    n_tasks: int = 6, candidates_per_task: int = 10
) -> list[dict[str, object]]:
    """Generate synthetic benchmark JSONL records with varied code and rewards."""
    records = []
    for t in range(n_tasks):
        task_id = f"task-{t:02d}"
        for c in range(candidates_per_task):
            # Alternate pass/fail to ensure both classes are present
            reward = 1.0 if c % 2 == 0 else 0.0
            code = (
                f"def fn_{t}_{c}(x):\n"
                f"    if x > {c}:\n"
                f"        return x * {c + 1}\n"
                f"    return x\n"
            )
            records.append(
                {
                    "task_id": task_id,
                    "candidate_id": f"cand-{c:02d}",
                    "code": code,
                    "status": "PASSED" if reward == 1.0 else "FAILED",
                    "reward": reward,
                }
            )
        # Add one VIOLATION record per task — must be filtered out
        records.append(
            {
                "task_id": task_id,
                "candidate_id": "cand-viol",
                "code": "import os",
                "status": "VIOLATION",
                "reward": -1.0,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Feature extractor tests (7.1)
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_returns_correct_length(self) -> None:
        feats = extract_features("x = 1")
        assert len(feats) == FEATURE_DIM

    def test_feature_names_length(self) -> None:
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_all_floats(self) -> None:
        feats = extract_features("def f(x):\n    return x + 1\n")
        assert all(isinstance(v, float) for v in feats)

    def test_syntax_error_returns_zeros(self) -> None:
        feats = extract_features("def f(:")
        assert feats == [0.0] * FEATURE_DIM

    def test_depth_increases_with_nesting(self) -> None:
        shallow = extract_features("x = 1")[0]
        nested = (
            "def f():\n    if True:\n        for i in range(3):\n            pass\n"
        )
        deep = extract_features(nested)[0]
        assert deep > shallow

    def test_cyclomatic_complexity_increases_with_branches(self) -> None:
        simple = extract_features("x = 1")[2]
        branchy_code = (
            "def f(x):\n"
            "    if x > 0:\n"
            "        return x\n"
            "    elif x < 0:\n"
            "        return -x\n"
            "    return 0\n"
        )
        branchy = extract_features(branchy_code)[2]
        assert branchy > simple

    def test_exception_density_nonzero_with_handler(self) -> None:
        code = "try:\n    pass\nexcept ValueError:\n    pass\n"
        feats = extract_features(code)
        assert feats[4] > 0.0

    def test_token_length_log1p_positive(self) -> None:
        feats = extract_features("x = 1 + 2")
        assert feats[5] > 0.0

    def test_empty_string(self) -> None:
        feats = extract_features("")
        assert len(feats) == FEATURE_DIM
        assert all(isinstance(v, float) for v in feats)


def _tmp_jsonl(records: list[dict[str, object]]) -> pathlib.Path:
    """Write *records* to a temp JSONL file and return its path."""
    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", mode="w", delete=False, encoding="utf-8"
    ) as f:
        tmp = pathlib.Path(f.name)
    _make_jsonl(records, tmp)
    return tmp


class TestBuildDataset:
    def test_shapes(self) -> None:
        records = _synthetic_records(n_tasks=4, candidates_per_task=8)
        tmp = _tmp_jsonl(records)
        X, y, groups = build_dataset(tmp)
        tmp.unlink()

        # 4 tasks * 8 candidates = 32 rows (VIOLATION rows excluded)
        assert X.shape == (32, FEATURE_DIM)
        assert y.shape == (32,)
        assert groups.shape == (32,)

    def test_violation_rows_excluded(self) -> None:
        records: list[dict[str, object]] = [
            {
                "task_id": "t0",
                "candidate_id": "c0",
                "code": "x=1",
                "status": "PASSED",
                "reward": 1.0,
            },
            {
                "task_id": "t0",
                "candidate_id": "c1",
                "code": "import os",
                "status": "VIOLATION",
                "reward": -1.0,
            },
        ]
        tmp = _tmp_jsonl(records)
        X, y, groups = build_dataset(tmp)
        tmp.unlink()

        assert X.shape[0] == 1
        assert y[0] == 1

    def test_labels_binary(self) -> None:
        records = _synthetic_records(n_tasks=2, candidates_per_task=4)
        tmp = _tmp_jsonl(records)
        X, y, groups = build_dataset(tmp)
        tmp.unlink()

        assert set(y.tolist()).issubset({0, 1})

    def test_dtypes(self) -> None:
        records = _synthetic_records(n_tasks=2, candidates_per_task=4)
        tmp = _tmp_jsonl(records)
        X, y, groups = build_dataset(tmp)
        tmp.unlink()

        assert X.dtype == np.float32
        assert y.dtype == np.int64

    def test_empty_file_returns_empty_arrays(self) -> None:
        tmp = _tmp_jsonl([])
        X, y, groups = build_dataset(tmp)
        tmp.unlink()

        assert X.shape == (0, FEATURE_DIM)
        assert y.shape == (0,)

    def test_group_kfold_no_task_overlap(self) -> None:
        records = _synthetic_records(n_tasks=6, candidates_per_task=10)
        tmp = _tmp_jsonl(records)
        X, y, groups = build_dataset(tmp)
        tmp.unlink()

        splits = group_kfold_splits(X, y, groups, n_splits=3)
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            train_tasks = set(groups[train_idx])
            test_tasks = set(groups[test_idx])
            assert train_tasks.isdisjoint(test_tasks)


# ---------------------------------------------------------------------------
# MLP model tests (7.3)
# ---------------------------------------------------------------------------


class TestPassFailMLP:
    def test_output_shape_single(self) -> None:
        model = PassFailMLP(input_dim=FEATURE_DIM)
        x = torch.zeros(1, FEATURE_DIM)
        out = model(x)
        assert out.shape == (1,)

    def test_output_shape_batch(self) -> None:
        model = PassFailMLP(input_dim=FEATURE_DIM)
        x = torch.zeros(16, FEATURE_DIM)
        out = model(x)
        assert out.shape == (16,)

    def test_output_in_unit_interval(self) -> None:
        model = PassFailMLP(input_dim=FEATURE_DIM)
        x = torch.randn(32, FEATURE_DIM)
        out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_train_mlp_returns_model(self) -> None:
        X = np.random.rand(20, FEATURE_DIM).astype(np.float32)
        y = np.array([0, 1] * 10, dtype=np.int64)
        model = train_mlp(X, y, epochs=5)
        assert isinstance(model, PassFailMLP)

    def test_train_mlp_eval_mode_after_evaluate(self) -> None:
        X = np.random.rand(20, FEATURE_DIM).astype(np.float32)
        y = np.array([0, 1] * 10, dtype=np.int64)
        model = train_mlp(X, y, epochs=5)
        evaluate_mlp(model, X, y)
        assert not model.training


# ---------------------------------------------------------------------------
# Training loop and baseline tests (7.4)
# ---------------------------------------------------------------------------


class TestTrainingAndBaseline:
    def _make_data(
        self, n_tasks: int = 6, candidates: int = 10
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int], np.dtype[np.int64]],
        np.ndarray[tuple[int], np.dtype[np.object_]],
    ]:
        records = _synthetic_records(n_tasks=n_tasks, candidates_per_task=candidates)
        tmp = _tmp_jsonl(records)
        X, y, groups = build_dataset(tmp)
        tmp.unlink()
        return X, y, groups

    def test_evaluate_mlp_returns_fold_metrics(self) -> None:
        X, y, _ = self._make_data()
        model = train_mlp(X, y, epochs=5)
        metrics = evaluate_mlp(model, X, y)
        assert isinstance(metrics, FoldMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.balanced_accuracy <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0

    def test_evaluate_baseline_returns_fold_metrics(self) -> None:
        X, y, _ = self._make_data()
        metrics = evaluate_baseline(X, y, X, y)
        assert isinstance(metrics, FoldMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_run_cross_validation_returns_cv_result(self) -> None:
        X, y, groups = self._make_data(n_tasks=6, candidates=10)
        splits = group_kfold_splits(X, y, groups, n_splits=3)
        result = run_cross_validation(X, y, splits, epochs=5)
        assert isinstance(result, CVResult)
        assert len(result.mlp_folds) == 3
        assert len(result.baseline_folds) == 3

    def test_cv_mean_metrics_in_range(self) -> None:
        X, y, groups = self._make_data(n_tasks=6, candidates=10)
        splits = group_kfold_splits(X, y, groups, n_splits=3)
        result = run_cross_validation(X, y, splits, epochs=5)
        mean_mlp = result.mlp_mean()
        mean_base = result.baseline_mean()
        for m in (mean_mlp, mean_base):
            assert 0.0 <= m.accuracy <= 1.0
            assert 0.0 <= m.balanced_accuracy <= 1.0
            assert 0.0 <= m.roc_auc <= 1.0
            assert 0.0 <= m.brier_score <= 1.0

    def test_no_task_overlap_in_splits(self) -> None:
        X, y, groups = self._make_data(n_tasks=6, candidates=10)
        splits = group_kfold_splits(X, y, groups, n_splits=3)
        for train_idx, test_idx in splits:
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
