"""Dataset loader with GroupKFold split for the ML predictor pipeline."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from rl_sandbox.ml.features import FEATURE_DIM, extract_features


def load_jsonl_records(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load all non-empty JSONL records from *path*."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(
    results_path: pathlib.Path,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.object_]]:
    """Build feature matrix, label vector, and group array from a benchmark JSONL file.

    Rows where ``status == "VIOLATION"`` (AST-blocked runs) are excluded because
    they never execute inside the sandbox.

    Returns:
        X: float32 array of shape (n_samples, FEATURE_DIM)
        y: int64 label array of shape (n_samples,); 1 if reward == 1.0 else 0
        groups: object array of task_id strings of shape (n_samples,)
    """
    records = load_jsonl_records(results_path)

    rows_x: list[list[float]] = []
    rows_y: list[int] = []
    rows_g: list[str] = []

    for rec in records:
        if rec.get("status") == "VIOLATION":
            continue

        code: str = rec.get("code", "")
        task_id: str = rec.get("task_id", "")
        reward: float = float(rec.get("reward", 0.0))

        features = extract_features(code)
        rows_x.append(features)
        rows_y.append(1 if reward == 1.0 else 0)
        rows_g.append(task_id)

    if not rows_x:
        X = np.empty((0, FEATURE_DIM), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        groups: NDArray[np.object_] = np.empty((0,), dtype=object)
        return X, y, groups

    X = np.array(rows_x, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int64)
    groups = np.array(rows_g, dtype=object)
    return X, y, groups


def group_kfold_splits(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    groups: NDArray[np.object_],
    n_splits: int = 5,
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Return GroupKFold train/test index pairs.

    Groups (task_ids) never appear in both train and test within a fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y, groups))
