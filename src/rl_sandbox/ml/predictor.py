"""Lightweight PyTorch MLP pass/fail predictor with scikit-learn baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from rl_sandbox.ml.features import FEATURE_DIM


class PassFailMLP(nn.Module):
    """Two-hidden-layer MLP binary classifier.

    Architecture: Input -> Linear(32) -> ReLU -> Dropout(0.2)
                        -> Linear(16) -> ReLU -> Linear(1) -> Sigmoid
    """

    def __init__(self, input_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.net(x).squeeze(-1)
        return res


@dataclass
class FoldMetrics:
    """Evaluation metrics for a single cross-validation fold."""

    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    brier_score: float


@dataclass
class CVResult:
    """Aggregated cross-validation results for MLP and baseline."""

    mlp_folds: list[FoldMetrics] = field(default_factory=list)
    baseline_folds: list[FoldMetrics] = field(default_factory=list)

    def mlp_mean(self) -> FoldMetrics:
        return _mean_metrics(self.mlp_folds)

    def baseline_mean(self) -> FoldMetrics:
        return _mean_metrics(self.baseline_folds)


def _mean_metrics(folds: list[FoldMetrics]) -> FoldMetrics:
    if not folds:
        return FoldMetrics(0.0, 0.0, 0.0, 0.0)
    return FoldMetrics(
        accuracy=float(np.mean([f.accuracy for f in folds])),
        balanced_accuracy=float(np.mean([f.balanced_accuracy for f in folds])),
        roc_auc=float(np.mean([f.roc_auc for f in folds])),
        brier_score=float(np.mean([f.brier_score for f in folds])),
    )


def _compute_metrics(
    y_true: NDArray[np.int64], y_prob: NDArray[np.float64]
) -> FoldMetrics:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(np.mean(y_true == y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    return FoldMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        roc_auc=auc,
        brier_score=brier,
    )


def train_mlp(
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> PassFailMLP:
    """Train a PassFailMLP on the given training split."""
    model = PassFailMLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

    return model


def evaluate_mlp(
    model: PassFailMLP,
    X_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
) -> FoldMetrics:
    """Evaluate a trained PassFailMLP on a test split."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        probs: NDArray[np.float64] = model(X_t).numpy().astype(np.float64)
    return _compute_metrics(y_test, probs)


def evaluate_baseline(
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
) -> FoldMetrics:
    """Evaluate a DummyClassifier(strategy='most_frequent') baseline."""
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    probs: NDArray[np.float64] = clf.predict_proba(X_test)[:, 1].astype(np.float64)
    return _compute_metrics(y_test, probs)


def run_cross_validation(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    splits: list[tuple[NDArray[np.intp], NDArray[np.intp]]],
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> CVResult:
    """Run GroupKFold cross-validation for MLP and DummyClassifier baseline."""
    result = CVResult()
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = train_mlp(
            X_train, y_train, epochs=epochs, lr=lr, weight_decay=weight_decay
        )
        result.mlp_folds.append(evaluate_mlp(model, X_test, y_test))
        result.baseline_folds.append(
            evaluate_baseline(X_train, y_train, X_test, y_test)
        )

    return result
