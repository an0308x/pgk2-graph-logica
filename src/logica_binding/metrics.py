from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BenchmarkMetrics:
    n: int
    positives: int
    prevalence: float
    auroc: float
    average_precision: float
    random_auroc: float
    random_average_precision: float
    permutation_p_value: float
    score_std: float
    passed: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return self.__dict__.copy()


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int = 2026,
    permutations: int = 10_000,
) -> BenchmarkMetrics:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values")
    if np.unique(labels).size != 2:
        raise ValueError("both positive and negative labels are required")

    rng = np.random.default_rng(seed)
    auroc = float(roc_auc_score(labels, scores))
    average_precision = float(average_precision_score(labels, scores))
    random_scores = rng.random(labels.size)
    random_auroc = float(roc_auc_score(labels, random_scores))
    random_ap = float(average_precision_score(labels, random_scores))

    null_at_least_observed = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        null_at_least_observed += roc_auc_score(permuted, scores) >= auroc
    p_value = float((null_at_least_observed + 1) / (permutations + 1))
    prevalence = float(labels.mean())
    score_std = float(scores.std())
    passed = bool(
        score_std > 0
        and auroc > 0.5
        and average_precision > prevalence
        and p_value < 0.05
    )
    return BenchmarkMetrics(
        n=int(labels.size),
        positives=int(labels.sum()),
        prevalence=prevalence,
        auroc=auroc,
        average_precision=average_precision,
        random_auroc=random_auroc,
        random_average_precision=random_ap,
        permutation_p_value=p_value,
        score_std=score_std,
        passed=passed,
    )


def stratified_bootstrap_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int = 2026,
    replicates: int = 2_000,
    confidence: float = 0.95,
) -> dict[str, list[float]]:
    """Percentile intervals from class-stratified bootstrap resampling."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both classes are required")
    rng = np.random.default_rng(seed)
    auc_values = np.empty(replicates, dtype=np.float64)
    ap_values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sample = np.concatenate(
            [
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            ]
        )
        auc_values[replicate] = roc_auc_score(labels[sample], scores[sample])
        ap_values[replicate] = average_precision_score(labels[sample], scores[sample])
    tail = (1 - confidence) / 2
    quantiles = [tail, 1 - tail]
    return {
        "auroc": [float(value) for value in np.quantile(auc_values, quantiles)],
        "average_precision": [float(value) for value in np.quantile(ap_values, quantiles)],
    }
