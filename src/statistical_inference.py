"""Small, dependency-light utilities for matched scientific inference.

The Step 9.6 analyses use the network instance as the independent sampling
unit.  These helpers deliberately expose that clustering instead of treating
the repeated topology/uncertainty/K/depth rows as independent observations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import norm, rankdata, spearmanr


def percentile_bootstrap_interval(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return estimate and percentile bootstrap interval for one sample."""

    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or len(sample) == 0 or not np.isfinite(sample).all():
        raise ValueError("values must be a nonempty finite one-dimensional sample")
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("confidence must be in (0, 1) and resamples positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        draw = rng.choice(sample, size=len(sample), replace=True)
        estimates[index] = float(statistic(draw))
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(statistic(sample)), float(lower), float(upper)


def cluster_bootstrap_mean_interval(
    values: Sequence[float],
    clusters: Sequence[object],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Bootstrap a row-weighted mean by resampling complete clusters."""

    sample = np.asarray(values, dtype=float)
    labels = np.asarray(clusters, dtype=object)
    if sample.ndim != 1 or labels.ndim != 1 or sample.shape != labels.shape:
        raise ValueError("values and clusters must be aligned one-dimensional arrays")
    if len(sample) == 0 or not np.isfinite(sample).all():
        raise ValueError("values must be nonempty and finite")
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("confidence must be in (0, 1) and resamples positive")
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    members = {label: sample[labels == label] for label in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        estimates[index] = float(
            np.mean(np.concatenate([members[label] for label in chosen]))
        )
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(np.mean(sample)), float(lower), float(upper), int(len(unique))


def paired_sign_flip_test(
    differences: Sequence[float],
    *,
    permutations: int = 100_000,
    seed: int = 0,
) -> float:
    """Two-sided paired randomization p-value for a mean difference.

    Zero differences are retained.  The plus-one correction prevents a zero
    Monte Carlo p-value.  Exact enumeration is used when n <= 18.
    """

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a nonempty finite one-dimensional sample")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    observed = abs(float(np.mean(values)))
    if len(values) <= 18:
        indices = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(len(values), dtype=np.uint64)) & 1
        signs = bits.astype(float) * 2.0 - 1.0
        null = np.abs(np.mean(signs * values, axis=1))
        return float(np.mean(null >= observed - 1e-15))
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    batch = 2_000
    while completed < permutations:
        count = min(batch, permutations - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, len(values)))
        null = np.abs(np.mean(signs * values, axis=1))
        exceedances += int(np.count_nonzero(null >= observed - 1e-15))
        completed += count
    return float((exceedances + 1) / (permutations + 1))


def wilson_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return a binomial proportion and Wilson score confidence interval."""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total and total > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    proportion = successes / total
    z = float(norm.ppf(0.5 + confidence / 2.0))
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(proportion), float(center - half), float(center + half)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return family-wise Holm-adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p_values must be a finite one-dimensional sequence")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * sorted_values
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def spearman_bootstrap_interval(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Return Spearman rho and a paired nonparametric bootstrap interval.

    The paired rows are resampled together. Bootstrap draws that become
    constant in either variable are excluded and the number of defined draws
    is returned so callers can expose that diagnostic.
    """

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x_values and y_values must be aligned one-dimensional arrays")
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman bootstrap requires at least three finite pairs")
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        raise ValueError("Spearman bootstrap requires variation in both variables")
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("confidence must be in (0, 1) and resamples positive")

    estimate = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=float)
    defined = 0
    completed = 0
    batch_size = 512
    while completed < resamples:
        count = min(batch_size, resamples - completed)
        indices = rng.integers(0, len(x), size=(count, len(x)))
        draw_x = x[indices]
        draw_y = y[indices]
        varying = (np.ptp(draw_x, axis=1) > 0.0) & (np.ptp(draw_y, axis=1) > 0.0)
        if np.any(varying):
            ranks_x = rankdata(draw_x[varying], axis=1)
            ranks_y = rankdata(draw_y[varying], axis=1)
            centered_x = ranks_x - ranks_x.mean(axis=1, keepdims=True)
            centered_y = ranks_y - ranks_y.mean(axis=1, keepdims=True)
            denominator = np.sqrt(
                np.sum(centered_x**2, axis=1) * np.sum(centered_y**2, axis=1)
            )
            correlations = np.sum(centered_x * centered_y, axis=1) / denominator
            bootstrap[defined : defined + len(correlations)] = correlations
            defined += len(correlations)
        completed += count
    minimum_defined = max(100, int(0.8 * resamples))
    if defined < minimum_defined:
        raise RuntimeError(
            f"only {defined}/{resamples} Spearman bootstrap draws were defined"
        )
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        bootstrap[:defined], [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    return estimate, float(lower), float(upper), int(defined)
