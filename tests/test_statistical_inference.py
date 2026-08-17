from __future__ import annotations

import numpy as np

from src.statistical_inference import (
    cluster_bootstrap_mean_interval,
    holm_adjust,
    paired_sign_flip_test,
    percentile_bootstrap_interval,
    wilson_interval,
)


def test_percentile_bootstrap_is_reproducible_and_contains_constant() -> None:
    first = percentile_bootstrap_interval([2.0, 2.0, 2.0], resamples=100, seed=4)
    second = percentile_bootstrap_interval([2.0, 2.0, 2.0], resamples=100, seed=4)
    assert first == second == (2.0, 2.0, 2.0)


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    estimate, lower, upper, cluster_count = cluster_bootstrap_mean_interval(
        [0.0, 0.0, 2.0, 2.0], ["a", "a", "b", "b"], resamples=500, seed=7
    )
    assert estimate == 1.0
    assert lower == 0.0
    assert upper == 2.0
    assert cluster_count == 2


def test_paired_sign_flip_detects_uniform_nonzero_difference() -> None:
    assert paired_sign_flip_test(np.ones(10)) == 2 / 2**10
    assert paired_sign_flip_test(np.zeros(10)) == 1.0


def test_wilson_interval_and_holm_adjustment() -> None:
    estimate, lower, upper = wilson_interval(5, 10)
    assert estimate == 0.5
    assert lower < estimate < upper
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
