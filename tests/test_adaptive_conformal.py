from __future__ import annotations

import numpy as np

from src.adaptive_conformal import AdaptiveMiscoverageConformal


def test_miss_decreases_alpha_and_covered_point_increases_it() -> None:
    adaptive = AdaptiveMiscoverageConformal(
        np.arange(1.0, 11.0), target_alpha=0.10, gamma=0.01
    )
    quantile = adaptive.current_quantile()
    miss = adaptive.observe(quantile + 1.0, quantile_used=quantile)
    assert miss["error"] == 1
    assert np.isclose(adaptive.alpha_t, 0.091)

    quantile = adaptive.current_quantile()
    covered = adaptive.observe(0.0, quantile_used=quantile)
    assert covered["error"] == 0
    assert np.isclose(adaptive.alpha_t, 0.092)


def test_update_clips_at_declared_bounds_and_preserves_scores() -> None:
    scores = np.array([1.0, 2.0, 3.0])
    adaptive = AdaptiveMiscoverageConformal(
        scores,
        target_alpha=0.10,
        gamma=0.20,
        minimum_alpha=0.01,
        maximum_alpha=0.20,
    )
    quantile = adaptive.current_quantile()
    adaptive.observe(100.0, quantile_used=quantile)
    assert adaptive.alpha_t == 0.01
    np.testing.assert_array_equal(adaptive.calibration_scores, scores)


def test_invalid_scores_are_rejected() -> None:
    try:
        AdaptiveMiscoverageConformal(np.array([1.0, np.nan]))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite calibration score was accepted")
