import numpy as np
import pytest

from src.statistical_inference import spearman_bootstrap_interval


def test_spearman_bootstrap_preserves_pairing_and_direction() -> None:
    x = np.arange(20, dtype=float)
    y = x**2
    estimate, lower, upper, defined = spearman_bootstrap_interval(
        x, y, resamples=500, seed=12
    )

    assert estimate == pytest.approx(1.0)
    assert lower > 0.99
    assert upper <= 1.0
    assert defined == 500


def test_spearman_bootstrap_rejects_constant_input() -> None:
    with pytest.raises(ValueError, match="variation"):
        spearman_bootstrap_interval([1, 1, 1], [1, 2, 3], resamples=100)
