"""A limited ACI-style adaptive conformal state for sequential evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.conformal import finite_sample_residual_quantile


@dataclass
class AdaptiveMiscoverageConformal:
    """Adapt the requested miscoverage level after each observed outcome.

    The conformity-score reference sample stays frozen; only the requested
    quantile level changes according to the Adaptive Conformal Inference update

        alpha_(t+1) = alpha_t + gamma * (target_alpha - error_t).

    The caller must request ``current_quantile`` before calling ``observe`` so
    the current label cannot leak into its own interval.  This deliberately
    limited baseline does not silently retrain the point or scale models.
    """

    calibration_scores: np.ndarray
    target_alpha: float = 0.10
    gamma: float = 0.01
    minimum_alpha: float = 0.001
    maximum_alpha: float = 0.50
    alpha_t: float = field(init=False)
    time_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        scores = np.asarray(self.calibration_scores, dtype=float)
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError("calibration_scores must be a nonempty 1D array")
        if not np.isfinite(scores).all() or np.any(scores < 0.0):
            raise ValueError("calibration_scores must be finite and nonnegative")
        parameters = np.asarray(
            [self.target_alpha, self.gamma, self.minimum_alpha, self.maximum_alpha],
            dtype=float,
        )
        if not np.isfinite(parameters).all() or self.gamma <= 0.0:
            raise ValueError("adaptive parameters must be finite and gamma positive")
        if not 0.0 < self.minimum_alpha <= self.target_alpha < self.maximum_alpha < 1.0:
            raise ValueError(
                "require 0 < minimum_alpha <= target_alpha < maximum_alpha < 1"
            )
        self.calibration_scores = scores.copy()
        self.alpha_t = float(self.target_alpha)

    def current_quantile(self) -> float:
        """Return the finite-sample score quantile for the pre-outcome interval."""

        return finite_sample_residual_quantile(
            self.calibration_scores, float(self.alpha_t)
        )

    def observe(self, score: float, *, quantile_used: float) -> dict[str, Any]:
        """Update after observing one score and return an auditable trace row."""

        numeric_score = float(score)
        numeric_quantile = float(quantile_used)
        if (
            not np.isfinite(numeric_score)
            or numeric_score < 0.0
            or not np.isfinite(numeric_quantile)
            or numeric_quantile < 0.0
        ):
            raise ValueError("score and quantile_used must be finite and nonnegative")
        alpha_before = float(self.alpha_t)
        error = int(numeric_score > numeric_quantile)
        updated = alpha_before + float(self.gamma) * (
            float(self.target_alpha) - error
        )
        self.alpha_t = float(
            np.clip(updated, float(self.minimum_alpha), float(self.maximum_alpha))
        )
        trace = {
            "time_index": int(self.time_index),
            "score": numeric_score,
            "quantile_used": numeric_quantile,
            "error": error,
            "covered": not bool(error),
            "alpha_before": alpha_before,
            "alpha_after": float(self.alpha_t),
            "target_alpha": float(self.target_alpha),
            "gamma": float(self.gamma),
        }
        self.time_index += 1
        return trace
