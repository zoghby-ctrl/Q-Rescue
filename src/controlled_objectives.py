"""Explicit objective ablations that do not confound deadline and uncertainty."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def controlled_objective_table(
    predictions: Sequence[float],
    uncertainty_half_widths: Sequence[float],
    *,
    deadline_minutes: float,
    lambda_deadline: float = 2.0,
    lambda_uncertainty: float = 1.0,
    reroute_penalties: Sequence[float] | None = None,
    lambda_reroute: float = 1.0,
) -> pd.DataFrame:
    """Return A/B/C and, when applicable, D scores for one candidate set.

    A is expected time only.  B adds a point-prediction deadline exceedance.
    C adds calibrated uncertainty while preserving B's exact deadline term.
    D additionally adds an explicit rerouting cost and is emitted only when a
    dynamic state supplies ``reroute_penalties``.

    This differs intentionally from the older conservative upper-bound proxy:
    using ``upper_bound - deadline`` inside B would change the deadline term
    whenever uncertainty changes and would invalidate the B-versus-C isolation.
    """

    predicted = np.asarray(predictions, dtype=float)
    widths = np.asarray(uncertainty_half_widths, dtype=float)
    if predicted.ndim != 1 or len(predicted) == 0:
        raise ValueError("predictions must be a nonempty one-dimensional sequence")
    if widths.shape != predicted.shape:
        raise ValueError("uncertainty_half_widths must match predictions")
    deadline = float(deadline_minutes)
    weights = np.asarray(
        [lambda_deadline, lambda_uncertainty, lambda_reroute], dtype=float
    )
    if (
        not np.isfinite(predicted).all()
        or not np.isfinite(widths).all()
        or not np.isfinite(deadline)
        or not np.isfinite(weights).all()
    ):
        raise ValueError("objective inputs and weights must be finite")
    if np.any(predicted < 0.0) or np.any(widths < 0.0) or np.any(weights < 0.0):
        raise ValueError("predictions, widths, and weights must be nonnegative")

    point_deadline_penalty = np.maximum(0.0, predicted - deadline)
    score_a = predicted
    score_b = score_a + float(lambda_deadline) * point_deadline_penalty
    score_c = score_b + float(lambda_uncertainty) * widths
    data: dict[str, object] = {
        "route_index": np.arange(len(predicted), dtype=int),
        "route_id": [f"x_{index}" for index in range(len(predicted))],
        "prediction_minutes": predicted,
        "uncertainty_half_width_minutes": widths,
        "deadline_minutes": deadline,
        "point_deadline_exceedance_minutes": point_deadline_penalty,
        "A_expected_time": score_a,
        "B_expected_plus_deadline": score_b,
        "C_expected_plus_deadline_plus_uncertainty": score_c,
    }
    if reroute_penalties is not None:
        reroute = np.asarray(reroute_penalties, dtype=float)
        if reroute.shape != predicted.shape:
            raise ValueError("reroute_penalties must match predictions")
        if not np.isfinite(reroute).all() or np.any(reroute < 0.0):
            raise ValueError("reroute_penalties must be finite and nonnegative")
        data["reroute_penalty_minutes"] = reroute
        data["D_full_dynamic"] = score_c + float(lambda_reroute) * reroute
    return pd.DataFrame(data)

