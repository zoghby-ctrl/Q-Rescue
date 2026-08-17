"""Network-independent emergency deadline-risk objective.

All terms use minutes or minute-equivalent units.  The weights are
dimensionless policy parameters and must be reported in every experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


AblationVariant = Literal["A", "B", "C", "D", "E"]


@dataclass(frozen=True)
class RiskWeights:
    """Dimensionless weights for uncertainty, deadline, and rerouting terms."""

    lambda_uncertainty: float = 1.0
    lambda_deadline: float = 2.0
    lambda_reroute: float = 1.0

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RouteRisk:
    """Auditable components of the score for one candidate route."""

    expected_travel_time: float
    lower_bound: float
    upper_bound: float
    interval_half_width: float
    deadline_minutes: float
    deadline_margin: float
    deadline_penalty: float
    reroute_penalty: float
    risk_score: float
    ablation_variant: str

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _finite_nonnegative(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def calculate_deadline_margin(deadline_minutes: float, upper_bound: float) -> float:
    """Return ``deadline - conservative upper travel-time bound`` in minutes."""

    deadline = _finite_nonnegative(deadline_minutes, "deadline_minutes")
    upper = _finite_nonnegative(upper_bound, "upper_bound")
    return deadline - upper


def calculate_route_risk(
    expected_travel_time: float,
    lower_bound: float,
    upper_bound: float,
    deadline_minutes: float,
    *,
    reroute_penalty_minutes: float = 0.0,
    weights: RiskWeights | None = None,
    variant: AblationVariant = "E",
) -> RouteRisk:
    """Calculate one transparent route score and all of its components.

    The uncertainty term is the conformal interval half-width.  The deadline
    penalty is the positive exceedance of the upper bound beyond the deadline;
    it is zero for routes whose conservative bound meets the deadline.  The
    reroute term is an externally specified minute-equivalent switching cost.

    Variants follow the research ablation: A expected time; B expected plus
    uncertainty; C expected plus deadline exceedance; D both risk terms; E the
    full dynamic score including a rerouting penalty.
    """

    if variant not in {"A", "B", "C", "D", "E"}:
        raise ValueError("variant must be one of A, B, C, D, or E")
    expected = _finite_nonnegative(expected_travel_time, "expected_travel_time")
    lower = _finite_nonnegative(lower_bound, "lower_bound")
    upper = _finite_nonnegative(upper_bound, "upper_bound")
    deadline = _finite_nonnegative(deadline_minutes, "deadline_minutes")
    reroute = _finite_nonnegative(reroute_penalty_minutes, "reroute_penalty_minutes")
    if lower > expected or expected > upper:
        raise ValueError("bounds must satisfy lower_bound <= prediction <= upper_bound")

    chosen_weights = weights or RiskWeights()
    half_width = (upper - lower) / 2.0
    margin = calculate_deadline_margin(deadline, upper)
    exceedance = max(0.0, -margin)

    include_uncertainty = variant in {"B", "D", "E"}
    include_deadline = variant in {"C", "D", "E"}
    include_reroute = variant == "E"
    risk_score = expected
    if include_uncertainty:
        risk_score += chosen_weights.lambda_uncertainty * half_width
    if include_deadline:
        risk_score += chosen_weights.lambda_deadline * exceedance
    if include_reroute:
        risk_score += chosen_weights.lambda_reroute * reroute

    return RouteRisk(
        expected_travel_time=expected,
        lower_bound=lower,
        upper_bound=upper,
        interval_half_width=half_width,
        deadline_minutes=deadline,
        deadline_margin=margin,
        deadline_penalty=exceedance,
        reroute_penalty=reroute,
        risk_score=float(risk_score),
        ablation_variant=variant,
    )


def score_route_table(
    route_table: pd.DataFrame,
    deadline_minutes: float,
    *,
    weights: RiskWeights | None = None,
    variant: AblationVariant = "E",
) -> pd.DataFrame:
    """Add risk components to a candidate-route table without mutating it."""

    required = {"prediction", "lower_bound", "upper_bound"}
    missing = sorted(required.difference(route_table.columns))
    if missing:
        raise ValueError(f"route_table is missing required columns: {missing}")

    scored_rows: list[dict[str, object]] = []
    for _, row in route_table.iterrows():
        reroute = float(row.get("reroute_penalty_minutes", 0.0))
        result = calculate_route_risk(
            expected_travel_time=float(row["prediction"]),
            lower_bound=float(row["lower_bound"]),
            upper_bound=float(row["upper_bound"]),
            deadline_minutes=deadline_minutes,
            reroute_penalty_minutes=reroute,
            weights=weights,
            variant=variant,
        )
        scored_rows.append({**row.to_dict(), **result.as_dict()})
    return pd.DataFrame(scored_rows)

