"""Controlled exactly-one QUBOs on both sides of the feasibility threshold."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from src.qaoa_diagnostics import QAOADistributionDiagnostics, summarize_qaoa_distribution
from src.risk_qubo import RiskQUBO, brute_force_risk_qubo


def exact_zero_state_penalty_threshold(risk_scores: Sequence[float]) -> float:
    """Return the exact P where the zero state ties the best one-hot state.

    For nonnegative route risks and ``c.x + P(sum(x)-1)^2``, every multi-hot
    state is more expensive than the best one-hot state. The only competing
    invalid ground state is therefore the all-zero state, whose energy is P.
    """

    scores = np.asarray(tuple(risk_scores), dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("risk_scores must be a nonempty one-dimensional sequence")
    if not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise ValueError("risk_scores must be finite and nonnegative")
    return float(np.min(scores))


def build_unrestricted_penalty_qubo(
    risk_scores: Sequence[float], penalty: float
) -> RiskQUBO:
    """Build the same exactly-one QUBO without enforcing a safe penalty.

    This function is intentionally limited to sensitivity analysis. Production
    callers should use :func:`src.risk_qubo.build_risk_qubo`, which enforces the
    repository's conservative ``P > max(risk)`` sufficient condition.
    """

    scores = tuple(float(value) for value in risk_scores)
    exact_zero_state_penalty_threshold(scores)
    numeric_penalty = float(penalty)
    if not np.isfinite(numeric_penalty) or numeric_penalty <= 0.0:
        raise ValueError("penalty must be finite and positive")
    return RiskQUBO(
        risk_scores=scores,
        penalty=numeric_penalty,
        constant=numeric_penalty,
        linear=tuple(score - numeric_penalty for score in scores),
        quadratic={
            (first, second): 2.0 * numeric_penalty
            for first in range(len(scores))
            for second in range(first + 1, len(scores))
        },
    )


def penalty_ground_state_record(qubo: RiskQUBO) -> dict[str, object]:
    """Enumerate the small QUBO and describe its exact global ground set."""

    enumeration = brute_force_risk_qubo(qubo)
    ground_energy = float(enumeration["qubo_cost"].min())
    ground = enumeration[
        np.isclose(enumeration["qubo_cost"], ground_energy, atol=1e-12, rtol=1e-12)
    ]
    return {
        "global_ground_state_energy": ground_energy,
        "global_ground_state_count": int(len(ground)),
        "global_ground_states_all_feasible": bool(ground["valid"].all()),
        "global_ground_states_any_feasible": bool(ground["valid"].any()),
        "zero_state_is_global_ground_state": bool(
            (ground["bitstring"] == "0" * qubo.number_of_variables).any()
        ),
        "global_ground_bitstrings_json": ground["bitstring"].tolist(),
    }


def summarize_penalty_distribution(
    probabilities: Sequence[float], qubo: RiskQUBO
) -> QAOADistributionDiagnostics:
    """Return ordinary route diagnostics with the true global QUBO ground energy."""

    diagnostics = summarize_qaoa_distribution(probabilities, qubo)
    ground = penalty_ground_state_record(qubo)
    metrics = dict(diagnostics.metrics)
    metrics["feasible_ground_state_energy"] = metrics["ground_state_energy"]
    metrics["ground_state_energy"] = float(ground["global_ground_state_energy"])
    metrics["energy_regret"] = max(
        0.0,
        float(metrics["expected_hamiltonian_energy"])
        - float(ground["global_ground_state_energy"]),
    )
    metrics.update(ground)
    metrics["global_ground_bitstrings_json"] = str(
        ground["global_ground_bitstrings_json"]
    )
    return replace(diagnostics, metrics=metrics)
