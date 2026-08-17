"""Distribution-level diagnostics for exactly-one route-selection QAOA.

The stable :mod:`src.feasibility_qaoa` module exposes the circuit and its basic
valid/optimal probabilities.  This companion module deliberately leaves that
validated implementation untouched and adds the publication diagnostics needed
to distinguish feasibility, encoded optimality, concentration, and route
utility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.feasibility_qaoa import one_hot_basis_indices
from src.risk_qubo import RiskQUBO, evaluate_risk_qubo


@dataclass(frozen=True)
class QAOADistributionDiagnostics:
    """Scalar metrics and the complete feasible-route probability table."""

    metrics: dict[str, object]
    feasible_distribution: pd.DataFrame


def _validated_probabilities(
    probabilities: Sequence[float], number_of_qubits: int
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (2**number_of_qubits,):
        raise ValueError("probability vector length does not match the QUBO")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if np.any(values < -1e-10):
        raise ValueError("probabilities must be nonnegative")
    values = np.clip(values, 0.0, None)
    if not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("probabilities must sum to one")
    return values / values.sum()


def _minimum_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based ascending ranks with tied values sharing minimum rank."""

    return np.asarray(
        [1 + int(np.sum(values < value - 1e-12)) for value in values], dtype=int
    )


def summarize_qaoa_distribution(
    probabilities: Sequence[float],
    qubo: RiskQUBO,
    *,
    optimal_route_indices: Sequence[int] | None = None,
) -> QAOADistributionDiagnostics:
    """Compute feasibility, concentration, energy, and route-regret metrics.

    Entropy and top-k concentration are calculated on the feasible distribution
    conditional on exactly one selected route.  Raw feasible-route probability
    is also retained, so a standard-X circuit with little feasible mass cannot
    appear equivalent to an XY circuit merely because their conditional shapes
    are similar.
    """

    k = qubo.number_of_variables
    values = _validated_probabilities(probabilities, k)
    risks = np.asarray(qubo.risk_scores, dtype=float)
    minimum_risk = float(np.min(risks))
    inferred_optima = np.flatnonzero(
        np.isclose(risks, minimum_risk, rtol=1e-12, atol=1e-12)
    )
    if optimal_route_indices is None:
        optima = inferred_optima
    else:
        optima = np.asarray(tuple(int(index) for index in optimal_route_indices))
        if optima.ndim != 1 or len(optima) == 0:
            raise ValueError("optimal_route_indices must contain at least one index")
        if np.any((optima < 0) | (optima >= k)):
            raise ValueError("optimal_route_indices contains an out-of-range index")
        if set(optima.tolist()) != set(inferred_optima.tolist()):
            raise ValueError("optimal_route_indices does not match the risk-vector argmin")

    feasible_basis = np.asarray(one_hot_basis_indices(k), dtype=int)
    feasible_raw = values[feasible_basis]
    feasible_probability = float(feasible_raw.sum())
    conditional = (
        feasible_raw / feasible_probability
        if feasible_probability > 0.0
        else np.full(k, np.nan, dtype=float)
    )
    optimal_raw = float(feasible_raw[optima].sum())
    optimal_given_feasible = (
        optimal_raw / feasible_probability if feasible_probability > 0.0 else np.nan
    )

    if feasible_probability > 0.0:
        selected_index = int(np.argmax(feasible_raw))
        positive = conditional[conditional > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        normalized_entropy = float(entropy / np.log(k)) if k > 1 else 0.0
        sorted_conditional = np.sort(conditional)[::-1]
        top1_conditional = float(sorted_conditional[0])
        top2_conditional = float(sorted_conditional[: min(2, k)].sum())
        probability_gap = float(
            sorted_conditional[0] - sorted_conditional[1] if k > 1 else sorted_conditional[0]
        )
        feasible_expected_risk = float(np.dot(conditional, risks))
    else:
        selected_index = -1
        entropy = normalized_entropy = np.nan
        top1_conditional = top2_conditional = probability_gap = np.nan
        feasible_expected_risk = np.nan

    energies = np.asarray(
        [
            evaluate_risk_qubo(
                qubo, tuple(int(bit) for bit in format(index, f"0{k}b"))
            )
            for index in range(2**k)
        ],
        dtype=float,
    )
    expected_energy = float(np.dot(values, energies))
    energy_regret = expected_energy - minimum_risk
    if -1e-9 < energy_regret < 0.0:
        energy_regret = 0.0

    risk_ranks = _minimum_ranks(risks)
    selected_risk = float(risks[selected_index]) if selected_index >= 0 else np.nan
    selected_risk_regret = (
        selected_risk - minimum_risk if selected_index >= 0 else np.nan
    )
    if np.isfinite(selected_risk_regret) and -1e-12 < selected_risk_regret < 0.0:
        selected_risk_regret = 0.0
    second_risk = float(np.partition(risks, 1)[1]) if k > 1 else minimum_risk

    distribution = pd.DataFrame(
        {
            "route_index": np.arange(k, dtype=int),
            "route_id": [f"x_{index}" for index in range(k)],
            "bitstring": [format(index, f"0{k}b") for index in feasible_basis],
            "raw_probability": feasible_raw,
            "conditional_feasible_probability": conditional,
            "risk_score": risks,
            "risk_rank": risk_ranks,
            "is_exact_optimum": [index in set(optima.tolist()) for index in range(k)],
            "is_modal_feasible_route": [index == selected_index for index in range(k)],
        }
    )

    metrics: dict[str, object] = {
        "feasibility_probability": feasible_probability,
        "optimal_state_probability": optimal_raw,
        "optimal_probability_given_feasible": float(optimal_given_feasible),
        "expected_hamiltonian_energy": expected_energy,
        "ground_state_energy": minimum_risk,
        "energy_regret": float(energy_regret),
        "selected_route_index": selected_index,
        "selected_route_id": f"x_{selected_index}" if selected_index >= 0 else None,
        "selected_route_risk": selected_risk,
        "risk_regret": float(selected_risk_regret),
        "selected_route_risk_rank": (
            int(risk_ranks[selected_index]) if selected_index >= 0 else None
        ),
        "selected_route_is_optimal": bool(selected_index in set(optima.tolist())),
        "optimal_route_count": int(len(optima)),
        "canonical_optimal_route_index": int(optima[0]),
        "canonical_optimal_route_id": f"x_{int(optima[0])}",
        "feasible_distribution_entropy": entropy,
        "normalized_feasible_entropy": normalized_entropy,
        "effective_feasible_route_count": (
            float(np.exp(entropy)) if np.isfinite(entropy) else np.nan
        ),
        "top1_feasible_probability_raw": (
            float(np.max(feasible_raw)) if feasible_probability > 0.0 else np.nan
        ),
        "top1_feasible_probability_conditional": top1_conditional,
        "top2_feasible_probability_raw": (
            float(np.sort(feasible_raw)[::-1][: min(2, k)].sum())
            if feasible_probability > 0.0
            else np.nan
        ),
        "top2_feasible_probability_conditional": top2_conditional,
        "top1_top2_probability_gap_conditional": probability_gap,
        "conditional_feasible_expected_risk": feasible_expected_risk,
        "conditional_feasible_risk_regret": (
            float(feasible_expected_risk - minimum_risk)
            if np.isfinite(feasible_expected_risk)
            else np.nan
        ),
        "best_second_risk_gap": float(second_risk - minimum_risk),
        "approximation_ratio": np.nan,
        "approximation_ratio_status": (
            "not_reported: minimization QUBO energy includes a feasibility penalty "
            "and offset, so a ratio would be scale-sensitive and misleading"
        ),
    }
    return QAOADistributionDiagnostics(metrics=metrics, feasible_distribution=distribution)

