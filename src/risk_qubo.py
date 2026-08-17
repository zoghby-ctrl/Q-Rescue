"""Generalized exactly-one route QUBO for arbitrary risk vectors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pennylane as qml

from src.qubo import qubo_to_ising


@dataclass(frozen=True)
class RiskQUBO:
    """Transparent coefficients for ``c.x + P(sum(x)-1)^2``."""

    risk_scores: tuple[float, ...]
    penalty: float
    constant: float
    linear: tuple[float, ...]
    quadratic: dict[tuple[int, int], float]

    @property
    def number_of_variables(self) -> int:
        return len(self.risk_scores)

    def matrix(self) -> np.ndarray:
        """Return symmetric Q with half of each pair coefficient per triangle."""

        matrix = np.zeros((self.number_of_variables, self.number_of_variables))
        np.fill_diagonal(matrix, self.linear)
        for (first, second), coefficient in self.quadratic.items():
            matrix[first, second] = coefficient / 2.0
            matrix[second, first] = coefficient / 2.0
        return matrix


def _validate_risk_scores(risk_scores: Sequence[float]) -> tuple[float, ...]:
    scores = tuple(float(value) for value in risk_scores)
    if not scores:
        raise ValueError("risk_scores must contain at least one route")
    if not np.isfinite(scores).all():
        raise ValueError("risk_scores must be finite")
    if any(value < 0.0 for value in scores):
        raise ValueError("risk_scores must be non-negative")
    return scores


def choose_safe_penalty(risk_scores: Sequence[float], margin: float = 1.0) -> float:
    """Choose P above every valid route cost, making the zero state worse."""

    scores = _validate_risk_scores(risk_scores)
    margin_value = float(margin)
    if not np.isfinite(margin_value) or margin_value <= 0.0:
        raise ValueError("margin must be finite and positive")
    return max(scores) + margin_value


def build_risk_qubo(
    risk_scores: Sequence[float],
    penalty: float | None = None,
    *,
    penalty_margin: float = 1.0,
) -> RiskQUBO:
    """Expand the exactly-one penalty for any K-route risk vector manually."""

    scores = _validate_risk_scores(risk_scores)
    chosen_penalty = (
        choose_safe_penalty(scores, penalty_margin)
        if penalty is None
        else float(penalty)
    )
    if not np.isfinite(chosen_penalty) or chosen_penalty <= max(scores):
        raise ValueError(
            "penalty must be finite and greater than max(risk_scores); "
            "this guarantees every valid route is cheaper than the zero state"
        )

    linear = tuple(score - chosen_penalty for score in scores)
    quadratic = {
        (first, second): 2.0 * chosen_penalty
        for first in range(len(scores))
        for second in range(first + 1, len(scores))
    }
    return RiskQUBO(
        risk_scores=scores,
        penalty=chosen_penalty,
        constant=chosen_penalty,
        linear=linear,
        quadratic=quadratic,
    )


def _validate_assignment(assignment: Sequence[int], number_of_variables: int) -> tuple[int, ...]:
    bits = tuple(int(bit) for bit in assignment)
    if len(bits) != number_of_variables:
        raise ValueError("assignment length must equal the number of QUBO variables")
    if any(bit not in {0, 1} for bit in bits):
        raise ValueError("assignment values must be binary")
    return bits


def evaluate_risk_qubo(qubo: RiskQUBO, assignment: Sequence[int]) -> float:
    """Evaluate the displayed constant, linear, and pair coefficients."""

    bits = _validate_assignment(assignment, qubo.number_of_variables)
    value = qubo.constant + sum(
        coefficient * bit for coefficient, bit in zip(qubo.linear, bits, strict=True)
    )
    value += sum(
        coefficient * bits[first] * bits[second]
        for (first, second), coefficient in qubo.quadratic.items()
    )
    return float(value)


def brute_force_risk_qubo(qubo: RiskQUBO, *, maximum_variables: int = 20) -> pd.DataFrame:
    """Evaluate every bitstring for tractable instances and return a full table."""

    if qubo.number_of_variables > maximum_variables:
        raise ValueError(
            f"brute force disabled above {maximum_variables} variables; "
            f"received {qubo.number_of_variables}"
        )
    rows: list[dict[str, object]] = []
    for assignment in product((0, 1), repeat=qubo.number_of_variables):
        selected_count = sum(assignment)
        selected_index = assignment.index(1) if selected_count == 1 else None
        rows.append(
            {
                "bitstring": "".join(str(bit) for bit in assignment),
                "selected_count": selected_count,
                "valid": selected_count == 1,
                "selected_route_index": selected_index,
                "objective_cost": sum(
                    score * bit
                    for score, bit in zip(qubo.risk_scores, assignment, strict=True)
                ),
                "constraint_penalty": qubo.penalty * (selected_count - 1) ** 2,
                "qubo_cost": evaluate_risk_qubo(qubo, assignment),
            }
        )
    return pd.DataFrame(rows).sort_values("bitstring").reset_index(drop=True)


def risk_qubo_to_ising(
    qubo: RiskQUBO,
) -> tuple[qml.Hamiltonian, float, np.ndarray, dict[tuple[int, int], float]]:
    """Apply the manual ``x_i=(1-Z_i)/2`` conversion used in Prototype 0.1."""

    return qubo_to_ising(qubo.constant, qubo.linear, qubo.quadratic)


def qubo_scaling_record(qubo: RiskQUBO) -> dict[str, int]:
    """Return exact classical search and algebraic term counts for one K."""

    k = qubo.number_of_variables
    pair_terms = comb(k, 2)
    return {
        "K": k,
        "number_of_qubits": k,
        "search_space_size": 2**k,
        "qubo_constant_terms": 1,
        "qubo_linear_terms": k,
        "qubo_quadratic_terms": pair_terms,
        "qubo_total_terms": 1 + k + pair_terms,
        "hamiltonian_identity_terms": 1,
        "hamiltonian_z_terms": k,
        "hamiltonian_zz_terms": pair_terms,
        "hamiltonian_nonidentity_terms": k + pair_terms,
    }

