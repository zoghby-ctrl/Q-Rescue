"""Transparent exhaustive solver for the four-variable QUBO."""

from collections.abc import Sequence
from itertools import product

import pandas as pd

from src.qubo import build_qubo, calculate_constraint_penalty, evaluate_qubo, is_valid_assignment


def generate_binary_assignments(number_of_variables: int = 4) -> list[tuple[int, ...]]:
    """Return every zero-one assignment in ordinary left-to-right order."""

    return list(product([0, 1], repeat=number_of_variables))


def decode_assignment(assignment: Sequence[int], route_names: Sequence[str]) -> str | None:
    """Return the selected route name, or None when the assignment is invalid."""

    if not is_valid_assignment(assignment):
        return None
    selected_index = list(assignment).index(1)
    return str(route_names[selected_index])


def evaluate_all_assignments(
    route_names: Sequence[str],
    risk_scores: Sequence[float],
    penalty: float = 50.0,
) -> pd.DataFrame:
    """Evaluate all assignments and return the complete transparent result table."""

    constant, linear, quadratic, _ = build_qubo(risk_scores, penalty)
    result_rows: list[dict[str, object]] = []

    for assignment in generate_binary_assignments(len(route_names)):
        selected_routes = [
            route_names[index] for index, bit in enumerate(assignment) if bit == 1
        ]
        row: dict[str, object] = {
            "bitstring": "".join(str(bit) for bit in assignment),
            "selected_count": sum(assignment),
            "valid": is_valid_assignment(assignment),
            "risk_term": sum(
                float(risk_scores[index]) * bit for index, bit in enumerate(assignment)
            ),
            "penalty_term": calculate_constraint_penalty(assignment, penalty),
            "qubo_cost": evaluate_qubo(assignment, constant, linear, quadratic),
            "selected_routes": ", ".join(selected_routes) if selected_routes else "None",
        }
        for index, bit in enumerate(assignment):
            row[f"x_{index}"] = bit
        result_rows.append(row)

    column_order = [
        "bitstring",
        *[f"x_{index}" for index in range(len(route_names))],
        "selected_count",
        "valid",
        "risk_term",
        "penalty_term",
        "qubo_cost",
        "selected_routes",
    ]
    return pd.DataFrame(result_rows)[column_order]


def solve_qubo(
    route_names: Sequence[str],
    risk_scores: Sequence[float],
    penalty: float = 50.0,
) -> tuple[pd.DataFrame, tuple[int, ...], str]:
    """Return the full table, lowest-cost assignment, and decoded route."""

    result_table = evaluate_all_assignments(route_names, risk_scores, penalty)
    best_row = result_table.loc[result_table["qubo_cost"].idxmin()]
    best_assignment = tuple(int(best_row[f"x_{index}"]) for index in range(len(route_names)))
    best_route = decode_assignment(best_assignment, route_names)

    if best_route is None:
        raise ValueError("The lowest-cost assignment is invalid; increase the penalty.")

    return result_table, best_assignment, best_route

