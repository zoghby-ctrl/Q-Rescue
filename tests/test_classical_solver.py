"""Tests for exhaustive search and quantum-sample decoding."""

import numpy as np

from src.classical_solver import evaluate_all_assignments, solve_qubo
from src.quantum_qaoa import sample_to_bitstring, sample_to_route, summarize_samples


ROUTE_NAMES = ["Route A", "Route B", "Route C", "Route D"]
RISK_SCORES = [28.0, 23.0, 27.0, 27.0]


def test_all_sixteen_assignments_are_evaluated() -> None:
    """Four binary variables must produce exactly 2^4 rows."""

    result_table = evaluate_all_assignments(ROUTE_NAMES, RISK_SCORES, penalty=50.0)
    assert len(result_table) == 16


def test_exactly_four_assignments_are_valid() -> None:
    """Only the four one-hot assignments satisfy exactly one route."""

    result_table = evaluate_all_assignments(ROUTE_NAMES, RISK_SCORES, penalty=50.0)
    assert int(result_table["valid"].sum()) == 4


def test_brute_force_optimum_is_route_b() -> None:
    """The lowest-cost QUBO state must be x = 0100, Route B."""

    _, best_assignment, best_route = solve_qubo(ROUTE_NAMES, RISK_SCORES, penalty=50.0)
    assert best_assignment == (0, 1, 0, 0)
    assert best_route == "Route B"


def test_quantum_sample_order_and_route_decoding() -> None:
    """Wire-order samples must map directly to x_0 through x_3."""

    assert sample_to_bitstring([0, 1, 0, 0]) == "0100"
    assert sample_to_route([1, 0, 0, 0], ROUTE_NAMES) == "Route A"
    assert sample_to_route([0, 1, 0, 0], ROUTE_NAMES) == "Route B"
    assert sample_to_route([0, 0, 0, 0], ROUTE_NAMES) is None
    assert sample_to_route([1, 1, 0, 0], ROUTE_NAMES) is None


def test_summary_reports_measured_most_probable_valid_route() -> None:
    """Reporting follows observed counts and does not assume Route B must lead."""

    samples = np.asarray(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    summary = summarize_samples(samples, ROUTE_NAMES, optimal_route="Route B")

    assert summary["most_frequent_valid_route"] == "Route C"
    assert np.isclose(summary["valid_solution_probability"], 6 / 7)
    assert np.isclose(summary["optimal_route_probability"], 2 / 7)
    assert sum(summary["counts"].values()) == len(samples)

