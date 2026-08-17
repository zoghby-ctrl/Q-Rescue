"""Tests for manual QUBO coefficients and the Ising conversion."""

import numpy as np
import pennylane as qml

from src.classical_solver import generate_binary_assignments
from src.qubo import (
    build_qubo,
    calculate_constraint_penalty,
    evaluate_qubo,
    qubo_to_ising,
)


RISK_SCORES = [28.0, 23.0, 27.0, 27.0]
PENALTY = 50.0


def test_manual_qubo_coefficients() -> None:
    """The displayed coefficients must match the expanded equation."""

    constant, linear, quadratic, matrix = build_qubo(RISK_SCORES, PENALTY)

    assert constant == 50.0
    np.testing.assert_allclose(linear, [-22.0, -27.0, -23.0, -23.0])
    assert len(quadratic) == 6
    assert set(quadratic.values()) == {100.0}
    np.testing.assert_allclose(np.diag(matrix), linear)
    assert matrix.loc["x_0", "x_1"] == 50.0
    assert matrix.loc["x_1", "x_0"] == 50.0


def test_invalid_assignments_receive_large_penalty() -> None:
    """Every invalid state must cost more than the best valid route."""

    constant, linear, quadratic, _ = build_qubo(RISK_SCORES, PENALTY)
    best_valid_cost = min(RISK_SCORES)

    for assignment in generate_binary_assignments(4):
        if sum(assignment) != 1:
            assert calculate_constraint_penalty(assignment, PENALTY) >= PENALTY
            assert evaluate_qubo(assignment, constant, linear, quadratic) > best_valid_cost


def test_hamiltonian_matches_qubo_up_to_offset() -> None:
    """Every basis-state Hamiltonian energy plus its offset equals QUBO cost."""

    constant, linear, quadratic, _ = build_qubo(RISK_SCORES, PENALTY)
    hamiltonian, offset, _, _ = qubo_to_ising(constant, linear, quadratic)
    hamiltonian_matrix = np.asarray(qml.matrix(hamiltonian, wire_order=range(4)))

    for assignment in generate_binary_assignments(4):
        bitstring = "".join(str(bit) for bit in assignment)
        basis_index = int(bitstring, 2)
        hamiltonian_energy = float(np.real(hamiltonian_matrix[basis_index, basis_index]))
        qubo_cost = evaluate_qubo(assignment, constant, linear, quadratic)
        assert np.isclose(hamiltonian_energy + offset, qubo_cost)

