"""Manual QUBO construction, evaluation, and binary-to-Ising conversion."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pennylane as qml


def build_qubo(
    risk_scores: Sequence[float],
    penalty: float = 50.0,
) -> tuple[float, np.ndarray, dict[tuple[int, int], float], pd.DataFrame]:
    """Return the constant, linear, quadratic, and matrix QUBO coefficients."""

    scores = np.asarray(risk_scores, dtype=float)
    number_of_variables = len(scores)

    constant = float(penalty)
    linear = scores - penalty
    quadratic: dict[tuple[int, int], float] = {}

    for first_index in range(number_of_variables):
        for second_index in range(first_index + 1, number_of_variables):
            quadratic[(first_index, second_index)] = 2.0 * penalty

    matrix_values = np.zeros((number_of_variables, number_of_variables), dtype=float)
    np.fill_diagonal(matrix_values, linear)
    for (first_index, second_index), coefficient in quadratic.items():
        matrix_values[first_index, second_index] = coefficient / 2.0
        matrix_values[second_index, first_index] = coefficient / 2.0

    labels = [f"x_{index}" for index in range(number_of_variables)]
    readable_matrix = pd.DataFrame(matrix_values, index=labels, columns=labels)
    return constant, linear, quadratic, readable_matrix


def evaluate_qubo(
    assignment: Sequence[int],
    constant: float,
    linear: Sequence[float],
    quadratic: dict[tuple[int, int], float],
) -> float:
    """Calculate a QUBO cost directly from its displayed coefficients."""

    bits = np.asarray(assignment, dtype=int)
    cost = float(constant + np.dot(np.asarray(linear, dtype=float), bits))
    for (first_index, second_index), coefficient in quadratic.items():
        cost += coefficient * bits[first_index] * bits[second_index]
    return float(cost)


def calculate_constraint_penalty(
    assignment: Sequence[int],
    penalty: float = 50.0,
) -> float:
    """Return penalty * (number selected - 1)^2 for one assignment."""

    number_selected = sum(int(bit) for bit in assignment)
    return float(penalty * (number_selected - 1) ** 2)


def is_valid_assignment(assignment: Sequence[int]) -> bool:
    """Return True only when exactly one bit is equal to one."""

    return sum(int(bit) for bit in assignment) == 1


def qubo_to_ising(
    constant: float,
    linear: Sequence[float],
    quadratic: dict[tuple[int, int], float],
) -> tuple[qml.Hamiltonian, float, np.ndarray, dict[tuple[int, int], float]]:
    """Convert x=(1-Z)/2 manually and return H, offset, Z, and ZZ terms."""

    linear_array = np.asarray(linear, dtype=float)
    z_coefficients = -linear_array / 2.0
    zz_coefficients: dict[tuple[int, int], float] = {}
    offset = float(constant + np.sum(linear_array) / 2.0)

    for (first_index, second_index), coefficient in quadratic.items():
        offset += coefficient / 4.0
        z_coefficients[first_index] -= coefficient / 4.0
        z_coefficients[second_index] -= coefficient / 4.0
        zz_coefficients[(first_index, second_index)] = coefficient / 4.0

    hamiltonian_coefficients: list[float] = []
    hamiltonian_operations: list[qml.operation.Operator] = []

    for wire, coefficient in enumerate(z_coefficients):
        hamiltonian_coefficients.append(float(coefficient))
        hamiltonian_operations.append(qml.PauliZ(wire))

    for (first_wire, second_wire), coefficient in zz_coefficients.items():
        hamiltonian_coefficients.append(float(coefficient))
        hamiltonian_operations.append(qml.PauliZ(first_wire) @ qml.PauliZ(second_wire))

    hamiltonian = qml.Hamiltonian(hamiltonian_coefficients, hamiltonian_operations)
    return hamiltonian, float(offset), z_coefficients, zz_coefficients


def calculate_ising_energy(
    assignment: Sequence[int],
    offset: float,
    z_coefficients: Sequence[float],
    zz_coefficients: dict[tuple[int, int], float],
) -> float:
    """Evaluate offset + H by replacing each Z with 1 for 0 or -1 for 1."""

    z_values = 1.0 - 2.0 * np.asarray(assignment, dtype=float)
    energy = float(offset + np.dot(np.asarray(z_coefficients), z_values))
    for (first_index, second_index), coefficient in zz_coefficients.items():
        energy += coefficient * z_values[first_index] * z_values[second_index]
    return float(energy)

