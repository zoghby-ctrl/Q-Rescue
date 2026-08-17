"""Generalized standard and one-hot feasibility-preserving PennyLane QAOA."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json

import networkx as nx
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

from src.risk_qubo import RiskQUBO, evaluate_risk_qubo


QAOAMethod = str
SUPPORTED_METHODS = ("standard_x", "feasible_xy")


@dataclass(frozen=True)
class OptimizedAngles:
    gammas: pnp.ndarray
    betas: pnp.ndarray
    history: pd.DataFrame


def one_hot_basis_indices(number_of_qubits: int) -> tuple[int, ...]:
    """Return statevector indices for x_0...x_(K-1) one-hot bitstrings."""

    if number_of_qubits <= 0:
        raise ValueError("number_of_qubits must be positive")
    return tuple(1 << (number_of_qubits - 1 - wire) for wire in range(number_of_qubits))


def one_hot_w_state(number_of_qubits: int) -> np.ndarray:
    """Return a normalized uniform superposition over exactly-one states."""

    state = np.zeros(2**number_of_qubits, dtype=complex)
    state[list(one_hot_basis_indices(number_of_qubits))] = 1.0 / np.sqrt(number_of_qubits)
    return state


def create_mixer_hamiltonian(
    number_of_qubits: int,
    method: QAOAMethod,
) -> qml.Hamiltonian:
    """Create either the standard X mixer or a ring-graph XY mixer."""

    if method == "standard_x":
        return qml.Hamiltonian(
            [1.0] * number_of_qubits,
            [qml.PauliX(wire) for wire in range(number_of_qubits)],
        )
    if method == "feasible_xy":
        if number_of_qubits < 2:
            raise ValueError("feasible_xy requires at least two routes")
        return qml.qaoa.xy_mixer(nx.cycle_graph(number_of_qubits))
    raise ValueError(f"method must be one of {SUPPORTED_METHODS}")


def _prepare_initial_state(number_of_qubits: int, method: QAOAMethod) -> None:
    if method == "standard_x":
        for wire in range(number_of_qubits):
            qml.Hadamard(wires=wire)
    elif method == "feasible_xy":
        qml.StatePrep(one_hot_w_state(number_of_qubits), wires=range(number_of_qubits))
    else:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")


def _apply_qaoa_layers(
    gammas: Sequence[float],
    betas: Sequence[float],
    cost_hamiltonian: qml.Hamiltonian,
    mixer_hamiltonian: qml.Hamiltonian,
) -> None:
    for gamma, beta in zip(gammas, betas, strict=True):
        qml.qaoa.cost_layer(gamma, cost_hamiltonian)
        qml.qaoa.mixer_layer(beta, mixer_hamiltonian)


def create_expectation_qnode(
    cost_hamiltonian: qml.Hamiltonian,
    *,
    number_of_qubits: int,
    depth: int,
    method: QAOAMethod,
) -> qml.QNode:
    """Create an analytic differentiable cost-expectation circuit."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    mixer = create_mixer_hamiltonian(number_of_qubits, method)
    device = qml.device("default.qubit", wires=number_of_qubits)

    @qml.qnode(device, interface="autograd")
    def circuit(gammas: pnp.ndarray, betas: pnp.ndarray) -> pnp.ndarray:
        _prepare_initial_state(number_of_qubits, method)
        _apply_qaoa_layers(gammas, betas, cost_hamiltonian, mixer)
        return qml.expval(cost_hamiltonian)

    return circuit


def create_probability_qnode(
    cost_hamiltonian: qml.Hamiltonian,
    *,
    number_of_qubits: int,
    depth: int,
    method: QAOAMethod,
) -> qml.QNode:
    """Create an analytic circuit returning probabilities in wire order."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    mixer = create_mixer_hamiltonian(number_of_qubits, method)
    device = qml.device("default.qubit", wires=number_of_qubits)

    @qml.qnode(device, interface="autograd")
    def circuit(gammas: pnp.ndarray, betas: pnp.ndarray) -> pnp.ndarray:
        _prepare_initial_state(number_of_qubits, method)
        _apply_qaoa_layers(gammas, betas, cost_hamiltonian, mixer)
        return qml.probs(wires=range(number_of_qubits))

    return circuit


def optimize_qaoa_angles(
    expectation_qnode: Callable[[pnp.ndarray, pnp.ndarray], pnp.ndarray],
    *,
    depth: int,
    seed: int,
    iterations: int = 60,
    step_size: float = 0.05,
) -> OptimizedAngles:
    """Optimize one seeded angle initialization with Adam."""

    if iterations <= 0 or step_size <= 0.0:
        raise ValueError("iterations and step_size must be positive")
    rng = np.random.default_rng(seed)
    gammas = pnp.array(rng.uniform(-0.15, 0.15, depth), requires_grad=True)
    betas = pnp.array(rng.uniform(-0.30, 0.30, depth), requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=step_size)
    rows = [
        {
            "iteration": 0,
            "hamiltonian_expectation": float(expectation_qnode(gammas, betas)),
        }
    ]
    for iteration in range(1, iterations + 1):
        gammas, betas = optimizer.step(expectation_qnode, gammas, betas)
        rows.append(
            {
                "iteration": iteration,
                "hamiltonian_expectation": float(expectation_qnode(gammas, betas)),
            }
        )
    return OptimizedAngles(gammas, betas, pd.DataFrame(rows))


def summarize_probabilities(
    probabilities: Sequence[float],
    qubo: RiskQUBO,
    *,
    optimal_route_index: int,
) -> dict[str, object]:
    """Compute exact probability, validity, and energy metrics."""

    values = np.asarray(probabilities, dtype=float)
    if values.shape != (2**qubo.number_of_variables,):
        raise ValueError("probability vector length does not match the QUBO")
    if not np.isclose(values.sum(), 1.0, atol=1e-8) or np.any(values < -1e-10):
        raise ValueError("probabilities must be nonnegative and sum to one")
    valid_indices = one_hot_basis_indices(qubo.number_of_variables)
    if not 0 <= optimal_route_index < qubo.number_of_variables:
        raise ValueError("optimal_route_index is outside the route vector")
    optimal_basis_index = valid_indices[optimal_route_index]
    most_probable_route_index = int(
        np.argmax([values[index] for index in valid_indices])
    )
    energies = np.asarray(
        [
            evaluate_risk_qubo(
                qubo, tuple(int(bit) for bit in format(index, f"0{qubo.number_of_variables}b"))
            )
            for index in range(2**qubo.number_of_variables)
        ]
    )
    return {
        "expected_qubo_energy": float(np.dot(values, energies)),
        "valid_probability": float(values[list(valid_indices)].sum()),
        "optimal_probability": float(values[optimal_basis_index]),
        "most_probable_valid_route_index": most_probable_route_index,
        "most_probable_valid_bitstring": format(
            valid_indices[most_probable_route_index], f"0{qubo.number_of_variables}b"
        ),
    }


def finite_shot_resample(
    probabilities: Sequence[float],
    *,
    number_of_qubits: int,
    shots: int,
    seed: int,
) -> dict[str, int]:
    """Draw reproducible ideal-simulator shots from exact circuit probabilities."""

    if shots <= 0:
        raise ValueError("shots must be positive")
    values = np.asarray(probabilities, dtype=float)
    values = np.clip(values, 0.0, None)
    values = values / values.sum()
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(shots, values)
    return {
        format(index, f"0{number_of_qubits}b"): int(count)
        for index, count in enumerate(sampled)
        if count
    }


def circuit_resource_record(
    probability_qnode: qml.QNode,
    gammas: pnp.ndarray,
    betas: pnp.ndarray,
) -> dict[str, object]:
    """Return PennyLane's expanded circuit depth and gate accounting."""

    specifications = qml.specs(probability_qnode, level="device")(gammas, betas)
    resources = specifications["resources"]
    return {
        "circuit_depth": int(resources.depth),
        "gate_count": int(resources.num_gates),
        "gate_types_json": json.dumps(dict(resources.gate_types), sort_keys=True),
        "gate_sizes_json": json.dumps(dict(resources.gate_sizes), sort_keys=True),
    }

