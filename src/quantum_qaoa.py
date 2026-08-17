"""Small PennyLane QAOA circuits, training, sampling, and decoding helpers."""

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp


def create_mixer_hamiltonian(number_of_qubits: int = 4) -> qml.Hamiltonian:
    """Return the standard mixer H_M = X_0 + ... + X_(n-1)."""

    coefficients = [1.0] * number_of_qubits
    operations = [qml.PauliX(wire) for wire in range(number_of_qubits)]
    return qml.Hamiltonian(coefficients, operations)


def create_training_qnode(
    cost_hamiltonian: qml.Hamiltonian,
    mixer_hamiltonian: qml.Hamiltonian,
    depth: int,
    number_of_qubits: int = 4,
) -> qml.QNode:
    """Create an analytic QNode that returns the cost expectation value."""

    device = qml.device("default.qubit", wires=number_of_qubits)

    @qml.qnode(device, interface="autograd")
    def training_qnode(gammas: pnp.ndarray, betas: pnp.ndarray) -> pnp.ndarray:
        for wire in range(number_of_qubits):
            qml.Hadamard(wires=wire)
        for layer_index in range(depth):
            qml.qaoa.cost_layer(gammas[layer_index], cost_hamiltonian)
            qml.qaoa.mixer_layer(betas[layer_index], mixer_hamiltonian)
        return qml.expval(cost_hamiltonian)

    return training_qnode


def create_sampling_qnode(
    cost_hamiltonian: qml.Hamiltonian,
    mixer_hamiltonian: qml.Hamiltonian,
    depth: int,
    shots: int = 2000,
    seed: int = 2026,
    number_of_qubits: int = 4,
) -> qml.QNode:
    """Create a finite-shot QNode returning samples in wire order 0,1,2,3."""

    device = qml.device("default.qubit", wires=number_of_qubits, seed=seed)

    @qml.qnode(device, interface="autograd", shots=shots)
    def sampling_qnode(gammas: pnp.ndarray, betas: pnp.ndarray) -> pnp.ndarray:
        for wire in range(number_of_qubits):
            qml.Hadamard(wires=wire)
        for layer_index in range(depth):
            qml.qaoa.cost_layer(gammas[layer_index], cost_hamiltonian)
            qml.qaoa.mixer_layer(betas[layer_index], mixer_hamiltonian)
        return qml.sample(wires=list(range(number_of_qubits)))

    return sampling_qnode


def optimize_qaoa(
    training_qnode: Callable[[pnp.ndarray, pnp.ndarray], pnp.ndarray],
    depth: int,
    iterations: int = 100,
    step_size: float = 0.02,
    initial_gammas: Sequence[float] | None = None,
    initial_betas: Sequence[float] | None = None,
) -> tuple[pnp.ndarray, pnp.ndarray, pd.DataFrame]:
    """Optimize fixed QAOA angles with Adam and return angles plus history."""

    if initial_gammas is None:
        initial_gammas = [0.02] * depth
    if initial_betas is None:
        initial_betas = [-0.20] * depth

    gammas = pnp.array(initial_gammas, requires_grad=True)
    betas = pnp.array(initial_betas, requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=step_size)

    history_rows = [
        {"iteration": 0, "cost_expectation": float(training_qnode(gammas, betas))}
    ]
    for iteration in range(1, iterations + 1):
        gammas, betas = optimizer.step(training_qnode, gammas, betas)
        current_cost = float(training_qnode(gammas, betas))
        history_rows.append({"iteration": iteration, "cost_expectation": current_cost})

    return gammas, betas, pd.DataFrame(history_rows)


def sample_to_bitstring(sample: Sequence[int]) -> str:
    """Convert [wire 0,...,wire 3] directly to x_0...x_3 with no reversal."""

    return "".join(str(int(bit)) for bit in sample)


def sample_to_route(sample: Sequence[int], route_names: Sequence[str]) -> str | None:
    """Decode one quantum sample to a route, or None if it is not one-hot."""

    bits = [int(bit) for bit in sample]
    if len(bits) != len(route_names):
        raise ValueError("The sample and route_names must have the same length.")
    if sum(bits) != 1:
        return None
    return str(route_names[bits.index(1)])


def summarize_samples(
    samples: np.ndarray,
    route_names: Sequence[str],
    optimal_route: str,
) -> dict[str, object]:
    """Return counts and route probabilities from finite-shot samples."""

    bitstrings = [sample_to_bitstring(sample) for sample in samples]
    measured_counts = Counter(bitstrings)
    number_of_qubits = len(route_names)
    all_bitstrings = [format(index, f"0{number_of_qubits}b") for index in range(2**number_of_qubits)]
    counts = {bitstring: measured_counts.get(bitstring, 0) for bitstring in all_bitstrings}

    total_shots = len(samples)
    valid_route_counts = {route_name: 0 for route_name in route_names}
    for sample in samples:
        route_name = sample_to_route(sample, route_names)
        if route_name is not None:
            valid_route_counts[route_name] += 1

    valid_count = sum(valid_route_counts.values())
    most_frequent_valid_route = None
    if valid_count > 0:
        most_frequent_valid_route = max(
            route_names,
            key=lambda route_name: valid_route_counts[route_name],
        )

    return {
        "counts": counts,
        "valid_route_counts": valid_route_counts,
        "valid_solution_probability": valid_count / total_shots,
        "optimal_route_probability": valid_route_counts[optimal_route] / total_shots,
        "most_frequent_valid_route": most_frequent_valid_route,
    }
