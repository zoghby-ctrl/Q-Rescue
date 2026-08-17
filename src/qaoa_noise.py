"""Transparent local channel model for small-QAOA density-matrix studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.feasibility_qaoa import create_mixer_hamiltonian, one_hot_w_state


@dataclass(frozen=True)
class LayerNoiseParameters:
    """Effective per-wire channel strengths after logical QAOA blocks."""

    preparation_depolarizing: float
    preparation_amplitude_damping: float
    cost_depolarizing: float
    cost_amplitude_damping: float
    mixer_depolarizing: float
    mixer_amplitude_damping: float
    readout_bit_flip: float

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not np.isfinite(values).all() or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("noise probabilities must be finite and lie in [0, 1]")


def effective_error_probability(
    one_qubit_error: float,
    one_qubit_count: int,
    two_qubit_error: float,
    two_qubit_count: int,
) -> float:
    """Combine independent logical interaction error opportunities."""

    if min(one_qubit_error, two_qubit_error) < 0.0 or max(
        one_qubit_error, two_qubit_error
    ) > 1.0:
        raise ValueError("gate error probabilities must lie in [0, 1]")
    if one_qubit_count < 0 or two_qubit_count < 0:
        raise ValueError("interaction counts must be nonnegative")
    return float(
        1.0
        - (1.0 - one_qubit_error) ** one_qubit_count
        * (1.0 - two_qubit_error) ** two_qubit_count
    )


def damping_probability(duration_nanoseconds: float, t1_microseconds: float) -> float:
    """Convert a block duration and T1 into amplitude-damping probability."""

    if duration_nanoseconds < 0.0 or t1_microseconds <= 0.0:
        raise ValueError("duration must be nonnegative and T1 positive")
    return float(1.0 - np.exp(-duration_nanoseconds / (1_000.0 * t1_microseconds)))


def _channels(depolarizing: float, damping: float, wire: int) -> None:
    if depolarizing > 0.0:
        qml.DepolarizingChannel(depolarizing, wires=wire)
    if damping > 0.0:
        qml.AmplitudeDamping(damping, wires=wire)


def create_layer_noise_probability_qnode(
    cost_hamiltonian: qml.Hamiltonian,
    *,
    number_of_qubits: int,
    depth: int,
    method: str,
    noise: LayerNoiseParameters,
) -> qml.QNode:
    """Return a density-matrix QNode with local channels after logical blocks.

    Channels summarize documented per-gate error opportunities and durations;
    this is a phenomenological local model, not a transpiled backend model.
    """

    if number_of_qubits <= 0 or depth <= 0:
        raise ValueError("number_of_qubits and depth must be positive")
    mixer = create_mixer_hamiltonian(number_of_qubits, method)
    device = qml.device("default.mixed", wires=number_of_qubits)

    @qml.qnode(device, interface="autograd")
    def circuit(gammas: pnp.ndarray, betas: pnp.ndarray) -> pnp.ndarray:
        if method == "standard_x":
            for wire in range(number_of_qubits):
                qml.Hadamard(wires=wire)
        elif method == "feasible_xy":
            qml.StatePrep(one_hot_w_state(number_of_qubits), wires=range(number_of_qubits))
        else:
            raise ValueError("unknown QAOA method")
        for wire in range(number_of_qubits):
            _channels(
                noise.preparation_depolarizing,
                noise.preparation_amplitude_damping,
                wire,
            )
        for gamma, beta in zip(gammas, betas, strict=True):
            qml.qaoa.cost_layer(gamma, cost_hamiltonian)
            for wire in range(number_of_qubits):
                _channels(noise.cost_depolarizing, noise.cost_amplitude_damping, wire)
            qml.qaoa.mixer_layer(beta, mixer)
            for wire in range(number_of_qubits):
                _channels(noise.mixer_depolarizing, noise.mixer_amplitude_damping, wire)
        return qml.probs(wires=range(number_of_qubits))

    return circuit


def apply_symmetric_readout_error(
    probabilities: np.ndarray, *, number_of_qubits: int, bit_flip_probability: float
) -> np.ndarray:
    """Apply independent symmetric readout bit flips to a distribution."""

    values = np.asarray(probabilities, dtype=float)
    if values.shape != (2**number_of_qubits,):
        raise ValueError("probability vector length does not match number_of_qubits")
    if not np.isfinite(values).all() or np.any(values < -1e-12):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("probabilities must sum to one")
    error = float(bit_flip_probability)
    if not 0.0 <= error <= 1.0:
        raise ValueError("bit_flip_probability must lie in [0, 1]")
    output = np.zeros_like(values)
    for true_index, true_probability in enumerate(values):
        for observed_index in range(len(values)):
            hamming = (true_index ^ observed_index).bit_count()
            output[observed_index] += true_probability * error**hamming * (
                1.0 - error
            ) ** (number_of_qubits - hamming)
    output = np.clip(output, 0.0, None)
    return output / output.sum()
