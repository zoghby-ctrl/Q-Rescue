from __future__ import annotations

import numpy as np
from pennylane import numpy as pnp

from src.feasibility_qaoa import create_probability_qnode
from src.qaoa_noise import (
    LayerNoiseParameters,
    apply_symmetric_readout_error,
    create_layer_noise_probability_qnode,
    damping_probability,
    effective_error_probability,
)
from src.risk_qubo import build_risk_qubo, risk_qubo_to_ising


def _zero_noise() -> LayerNoiseParameters:
    return LayerNoiseParameters(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_effective_error_and_damping_are_bounded() -> None:
    assert np.isclose(effective_error_probability(0.01, 2, 0.02, 3), 1 - 0.99**2 * 0.98**3)
    assert 0.0 < damping_probability(300.0, 100.0) < 1.0


def test_readout_channel_preserves_mass_and_zero_error_identity() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(
        apply_symmetric_readout_error(values, number_of_qubits=2, bit_flip_probability=0.0),
        values,
    )
    noisy = apply_symmetric_readout_error(
        values, number_of_qubits=2, bit_flip_probability=0.05
    )
    assert np.isclose(noisy.sum(), 1.0)
    assert np.all(noisy >= 0.0)


def test_zero_layer_noise_matches_ideal_for_both_mixers() -> None:
    qubo = build_risk_qubo([1.0, 2.0, 3.0, 4.0])
    hamiltonian, _, _, _ = risk_qubo_to_ising(qubo)
    gammas = pnp.array([0.1])
    betas = pnp.array([-0.2])
    for method in ["standard_x", "feasible_xy"]:
        ideal = create_probability_qnode(
            hamiltonian, number_of_qubits=4, depth=1, method=method
        )
        mixed = create_layer_noise_probability_qnode(
            hamiltonian,
            number_of_qubits=4,
            depth=1,
            method=method,
            noise=_zero_noise(),
        )
        np.testing.assert_allclose(
            np.asarray(mixed(gammas, betas)),
            np.asarray(ideal(gammas, betas)),
            atol=1e-9,
        )
