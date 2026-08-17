import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.feasibility_qaoa import (
    circuit_resource_record,
    create_probability_qnode,
    finite_shot_resample,
    one_hot_basis_indices,
    one_hot_w_state,
    summarize_probabilities,
)
from src.risk_qubo import build_risk_qubo, risk_qubo_to_ising


def test_one_hot_state_indices_match_displayed_bit_order() -> None:
    assert one_hot_basis_indices(4) == (8, 4, 2, 1)
    state = one_hot_w_state(4)
    np.testing.assert_allclose(np.abs(state[[8, 4, 2, 1]]) ** 2, 0.25)
    assert np.isclose(np.vdot(state, state), 1.0)


def test_xy_circuit_preserves_feasibility_while_x_mixer_does_not() -> None:
    qubo = build_risk_qubo([5.0, 6.0, 8.0, 9.0])
    hamiltonian, _, _, _ = risk_qubo_to_ising(qubo)
    gammas = pnp.array([0.21], requires_grad=False)
    betas = pnp.array([0.37], requires_grad=False)
    xy = create_probability_qnode(
        hamiltonian, number_of_qubits=4, depth=1, method="feasible_xy"
    )
    standard = create_probability_qnode(
        hamiltonian, number_of_qubits=4, depth=1, method="standard_x"
    )
    xy_summary = summarize_probabilities(xy(gammas, betas), qubo, optimal_route_index=0)
    standard_summary = summarize_probabilities(
        standard(gammas, betas), qubo, optimal_route_index=0
    )
    assert np.isclose(xy_summary["valid_probability"], 1.0)
    assert standard_summary["valid_probability"] < 1.0


def test_probability_summary_shots_and_resources_are_reproducible() -> None:
    qubo = build_risk_qubo([3.0, 4.0, 5.0, 6.0])
    hamiltonian, _, _, _ = risk_qubo_to_ising(qubo)
    circuit = create_probability_qnode(
        hamiltonian, number_of_qubits=4, depth=1, method="standard_x"
    )
    gammas = pnp.array([0.1], requires_grad=False)
    betas = pnp.array([0.2], requires_grad=False)
    probabilities = circuit(gammas, betas)
    summary = summarize_probabilities(probabilities, qubo, optimal_route_index=0)
    first = finite_shot_resample(probabilities, number_of_qubits=4, shots=200, seed=7)
    second = finite_shot_resample(probabilities, number_of_qubits=4, shots=200, seed=7)
    resources = circuit_resource_record(circuit, gammas, betas)
    assert first == second
    assert sum(first.values()) == 200
    assert summary["expected_qubo_energy"] >= 3.0
    assert resources["circuit_depth"] > 0
    assert resources["gate_count"] > 0
