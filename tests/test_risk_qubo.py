from itertools import product

import numpy as np
import pytest

from src.qubo import calculate_ising_energy
from src.risk_qubo import (
    brute_force_risk_qubo,
    build_risk_qubo,
    evaluate_risk_qubo,
    qubo_scaling_record,
    risk_qubo_to_ising,
)


@pytest.mark.parametrize("k", [4, 6, 8])
def test_generalized_qubo_ground_state_is_exact_route_argmin(k: int) -> None:
    risks = [20.0 + index for index in range(k)]
    risks[k // 2] = 7.5
    qubo = build_risk_qubo(risks)
    table = brute_force_risk_qubo(qubo)
    best = table.loc[table["qubo_cost"].idxmin()]
    assert len(table) == 2**k
    assert bool(best["valid"])
    assert int(best["selected_route_index"]) == k // 2
    assert float(best["qubo_cost"]) == 7.5


@pytest.mark.parametrize("k", [4, 6, 8])
def test_qubo_and_ising_are_equivalent_for_every_bitstring(k: int) -> None:
    qubo = build_risk_qubo(np.linspace(5.0, 11.0, k))
    _, offset, z_coefficients, zz_coefficients = risk_qubo_to_ising(qubo)
    for assignment in product((0, 1), repeat=k):
        qubo_energy = evaluate_risk_qubo(qubo, assignment)
        ising_energy = calculate_ising_energy(
            assignment, offset, z_coefficients, zz_coefficients
        )
        assert np.isclose(qubo_energy, ising_energy)


def test_scaling_counts_are_transparent_for_eight_routes() -> None:
    record = qubo_scaling_record(build_risk_qubo([1.0] * 8))
    assert record == {
        "K": 8,
        "number_of_qubits": 8,
        "search_space_size": 256,
        "qubo_constant_terms": 1,
        "qubo_linear_terms": 8,
        "qubo_quadratic_terms": 28,
        "qubo_total_terms": 37,
        "hamiltonian_identity_terms": 1,
        "hamiltonian_z_terms": 8,
        "hamiltonian_zz_terms": 28,
        "hamiltonian_nonidentity_terms": 36,
    }


def test_unsafe_penalty_and_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="greater than max"):
        build_risk_qubo([10.0, 20.0], penalty=20.0)
    with pytest.raises(ValueError, match="at least one"):
        build_risk_qubo([])
    qubo = build_risk_qubo([1.0, 2.0])
    with pytest.raises(ValueError, match="binary"):
        evaluate_risk_qubo(qubo, [0, 2])
