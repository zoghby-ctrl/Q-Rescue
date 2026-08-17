import numpy as np
import pytest

from src.qaoa_diagnostics import summarize_qaoa_distribution
from src.risk_qubo import build_risk_qubo


def test_uniform_distribution_separates_feasibility_and_conditional_optimality():
    qubo = build_risk_qubo([1.0, 2.0, 3.0, 4.0])
    result = summarize_qaoa_distribution(np.full(16, 1.0 / 16.0), qubo)

    assert result.metrics["feasibility_probability"] == pytest.approx(0.25)
    assert result.metrics["optimal_state_probability"] == pytest.approx(0.0625)
    assert result.metrics["optimal_probability_given_feasible"] == pytest.approx(0.25)
    assert result.metrics["normalized_feasible_entropy"] == pytest.approx(1.0)
    assert result.metrics["top2_feasible_probability_conditional"] == pytest.approx(0.5)
    assert result.metrics["risk_regret"] == pytest.approx(0.0)
    assert len(result.feasible_distribution) == 4


def test_feasible_distribution_reports_entropy_rank_and_regret():
    qubo = build_risk_qubo([1.0, 3.0, 2.0, 4.0])
    probabilities = np.zeros(16)
    # Wire-order one-hot indices are 8, 4, 2, 1 for x_0...x_3.
    probabilities[[8, 4, 2, 1]] = [0.10, 0.60, 0.20, 0.10]
    result = summarize_qaoa_distribution(probabilities, qubo)

    assert result.metrics["feasibility_probability"] == pytest.approx(1.0)
    assert result.metrics["optimal_state_probability"] == pytest.approx(0.10)
    assert result.metrics["selected_route_id"] == "x_1"
    assert result.metrics["selected_route_risk_rank"] == 3
    assert result.metrics["risk_regret"] == pytest.approx(2.0)
    assert result.metrics["top1_top2_probability_gap_conditional"] == pytest.approx(0.4)
    assert 0.0 < result.metrics["normalized_feasible_entropy"] < 1.0


def test_declared_optima_must_match_risk_argmin():
    qubo = build_risk_qubo([1.0, 2.0])
    with pytest.raises(ValueError, match="does not match"):
        summarize_qaoa_distribution([0.25] * 4, qubo, optimal_route_indices=[1])

