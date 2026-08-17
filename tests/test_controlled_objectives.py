import pytest

from src.controlled_objectives import controlled_objective_table


def test_b_to_c_changes_only_uncertainty_component():
    table = controlled_objective_table(
        [10.0, 12.0],
        [1.0, 3.0],
        deadline_minutes=11.0,
        lambda_deadline=2.0,
        lambda_uncertainty=1.5,
    )

    assert table["A_expected_time"].tolist() == pytest.approx([10.0, 12.0])
    assert table["B_expected_plus_deadline"].tolist() == pytest.approx([10.0, 14.0])
    assert table["C_expected_plus_deadline_plus_uncertainty"].tolist() == pytest.approx(
        [11.5, 18.5]
    )
    assert "D_full_dynamic" not in table


def test_dynamic_d_adds_only_reroute_penalty():
    table = controlled_objective_table(
        [10.0, 12.0],
        [1.0, 1.0],
        deadline_minutes=11.0,
        reroute_penalties=[0.0, 2.0],
        lambda_reroute=0.5,
    )
    difference = (
        table["D_full_dynamic"]
        - table["C_expected_plus_deadline_plus_uncertainty"]
    )
    assert difference.tolist() == pytest.approx([0.0, 1.0])

