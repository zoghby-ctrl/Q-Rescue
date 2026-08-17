import pandas as pd
import pytest

from src.deadline_risk import RiskWeights, calculate_route_risk, score_route_table


def test_deadline_margin_and_full_score_have_minute_units() -> None:
    result = calculate_route_risk(
        expected_travel_time=20.0,
        lower_bound=17.0,
        upper_bound=25.0,
        deadline_minutes=23.0,
        reroute_penalty_minutes=1.5,
        weights=RiskWeights(1.0, 2.0, 3.0),
        variant="E",
    )
    assert result.interval_half_width == 4.0
    assert result.deadline_margin == -2.0
    assert result.deadline_penalty == 2.0
    assert result.risk_score == 20.0 + 4.0 + 2.0 * 2.0 + 3.0 * 1.5


def test_ablation_variants_activate_only_documented_terms() -> None:
    common = dict(
        expected_travel_time=20.0,
        lower_bound=18.0,
        upper_bound=24.0,
        deadline_minutes=22.0,
        reroute_penalty_minutes=5.0,
        weights=RiskWeights(1.0, 2.0, 4.0),
    )
    scores = {
        variant: calculate_route_risk(**common, variant=variant).risk_score
        for variant in "ABCDE"
    }
    assert scores == {"A": 20.0, "B": 23.0, "C": 24.0, "D": 27.0, "E": 47.0}


def test_no_deadline_penalty_when_upper_bound_meets_deadline() -> None:
    result = calculate_route_risk(12.0, 10.0, 14.0, 14.0, variant="C")
    assert result.deadline_margin == 0.0
    assert result.deadline_penalty == 0.0
    assert result.risk_score == 12.0


def test_route_table_is_scored_without_mutation() -> None:
    routes = pd.DataFrame(
        {
            "route_id": ["r0", "r1"],
            "prediction": [10.0, 11.0],
            "lower_bound": [8.0, 10.0],
            "upper_bound": [12.0, 12.0],
            "reroute_penalty_minutes": [0.0, 2.0],
        }
    )
    original_columns = routes.columns.tolist()
    scored = score_route_table(routes, 11.0, variant="E")
    assert routes.columns.tolist() == original_columns
    assert {"deadline_margin", "deadline_penalty", "risk_score"}.issubset(scored)
    assert scored.loc[0, "deadline_margin"] == -1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_travel_time": -1, "lower_bound": 0, "upper_bound": 1, "deadline_minutes": 2},
        {"expected_travel_time": 2, "lower_bound": 3, "upper_bound": 4, "deadline_minutes": 5},
        {"expected_travel_time": 2, "lower_bound": 1, "upper_bound": 3, "deadline_minutes": 5, "variant": "Z"},
    ],
)
def test_invalid_risk_inputs_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        calculate_route_risk(**kwargs)  # type: ignore[arg-type]
