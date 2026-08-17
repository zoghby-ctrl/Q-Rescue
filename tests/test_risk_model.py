"""Tests for route inputs and the deterministic risk score."""

from src.risk_model import (
    add_risk_scores,
    choose_expected_time_route,
    choose_risk_aware_route,
)
from src.route_data import (
    calculate_path_time,
    create_road_network,
    create_route_dataframe,
    get_candidate_paths,
)


def test_graph_path_times_match_route_table() -> None:
    """Every graph path must reproduce its stated expected route time."""

    graph = create_road_network()
    paths = get_candidate_paths()
    route_data = create_route_dataframe().set_index("route")

    for route_name, path in paths.items():
        path_time = calculate_path_time(graph, path)
        expected_time = route_data.loc[route_name, "expected_time_minutes"]
        assert path_time == expected_time


def test_route_a_has_lowest_expected_time() -> None:
    """The expected-time-only baseline must select Route A."""

    route_data = create_route_dataframe()
    assert choose_expected_time_route(route_data) == "Route A"


def test_route_b_has_lowest_risk_score() -> None:
    """Weighted uncertainty must change the selection to Route B."""

    scored_routes = add_risk_scores(create_route_dataframe(), uncertainty_weight=1.0)
    assert scored_routes["risk_score"].tolist() == [28.0, 23.0, 27.0, 27.0]
    assert choose_risk_aware_route(scored_routes) == "Route B"

