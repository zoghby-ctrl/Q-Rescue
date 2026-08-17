import numpy as np
import pytest

from src.candidate_routes import generate_candidate_routes
from src.route_features import (
    RouteTrafficState,
    build_route_feature_table,
    estimate_pretrip_speed_kph,
)
from src.synthetic_networks import generate_grid_network
from src.traffic_model import FEATURE_COLUMNS


def _state_for_route(route: object, index: int) -> RouteTrafficState:
    speed = estimate_pretrip_speed_kph(
        route.distance_km,
        route.base_travel_time_min,
        congestion=0.25 + 0.05 * index,
        rain_indicator=1,
        incident_indicator=int(index == 2),
    )
    return RouteTrafficState(
        hour_of_day=8,
        day_of_week=1,
        congestion=0.25 + 0.05 * index,
        average_speed_kph=speed,
        rain_indicator=1,
        incident_indicator=int(index == 2),
        historical_delay_minutes=0.4 * index,
    )


def test_candidate_feature_table_exactly_matches_training_contract() -> None:
    graph = generate_grid_network(rows=4, columns=4, seed=91)
    routes = generate_candidate_routes(
        graph, graph.graph["origin"], graph.graph["destination"], 4
    )
    states = {route.route_id: _state_for_route(route, i) for i, route in enumerate(routes)}
    table = build_route_feature_table(
        graph, routes, states, distance_normalization_km=5.0
    )
    assert list(table.loc[:, FEATURE_COLUMNS].columns) == list(FEATURE_COLUMNS)
    assert table["route_id"].tolist() == ["x_0", "x_1", "x_2", "x_3"]
    assert np.isfinite(table.loc[:, FEATURE_COLUMNS].to_numpy()).all()
    assert (table["normalized_distance"] > 0).all()


def test_speed_estimate_is_pretrip_and_responds_to_observable_conditions() -> None:
    clear = estimate_pretrip_speed_kph(2.0, 4.0, 0.1, 0, 0)
    disrupted = estimate_pretrip_speed_kph(2.0, 4.0, 0.8, 1, 1)
    assert clear > disrupted > 0.0


def test_missing_route_state_and_invalid_state_are_rejected() -> None:
    graph = generate_grid_network(rows=3, columns=3, seed=4)
    routes = generate_candidate_routes(
        graph, graph.graph["origin"], graph.graph["destination"], 2
    )
    with pytest.raises(ValueError, match="exactly one"):
        build_route_feature_table(graph, routes, {}, distance_normalization_km=3.0)
    with pytest.raises(ValueError, match="congestion"):
        RouteTrafficState(8, 1, 1.5, 20.0)
