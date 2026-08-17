"""Build model-compatible features for candidate routes at a decision time."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import networkx as nx
import numpy as np
import pandas as pd

from src.candidate_routes import CandidateRoute
from src.network_schema import validate_network
from src.traffic_data import MODEL_FEATURE_COLUMNS, encode_road_class
from src.traffic_model import FEATURE_COLUMNS


@dataclass(frozen=True)
class RouteTrafficState:
    """Observable, pre-decision traffic context for one candidate route."""

    hour_of_day: int
    day_of_week: int
    congestion: float
    average_speed_kph: float
    rain_indicator: int = 0
    incident_indicator: int = 0
    historical_delay_minutes: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.hour_of_day, bool) or not 0 <= self.hour_of_day <= 23:
            raise ValueError("hour_of_day must be an integer from 0 through 23")
        if isinstance(self.day_of_week, bool) or not 0 <= self.day_of_week <= 6:
            raise ValueError("day_of_week must be an integer from 0 through 6")
        if not np.isfinite(self.congestion) or not 0.0 <= self.congestion <= 1.0:
            raise ValueError("congestion must be finite and lie in [0, 1]")
        if not np.isfinite(self.average_speed_kph) or self.average_speed_kph <= 0.0:
            raise ValueError("average_speed_kph must be finite and positive")
        if self.rain_indicator not in {0, 1} or self.incident_indicator not in {0, 1}:
            raise ValueError("rain_indicator and incident_indicator must be binary")
        if (
            not np.isfinite(self.historical_delay_minutes)
            or self.historical_delay_minutes < 0.0
        ):
            raise ValueError("historical_delay_minutes must be finite and non-negative")


def estimate_pretrip_speed_kph(
    distance_km: float,
    base_travel_time_min: float,
    congestion: float,
    rain_indicator: int,
    incident_indicator: int,
    *,
    sensor_adjustment_kph: float = 0.0,
) -> float:
    """Apply the traffic generator's pre-trip sensor relationship, never y/d."""

    if distance_km <= 0.0 or base_travel_time_min <= 0.0:
        raise ValueError("distance and base travel time must be positive")
    if not 0.0 <= congestion <= 1.0:
        raise ValueError("congestion must lie in [0, 1]")
    if rain_indicator not in {0, 1} or incident_indicator not in {0, 1}:
        raise ValueError("rain and incident indicators must be binary")
    free_flow_speed = 60.0 * distance_km / base_travel_time_min
    observable_multiplier = (
        1.0 + 0.72 * congestion + 0.10 * rain_indicator + 0.28 * incident_indicator
    )
    return max(1.0, free_flow_speed / observable_multiplier + sensor_adjustment_kph)


def _aggregate_route_attributes(
    graph: nx.DiGraph, route: CandidateRoute
) -> tuple[float, float, int]:
    base_times = np.asarray(
        [float(graph.edges[edge]["base_travel_time_min"]) for edge in route.edges]
    )
    road_codes = np.asarray(
        [encode_road_class(graph.edges[edge].get("road_class")) for edge in route.edges],
        dtype=float,
    )
    reliabilities = np.asarray(
        [float(graph.edges[edge].get("reliability", 0.90)) for edge in route.edges]
    )
    return (
        float(np.average(road_codes, weights=base_times)),
        float(np.average(reliabilities, weights=base_times)),
        max(0, len(route.nodes) - 2),
    )


def build_route_feature_table(
    graph: nx.DiGraph,
    routes: Sequence[CandidateRoute],
    traffic_states: Mapping[str, RouteTrafficState],
    *,
    distance_normalization_km: float,
) -> pd.DataFrame:
    """Return exactly the Step 4 feature contract plus route trace metadata."""

    validate_network(graph)
    normalization = float(distance_normalization_km)
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("distance_normalization_km must be finite and positive")
    if not routes:
        raise ValueError("at least one candidate route is required")
    route_ids = [route.route_id for route in routes]
    if len(route_ids) != len(set(route_ids)):
        raise ValueError("candidate route IDs must be unique")
    if set(traffic_states) != set(route_ids):
        raise ValueError("traffic_states must contain exactly one state per route_id")

    rows: list[dict[str, object]] = []
    for route in routes:
        state = traffic_states[route.route_id]
        road_class_code, reliability, intersection_count = _aggregate_route_attributes(
            graph, route
        )
        rows.append(
            {
                "route_id": route.route_id,
                "ordered_nodes": " -> ".join(str(node) for node in route.nodes),
                "distance_km": route.distance_km,
                "distance_normalization_km": normalization,
                "base_route_time_min": route.base_travel_time_min,
                "expected_graph_time_min": route.expected_travel_time_min,
                "edge_count": route.edge_count,
                "route_reliability": reliability,
                "normalized_distance": route.distance_km / normalization,
                "base_travel_time_min": route.base_travel_time_min,
                "hour_sin": math.sin(2.0 * math.pi * state.hour_of_day / 24.0),
                "hour_cos": math.cos(2.0 * math.pi * state.hour_of_day / 24.0),
                "day_sin": math.sin(2.0 * math.pi * state.day_of_week / 7.0),
                "day_cos": math.cos(2.0 * math.pi * state.day_of_week / 7.0),
                "congestion": state.congestion,
                "average_speed_kph": state.average_speed_kph,
                "rain_indicator": state.rain_indicator,
                "incident_indicator": state.incident_indicator,
                "road_class_code": road_class_code,
                "intersection_count": intersection_count,
                "historical_delay_minutes": state.historical_delay_minutes,
            }
        )
    table = pd.DataFrame(rows)
    if tuple(MODEL_FEATURE_COLUMNS) != FEATURE_COLUMNS:
        raise RuntimeError("traffic generation and model feature contracts disagree")
    if list(table.loc[:, FEATURE_COLUMNS].columns) != list(FEATURE_COLUMNS):
        raise RuntimeError("candidate feature table does not match the model contract")
    return table

