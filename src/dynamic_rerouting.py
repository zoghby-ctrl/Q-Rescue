"""Generic edge-by-edge dynamic rerouting simulator for canonical road graphs."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
import math

import networkx as nx
import numpy as np
import pandas as pd

from src.candidate_routes import CandidateRoute, generate_candidate_routes
from src.deadline_risk import RiskWeights, score_route_table
from src.network_schema import validate_network
from src.route_features import (
    RouteTrafficState,
    build_route_feature_table,
    estimate_pretrip_speed_kph,
)


Edge = tuple[Hashable, Hashable]
QuantumSelector = Callable[[pd.DataFrame], str]


@dataclass(frozen=True)
class DynamicEvent:
    """Observable traffic state plus hidden realized-time noise for one period."""

    name: str
    global_congestion: float = 0.2
    rain_indicator: int = 0
    localized_edges: frozenset[Edge] = frozenset()
    localized_congestion_boost: float = 0.0
    incident_edges: frozenset[Edge] = frozenset()
    closed_edges: frozenset[Edge] = frozenset()
    actual_noise_fraction: float = 0.03

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("event name must be nonempty")
        if not 0.0 <= self.global_congestion <= 1.0:
            raise ValueError("global_congestion must lie in [0, 1]")
        if self.rain_indicator not in {0, 1}:
            raise ValueError("rain_indicator must be binary")
        if self.localized_congestion_boost < 0.0:
            raise ValueError("localized_congestion_boost must be non-negative")
        if self.actual_noise_fraction < 0.0:
            raise ValueError("actual_noise_fraction must be non-negative")


@dataclass(frozen=True)
class DynamicSimulationResult:
    policy: str
    arrived: bool
    arrival_time_minutes: float | None
    deadline_violation: bool
    deadline_margin_minutes: float | None
    reroute_count: int
    unnecessary_reroutes: int
    forced_failure: bool
    decision_count: int
    route_stability: float
    decisions: pd.DataFrame

    def summary(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "arrived": self.arrived,
            "arrival_time_minutes": self.arrival_time_minutes,
            "deadline_violation": self.deadline_violation,
            "deadline_margin_minutes": self.deadline_margin_minutes,
            "reroute_count": self.reroute_count,
            "unnecessary_reroutes": self.unnecessary_reroutes,
            "forced_failure": self.forced_failure,
            "decision_count": self.decision_count,
            "route_stability": self.route_stability,
        }


def apply_dynamic_event(graph: nx.DiGraph, event: DynamicEvent) -> nx.DiGraph:
    """Return a canonical graph with closures removed and dynamic edge fields."""

    validate_network(graph)
    dynamic_graph = graph.copy()
    for edge in event.closed_edges:
        if dynamic_graph.has_edge(*edge):
            dynamic_graph.remove_edge(*edge)
    for source, target, attributes in dynamic_graph.edges(data=True):
        edge = (source, target)
        congestion = float(
            np.clip(
                event.global_congestion
                + (event.localized_congestion_boost if edge in event.localized_edges else 0.0),
                0.0,
                1.0,
            )
        )
        incident = int(edge in event.incident_edges)
        multiplier = (
            1.0
            + 0.90 * congestion
            + 0.20 * event.rain_indicator
            + 0.70 * incident
        )
        attributes["dynamic_congestion"] = congestion
        attributes["dynamic_incident"] = incident
        attributes["dynamic_expected_time_min"] = (
            float(attributes["base_travel_time_min"]) * multiplier
        )
    validate_network(dynamic_graph)
    dynamic_graph.graph["dynamic_event"] = event.name
    return dynamic_graph


def _route_state(
    graph: nx.DiGraph,
    route: CandidateRoute,
    event: DynamicEvent,
) -> RouteTrafficState:
    base_times = np.asarray(
        [float(graph.edges[edge]["base_travel_time_min"]) for edge in route.edges]
    )
    congestion = float(
        np.average(
            [float(graph.edges[edge]["dynamic_congestion"]) for edge in route.edges],
            weights=base_times,
        )
    )
    incident = int(any(int(graph.edges[edge]["dynamic_incident"]) for edge in route.edges))
    speed = estimate_pretrip_speed_kph(
        route.distance_km,
        route.base_travel_time_min,
        congestion,
        event.rain_indicator,
        incident,
    )
    return RouteTrafficState(
        hour_of_day=8,
        day_of_week=1,
        congestion=congestion,
        average_speed_kph=speed,
        rain_indicator=event.rain_indicator,
        incident_indicator=incident,
        historical_delay_minutes=0.0,
    )


def prepare_dynamic_candidates(
    graph: nx.DiGraph,
    current_node: Hashable,
    destination: Hashable,
    *,
    k: int,
    event: DynamicEvent,
    point_interval_model: object,
    distance_normalization_km: float,
    remaining_deadline_minutes: float,
    current_plan: Sequence[Hashable] | None,
    reroute_penalty_minutes: float,
    risk_weights: RiskWeights,
) -> tuple[pd.DataFrame, list[CandidateRoute]]:
    """Generate, predict, and risk-score the remaining candidate paths."""

    routes = generate_candidate_routes(
        graph,
        current_node,
        destination,
        k,
        weight="dynamic_expected_time_min",
        expected_time_attribute="dynamic_expected_time_min",
        require_k=False,
    )
    if not routes:
        raise nx.NetworkXNoPath(f"no dynamic route from {current_node!r} to {destination!r}")
    states = {route.route_id: _route_state(graph, route, event) for route in routes}
    features = build_route_feature_table(
        graph,
        routes,
        states,
        distance_normalization_km=distance_normalization_km,
    )
    intervals = point_interval_model.predict(features)
    plan_tuple = tuple(current_plan) if current_plan is not None else None
    route_nodes = {route.route_id: tuple(route.nodes) for route in routes}
    intervals["reroute_penalty_minutes"] = [
        0.0
        if plan_tuple is None or route_nodes[route_id] == plan_tuple
        else reroute_penalty_minutes
        for route_id in intervals["route_id"]
    ]
    scored = score_route_table(
        intervals,
        max(0.0, remaining_deadline_minutes),
        weights=risk_weights,
        variant="E",
    )
    return scored, routes


def choose_dynamic_route(
    candidates: pd.DataFrame,
    *,
    policy: str,
    quantum_selector: QuantumSelector | None = None,
) -> str:
    """Choose a route for one supported dynamic policy."""

    if candidates.empty:
        raise ValueError("candidate table cannot be empty")
    if policy == "expected_time_rerouting":
        ordered = candidates.sort_values(["prediction", "route_id"], kind="stable")
        return str(ordered.iloc[0]["route_id"])
    if policy == "uncertainty_aware_classical":
        ordered = candidates.sort_values(["risk_score", "route_id"], kind="stable")
        return str(ordered.iloc[0]["route_id"])
    if policy == "uncertainty_aware_qaoa":
        if quantum_selector is None:
            raise ValueError("uncertainty_aware_qaoa requires a quantum_selector")
        route_id = str(quantum_selector(candidates.copy()))
        if route_id not in set(candidates["route_id"]):
            raise ValueError("quantum_selector returned a route outside the candidate set")
        return route_id
    raise ValueError(f"unsupported dynamic policy {policy!r}")


def _remaining_dynamic_time(graph: nx.DiGraph, path: Sequence[Hashable]) -> float:
    return float(
        sum(
            graph.edges[source, target]["dynamic_expected_time_min"]
            for source, target in zip(path[:-1], path[1:])
        )
    )


def simulate_dynamic_route(
    graph: nx.DiGraph,
    origin: Hashable,
    destination: Hashable,
    events: Sequence[DynamicEvent],
    *,
    policy: str,
    k: int,
    deadline_minutes: float,
    point_interval_model: object,
    distance_normalization_km: float,
    risk_weights: RiskWeights | None = None,
    reroute_penalty_minutes: float = 1.5,
    random_seed: int = 2026,
    quantum_selector: QuantumSelector | None = None,
    maximum_decisions: int = 100,
) -> DynamicSimulationResult:
    """Traverse one edge per decision under a reproducible event sequence."""

    validate_network(graph)
    if not events:
        raise ValueError("at least one event is required")
    if deadline_minutes <= 0.0:
        raise ValueError("deadline_minutes must be positive")
    if policy == "never_reroute":
        current_plan: tuple[Hashable, ...] | None = tuple(
            nx.shortest_path(graph, origin, destination, weight="base_travel_time_min")
        )
    else:
        current_plan = None
    current_node = origin
    elapsed = 0.0
    reroute_count = 0
    unnecessary_reroutes = 0
    forced_failure = False
    rng = np.random.default_rng(random_seed)
    weights = risk_weights or RiskWeights()
    decision_rows: list[dict[str, object]] = []

    for decision_index in range(maximum_decisions):
        if current_node == destination:
            break
        event = events[min(decision_index, len(events) - 1)]
        dynamic_graph = apply_dynamic_event(graph, event)
        previous_plan = current_plan

        if current_plan is not None and (
            len(current_plan) < 2
            or current_plan[0] != current_node
            or not dynamic_graph.has_edge(current_plan[0], current_plan[1])
        ):
            if policy == "never_reroute":
                forced_failure = True
                decision_rows.append(
                    {
                        "decision_index": decision_index,
                        "event": event.name,
                        "current_node": current_node,
                        "selected_route_id": None,
                        "failure": "planned next edge unavailable",
                    }
                )
                break
            current_plan = None

        old_plan = previous_plan
        if policy == "never_reroute":
            chosen_nodes = current_plan
            selected_route_id = "fixed_plan"
            predicted_time = _remaining_dynamic_time(dynamic_graph, chosen_nodes)
            risk_score = math.nan
        else:
            try:
                candidates, routes = prepare_dynamic_candidates(
                    dynamic_graph,
                    current_node,
                    destination,
                    k=k,
                    event=event,
                    point_interval_model=point_interval_model,
                    distance_normalization_km=distance_normalization_km,
                    remaining_deadline_minutes=deadline_minutes - elapsed,
                    current_plan=current_plan,
                    reroute_penalty_minutes=reroute_penalty_minutes,
                    risk_weights=weights,
                )
            except (nx.NetworkXNoPath, ValueError):
                forced_failure = True
                break
            if len(routes) == 1:
                selected_route_id = routes[0].route_id
            else:
                selected_route_id = choose_dynamic_route(
                    candidates, policy=policy, quantum_selector=quantum_selector
                )
            chosen_route = next(route for route in routes if route.route_id == selected_route_id)
            chosen_nodes = tuple(chosen_route.nodes)
            chosen_row = candidates[candidates["route_id"] == selected_route_id].iloc[0]
            predicted_time = float(chosen_row["prediction"])
            risk_score = float(chosen_row["risk_score"])

        rerouted = old_plan is not None and tuple(chosen_nodes) != tuple(old_plan)
        if rerouted:
            reroute_count += 1
            if all(dynamic_graph.has_edge(a, b) for a, b in zip(old_plan[:-1], old_plan[1:])):
                if _remaining_dynamic_time(dynamic_graph, chosen_nodes) >= _remaining_dynamic_time(
                    dynamic_graph, old_plan
                ):
                    unnecessary_reroutes += 1

        source, target = chosen_nodes[0], chosen_nodes[1]
        expected_edge_time = float(
            dynamic_graph.edges[source, target]["dynamic_expected_time_min"]
        )
        realized_edge_time = max(
            0.05,
            expected_edge_time * (1.0 + rng.normal(0.0, event.actual_noise_fraction)),
        )
        elapsed += realized_edge_time
        decision_rows.append(
            {
                "decision_index": decision_index,
                "event": event.name,
                "current_node": current_node,
                "selected_route_id": selected_route_id,
                "selected_path": " -> ".join(str(node) for node in chosen_nodes),
                "rerouted": rerouted,
                "predicted_remaining_time": predicted_time,
                "risk_score": risk_score,
                "traversed_edge": f"{source} -> {target}",
                "realized_edge_time": realized_edge_time,
                "elapsed_time": elapsed,
            }
        )
        current_node = target
        current_plan = tuple(chosen_nodes[1:])

    arrived = current_node == destination and not forced_failure
    arrival_time = elapsed if arrived else None
    deadline_violation = (not arrived) or elapsed > deadline_minutes
    margin = deadline_minutes - elapsed if arrived else None
    opportunities = max(1, len(decision_rows) - 1)
    stability = max(0.0, 1.0 - reroute_count / opportunities)
    return DynamicSimulationResult(
        policy=policy,
        arrived=arrived,
        arrival_time_minutes=arrival_time,
        deadline_violation=deadline_violation,
        deadline_margin_minutes=margin,
        reroute_count=reroute_count,
        unnecessary_reroutes=unnecessary_reroutes,
        forced_failure=forced_failure,
        decision_count=len(decision_rows),
        route_stability=stability,
        decisions=pd.DataFrame(decision_rows),
    )
