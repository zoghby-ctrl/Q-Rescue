"""Transparent classical reference methods for candidate emergency routes.

These methods establish deterministic comparison points and exact ground truth
for tractable candidate sets.  They report objective values and selected paths;
they do not make claims based on wall-clock timing.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
import heapq
from itertools import count
import math
from numbers import Real

import networkx as nx
import pandas as pd

from src.network_schema import validate_network


class UnreachablePathError(ValueError):
    """Raised when no directed path connects the requested endpoints."""

    def __init__(self, origin: Hashable, destination: Hashable) -> None:
        self.origin = origin
        self.destination = destination
        super().__init__(f"no directed path exists from {origin!r} to {destination!r}")


@dataclass(frozen=True, slots=True)
class ClassicalBaselineResult:
    """One auditable classical selection and its available route summaries."""

    method: str
    route_id: str | None
    path: tuple[Hashable, ...]
    objective_name: str
    objective_value: float
    distance_km: float | None = None
    expected_travel_time_min: float | None = None
    upper_bound: float | None = None
    risk_score: float | None = None
    candidate_count: int | None = None

    def as_record(self) -> dict[str, object]:
        """Return a stable table-ready record without losing path order."""

        record = asdict(self)
        record["path"] = tuple(self.path)
        return record


RESULT_COLUMNS = (
    "method",
    "route_id",
    "path",
    "objective_name",
    "objective_value",
    "distance_km",
    "expected_travel_time_min",
    "upper_bound",
    "risk_score",
    "candidate_count",
)


def _finite_number(value: object, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    if positive and numeric <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return numeric


def _node_tie_key(node: Hashable) -> tuple[str, str, str]:
    node_type = type(node)
    return node_type.__module__, node_type.__qualname__, repr(node)


def _path_tie_key(path: Sequence[Hashable]) -> tuple[tuple[str, str, str], ...]:
    return tuple(_node_tie_key(node) for node in path)


def _route_id_tie_key(route_id: str) -> tuple[int, object, str]:
    """Use decision-variable order for x_N IDs, lexical order otherwise."""

    suffix = route_id.removeprefix("x_")
    if route_id.startswith("x_") and suffix.isdigit():
        return 0, int(suffix), route_id
    return 1, route_id, route_id


def _validated_path(value: object, row_label: str) -> tuple[Hashable, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{row_label} path must be an ordered node sequence")
    path = tuple(value)
    if len(path) < 2:
        raise ValueError(f"{row_label} path must contain at least two nodes")
    try:
        unique_count = len(set(path))
    except TypeError as error:
        raise ValueError(f"{row_label} path nodes must be hashable") from error
    if unique_count != len(path):
        raise ValueError(f"{row_label} path must be loopless")
    return path


def _validate_candidate_table(
    route_table: pd.DataFrame,
    *,
    objective_column: str,
    route_id_column: str,
    path_column: str,
) -> list[tuple[pd.Series, tuple[Hashable, ...], float]]:
    if not isinstance(route_table, pd.DataFrame):
        raise TypeError("route_table must be a pandas DataFrame")
    if route_table.empty:
        raise ValueError("route_table must contain at least one candidate")
    required = {route_id_column, path_column, objective_column}
    missing = sorted(required.difference(route_table.columns))
    if missing:
        raise ValueError(f"route_table is missing required columns: {missing}")
    if route_table[route_id_column].isna().any():
        raise ValueError(f"{route_id_column} cannot contain missing values")
    if route_table[route_id_column].duplicated().any():
        raise ValueError(f"{route_id_column} must be unique")

    validated: list[tuple[pd.Series, tuple[Hashable, ...], float]] = []
    for row_index, row in route_table.iterrows():
        route_id = row[route_id_column]
        if not isinstance(route_id, str) or not route_id.strip():
            raise ValueError(f"row {row_index!r} has an invalid {route_id_column}")
        path = _validated_path(row[path_column], f"route {route_id!r}")
        objective = _finite_number(row[objective_column], objective_column)
        validated.append((row, path, objective))
    return validated


def _optional_numeric(row: pd.Series, column_name: str) -> float | None:
    if column_name not in row.index or pd.isna(row[column_name]):
        return None
    return _finite_number(row[column_name], column_name)


def _select_candidate(
    route_table: pd.DataFrame,
    *,
    method: str,
    objective_column: str,
    route_id_column: str = "route_id",
    path_column: str = "ordered_nodes",
    distance_column: str = "distance_km",
    expected_time_column: str = "expected_travel_time_min",
    upper_bound_column: str = "upper_bound",
    risk_column: str = "risk_score",
) -> ClassicalBaselineResult:
    validated = _validate_candidate_table(
        route_table,
        objective_column=objective_column,
        route_id_column=route_id_column,
        path_column=path_column,
    )
    selected_row, selected_path, selected_objective = min(
        validated,
        key=lambda item: (
            item[2],
            _route_id_tie_key(str(item[0][route_id_column])),
            _path_tie_key(item[1]),
        ),
    )
    return ClassicalBaselineResult(
        method=method,
        route_id=str(selected_row[route_id_column]),
        path=selected_path,
        objective_name=objective_column,
        objective_value=selected_objective,
        distance_km=_optional_numeric(selected_row, distance_column),
        expected_travel_time_min=_optional_numeric(selected_row, expected_time_column),
        upper_bound=_optional_numeric(selected_row, upper_bound_column),
        risk_score=_optional_numeric(selected_row, risk_column),
        candidate_count=len(route_table),
    )


def select_shortest_distance_candidate(
    route_table: pd.DataFrame,
    **column_overrides: str,
) -> ClassicalBaselineResult:
    """Select minimum candidate-route distance with deterministic ties."""

    distance_column = column_overrides.pop("distance_column", "distance_km")
    return _select_candidate(
        route_table,
        method="shortest_distance_candidate",
        objective_column=distance_column,
        distance_column=distance_column,
        **column_overrides,
    )


def select_shortest_expected_time_candidate(
    route_table: pd.DataFrame,
    **column_overrides: str,
) -> ClassicalBaselineResult:
    """Select minimum predicted/expected candidate travel time."""

    expected_time_column = column_overrides.pop(
        "expected_time_column", "expected_travel_time_min"
    )
    return _select_candidate(
        route_table,
        method="shortest_expected_time_candidate",
        objective_column=expected_time_column,
        expected_time_column=expected_time_column,
        **column_overrides,
    )


def select_exact_risk_candidate(
    route_table: pd.DataFrame,
    **column_overrides: str,
) -> ClassicalBaselineResult:
    """Return the exact minimum-risk candidate, the tractable ground truth."""

    risk_column = column_overrides.pop("risk_column", "risk_score")
    return _select_candidate(
        route_table,
        method="exact_risk_argmin",
        objective_column=risk_column,
        risk_column=risk_column,
        **column_overrides,
    )


def select_upper_bound_heuristic(
    route_table: pd.DataFrame,
    **column_overrides: str,
) -> ClassicalBaselineResult:
    """Select the smallest conservative travel-time upper bound.

    This intentionally simple heuristic is interpretable: prefer the route
    with the best calibrated worst-case estimate, without adding policy
    weights or rerouting terms.
    """

    upper_bound_column = column_overrides.pop("upper_bound_column", "upper_bound")
    return _select_candidate(
        route_table,
        method="minimum_upper_bound_heuristic",
        objective_column=upper_bound_column,
        upper_bound_column=upper_bound_column,
        **column_overrides,
    )


def _validate_graph_weight(graph: nx.DiGraph, weight: str) -> None:
    if not isinstance(weight, str) or not weight.strip():
        raise ValueError("weight must be a non-empty edge-attribute name")
    for source, target, attributes in graph.edges(data=True):
        if weight not in attributes:
            raise ValueError(f"edge {source!r}->{target!r} is missing weight {weight!r}")
        try:
            _finite_number(attributes[weight], weight, positive=True)
        except ValueError as error:
            raise ValueError(f"edge {source!r}->{target!r}: {error}") from error


def _path_sum(graph: nx.DiGraph, path: Sequence[Hashable], attribute: str) -> float:
    return float(
        sum(
            float(graph.edges[source, target][attribute])
            for source, target in zip(path[:-1], path[1:], strict=True)
        )
    )


def _deterministic_dijkstra_path(
    graph: nx.DiGraph,
    origin: Hashable,
    destination: Hashable,
    weight: str,
) -> tuple[Hashable, ...]:
    """Return one shortest path without enumerating every tied solution.

    Queue labels contain both total cost and the canonical path key.  Thus an
    equal-cost update keeps the lexicographically smallest complete prefix,
    while ordinary Dijkstra pruning retains polynomial rather than
    path-enumeration behavior on highly tied grids.
    """

    insertion_order = count()
    origin_path = (origin,)
    origin_key = _path_tie_key(origin_path)
    queue: list[
        tuple[
            float,
            tuple[tuple[str, str, str], ...],
            int,
            Hashable,
            tuple[Hashable, ...],
        ]
    ] = [(0.0, origin_key, next(insertion_order), origin, origin_path)]
    best_labels: dict[
        Hashable, tuple[float, tuple[tuple[str, str, str], ...]]
    ] = {origin: (0.0, origin_key)}

    while queue:
        distance, path_key, _, node, path = heapq.heappop(queue)
        if best_labels.get(node) != (distance, path_key):
            continue
        if node == destination:
            return path

        for neighbor in sorted(graph.successors(node), key=_node_tie_key):
            candidate_distance = distance + float(graph.edges[node, neighbor][weight])
            candidate_path = (*path, neighbor)
            candidate_key = (*path_key, _node_tie_key(neighbor))
            current = best_labels.get(neighbor)
            improves = current is None or candidate_distance < current[0]
            ties_better = (
                current is not None
                and candidate_distance == current[0]
                and candidate_key < current[1]
            )
            if improves or ties_better:
                best_labels[neighbor] = (candidate_distance, candidate_key)
                heapq.heappush(
                    queue,
                    (
                        candidate_distance,
                        candidate_key,
                        next(insertion_order),
                        neighbor,
                        candidate_path,
                    ),
                )

    raise UnreachablePathError(origin, destination)


def dijkstra_shortest_distance_path(
    graph: nx.DiGraph,
    origin: Hashable,
    destination: Hashable,
    *,
    distance_weight: str = "distance_km",
    travel_time_weight: str = "base_travel_time_min",
    allow_parallel_edge_projection: bool = False,
) -> ClassicalBaselineResult:
    """Return the true graph-wide shortest-distance path.

    Both selected-path distance and travel time are reported with their proper
    units.  A GraphML network whose parallel roads were collapsed by the
    fastest-time projection is rejected by default: that projection is not the
    full multigraph distance baseline.  Callers may opt in explicitly, in
    which case the method label records that projected-graph limitation.
    """

    validate_network(graph)
    if origin not in graph:
        raise ValueError(f"origin node {origin!r} is not present in the graph")
    if destination not in graph:
        raise ValueError(f"destination node {destination!r} is not present in the graph")
    if origin == destination:
        raise ValueError("origin and destination must be distinct")
    if not isinstance(allow_parallel_edge_projection, bool):
        raise ValueError("allow_parallel_edge_projection must be a boolean")
    _validate_graph_weight(graph, distance_weight)
    _validate_graph_weight(graph, travel_time_weight)

    try:
        collapsed_count = max(
            int(graph.graph.get("loader_parallel_edges_collapsed", 0)),
            int(graph.graph.get("parallel_edges_collapsed", 0)),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("loader_parallel_edges_collapsed metadata must be an integer") from error
    is_projection = collapsed_count > 0
    if is_projection and not allow_parallel_edge_projection:
        raise ValueError(
            "shortest-distance baseline requires full parallel-edge fidelity; "
            "this graph is a fastest-base-time projection. Pass "
            "allow_parallel_edge_projection=True only for an explicitly labeled "
            "projected-graph result."
        )

    selected_path = _deterministic_dijkstra_path(
        graph, origin, destination, distance_weight
    )
    distance = _path_sum(graph, selected_path, distance_weight)
    travel_time = _path_sum(graph, selected_path, travel_time_weight)
    method = "dijkstra_shortest_distance_path"
    if is_projection:
        method = "dijkstra_shortest_distance_path_on_fastest_time_projection"
    return ClassicalBaselineResult(
        method=method,
        route_id=None,
        path=selected_path,
        objective_name=distance_weight,
        objective_value=distance,
        distance_km=distance,
        expected_travel_time_min=travel_time,
        candidate_count=None,
    )


def dijkstra_expected_time_path(
    graph: nx.DiGraph,
    origin: Hashable,
    destination: Hashable,
    *,
    weight: str = "base_travel_time_min",
) -> ClassicalBaselineResult:
    """Return a deterministic NetworkX Dijkstra expected-time path."""

    validate_network(graph)
    if origin not in graph:
        raise ValueError(f"origin node {origin!r} is not present in the graph")
    if destination not in graph:
        raise ValueError(f"destination node {destination!r} is not present in the graph")
    if origin == destination:
        raise ValueError("origin and destination must be distinct")
    _validate_graph_weight(graph, weight)

    selected_path = _deterministic_dijkstra_path(
        graph, origin, destination, weight
    )

    objective_value = _path_sum(graph, selected_path, weight)
    return ClassicalBaselineResult(
        method="dijkstra_expected_time_path",
        route_id=None,
        path=selected_path,
        objective_name=weight,
        objective_value=objective_value,
        distance_km=_path_sum(graph, selected_path, "distance_km"),
        expected_travel_time_min=objective_value,
        candidate_count=None,
    )


def baseline_results_to_table(
    results: Sequence[ClassicalBaselineResult],
) -> pd.DataFrame:
    """Convert baseline records to a stable, explicit comparison table."""

    if not results:
        raise ValueError("at least one baseline result is required")
    if any(not isinstance(result, ClassicalBaselineResult) for result in results):
        raise TypeError("results must contain ClassicalBaselineResult records")
    return pd.DataFrame(
        [result.as_record() for result in results],
        columns=RESULT_COLUMNS,
    )


def evaluate_candidate_baselines(
    route_table: pd.DataFrame,
    **column_overrides: str,
) -> pd.DataFrame:
    """Evaluate the four candidate-set baselines in a fixed order."""

    results = [
        select_shortest_distance_candidate(route_table, **column_overrides),
        select_shortest_expected_time_candidate(route_table, **column_overrides),
        select_exact_risk_candidate(route_table, **column_overrides),
        select_upper_bound_heuristic(route_table, **column_overrides),
    ]
    return baseline_results_to_table(results)
