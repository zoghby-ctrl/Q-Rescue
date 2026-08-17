"""Transparent, location-independent candidate-route generation and summaries."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import combinations
import math
from numbers import Real

import networkx as nx
import pandas as pd

from src.network_schema import NetworkSchemaError, validate_network


class InsufficientRoutesError(ValueError):
    """Raised when a graph contains fewer than the requested simple paths."""

    def __init__(self, requested_k: int, available_count: int) -> None:
        self.requested_k = requested_k
        self.available_count = available_count
        super().__init__(
            f"requested K={requested_k} candidate routes, but only "
            f"{available_count} loopless origin-to-destination routes exist"
        )


def _positive_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return numeric_value


@dataclass(frozen=True)
class CandidateRoute:
    """One ordered, loopless candidate route and its additive summaries.

    Route IDs use decision-variable order: the first candidate is ``x_0``, the
    second is ``x_1``, and so on.
    """

    route_id: str
    nodes: tuple[Hashable, ...]
    edges: tuple[tuple[Hashable, Hashable], ...]
    distance_km: float
    base_travel_time_min: float
    expected_travel_time_min: float
    edge_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(tuple(edge) for edge in self.edges))

        if not isinstance(self.route_id, str):
            raise ValueError("route_id must have canonical form x_0, x_1, ...")
        suffix = self.route_id.removeprefix("x_")
        if not suffix.isdigit() or self.route_id != f"x_{int(suffix)}":
            raise ValueError("route_id must have canonical form x_0, x_1, ...")
        if len(self.nodes) < 2:
            raise ValueError("a candidate route must contain at least two nodes")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("candidate routes must be loopless and cannot repeat nodes")

        expected_edges = tuple(zip(self.nodes[:-1], self.nodes[1:]))
        if self.edges != expected_edges:
            raise ValueError("edges must match consecutive ordered nodes")
        if (
            isinstance(self.edge_count, bool)
            or not isinstance(self.edge_count, int)
            or self.edge_count != len(self.edges)
        ):
            raise ValueError("edge_count must equal the number of route edges")

        _positive_finite_number(self.distance_km, "distance_km")
        _positive_finite_number(self.base_travel_time_min, "base_travel_time_min")
        _positive_finite_number(self.expected_travel_time_min, "expected_travel_time_min")

    @property
    def ordered_nodes(self) -> tuple[Hashable, ...]:
        """Alias emphasizing that ``nodes`` retain origin-to-destination order."""

        return self.nodes

    @property
    def route_index(self) -> int:
        """Return the zero-based decision-variable index encoded by ``route_id``."""

        return int(self.route_id.removeprefix("x_"))


def _node_tie_key(node_id: Hashable) -> tuple[str, str, str]:
    """Return a stable ordering key for ordinary string/numeric node IDs."""

    node_type = type(node_id)
    return node_type.__module__, node_type.__qualname__, repr(node_id)


def _path_tie_key(path: Sequence[Hashable]) -> tuple[tuple[str, str, str], ...]:
    return tuple(_node_tie_key(node_id) for node_id in path)


def _ordered_graph_copy(graph: nx.DiGraph) -> nx.DiGraph:
    """Copy a graph with canonical insertion order for deterministic tie handling."""

    ordered_graph = nx.DiGraph()
    ordered_graph.graph.update(graph.graph)
    for node_id, attributes in sorted(graph.nodes(data=True), key=lambda item: _node_tie_key(item[0])):
        ordered_graph.add_node(node_id, **dict(attributes))
    ordered_edges = sorted(
        graph.edges(data=True),
        key=lambda item: (_node_tie_key(item[0]), _node_tie_key(item[1])),
    )
    for source, target, attributes in ordered_edges:
        ordered_graph.add_edge(source, target, **dict(attributes))
    return ordered_graph


def _validate_weight_attribute(graph: nx.DiGraph, attribute_name: str) -> None:
    if not isinstance(attribute_name, str) or not attribute_name.strip():
        raise ValueError("edge-weight attribute names must be non-empty strings")
    for source, target, attributes in graph.edges(data=True):
        if attribute_name not in attributes:
            raise ValueError(
                f"edge {source!r}->{target!r} is missing weight attribute {attribute_name!r}"
            )
        try:
            _positive_finite_number(attributes[attribute_name], attribute_name)
        except ValueError as error:
            raise ValueError(f"edge {source!r}->{target!r}: {error}") from error


def _path_attribute_sum(
    graph: nx.DiGraph,
    path: Sequence[Hashable],
    attribute_name: str,
) -> float:
    return float(
        sum(
            float(graph.edges[source, target][attribute_name])
            for source, target in zip(path[:-1], path[1:])
        )
    )


def generate_candidate_routes(
    graph: nx.DiGraph,
    origin: Hashable,
    destination: Hashable,
    k: int,
    *,
    weight: str = "base_travel_time_min",
    expected_time_attribute: str = "base_travel_time_min",
    require_k: bool = True,
) -> list[CandidateRoute]:
    """Generate the K shortest loopless routes using NetworkX transparently.

    ``networkx.shortest_simple_paths`` supplies paths in nondecreasing total
    weight. All paths tied at the Kth cost are consumed and then sorted by a
    canonical node-ID representation; therefore equal-cost route IDs do not
    depend on graph insertion order. Custom node classes should provide a
    stable ``repr`` when cross-process tie reproducibility is required.

    By default the function returns exactly K routes or raises
    :class:`InsufficientRoutesError`. Set ``require_k=False`` explicitly to
    accept all available paths when fewer than K exist.
    """

    validate_network(graph)
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not isinstance(require_k, bool):
        raise ValueError("require_k must be a boolean")
    if origin not in graph:
        raise ValueError(f"origin node {origin!r} is not present in the graph")
    if destination not in graph:
        raise ValueError(f"destination node {destination!r} is not present in the graph")
    if origin == destination:
        raise ValueError("origin and destination must be distinct")

    _validate_weight_attribute(graph, weight)
    _validate_weight_attribute(graph, expected_time_attribute)
    ordered_graph = _ordered_graph_copy(graph)

    path_records: list[tuple[float, tuple[Hashable, ...]]] = []
    kth_cost: float | None = None
    try:
        path_generator = nx.shortest_simple_paths(
            ordered_graph,
            source=origin,
            target=destination,
            weight=weight,
        )
        for path in path_generator:
            path_tuple = tuple(path)
            path_cost = _path_attribute_sum(ordered_graph, path_tuple, weight)
            if kth_cost is not None and path_cost > kth_cost and not math.isclose(
                path_cost, kth_cost, rel_tol=1e-12, abs_tol=1e-12
            ):
                break
            path_records.append((path_cost, path_tuple))
            if len(path_records) == k:
                kth_cost = path_cost
    except nx.NetworkXNoPath:
        path_records = []

    if len(path_records) < k and require_k:
        raise InsufficientRoutesError(requested_k=k, available_count=len(path_records))

    ordered_paths = sorted(path_records, key=lambda item: (item[0], _path_tie_key(item[1])))[:k]
    routes: list[CandidateRoute] = []
    for route_index, (_, nodes) in enumerate(ordered_paths):
        if len(set(nodes)) != len(nodes):
            raise NetworkSchemaError("shortest_simple_paths returned a route with repeated nodes")
        edges = tuple(zip(nodes[:-1], nodes[1:]))
        routes.append(
            CandidateRoute(
                route_id=f"x_{route_index}",
                nodes=nodes,
                edges=edges,
                distance_km=_path_attribute_sum(graph, nodes, "distance_km"),
                base_travel_time_min=_path_attribute_sum(
                    graph, nodes, "base_travel_time_min"
                ),
                expected_travel_time_min=_path_attribute_sum(
                    graph, nodes, expected_time_attribute
                ),
                edge_count=len(edges),
            )
        )
    return routes


_ROUTE_TABLE_COLUMNS = [
    "route_id",
    "ordered_nodes",
    "edges",
    "distance_km",
    "base_travel_time_min",
    "expected_travel_time_min",
    "edge_count",
]


def routes_to_table(routes: Sequence[CandidateRoute]) -> pd.DataFrame:
    """Return route records in their supplied decision-variable order."""

    rows = [
        {
            "route_id": route.route_id,
            "ordered_nodes": route.nodes,
            "edges": route.edges,
            "distance_km": route.distance_km,
            "base_travel_time_min": route.base_travel_time_min,
            "expected_travel_time_min": route.expected_travel_time_min,
            "edge_count": route.edge_count,
        }
        for route in routes
    ]
    return pd.DataFrame(rows, columns=_ROUTE_TABLE_COLUMNS)


def route_edge_set(route: CandidateRoute) -> frozenset[tuple[Hashable, Hashable]]:
    """Return the directed edge set used by one route."""

    return frozenset(route.edges)


def edge_jaccard_similarity(first: CandidateRoute, second: CandidateRoute) -> float:
    """Return directed-edge Jaccard similarity in the interval [0, 1]."""

    first_edges = route_edge_set(first)
    second_edges = route_edge_set(second)
    union = first_edges | second_edges
    if not union:
        return 1.0
    return len(first_edges & second_edges) / len(union)


def pairwise_route_diversity_table(routes: Sequence[CandidateRoute]) -> pd.DataFrame:
    """Return shared-edge, Jaccard-similarity, and diversity values by route pair."""

    columns = [
        "first_route_id",
        "second_route_id",
        "shared_edge_count",
        "union_edge_count",
        "jaccard_similarity",
        "jaccard_diversity",
    ]
    rows: list[dict[str, int | float | str]] = []
    for first, second in combinations(routes, 2):
        first_edges = route_edge_set(first)
        second_edges = route_edge_set(second)
        shared_count = len(first_edges & second_edges)
        union_count = len(first_edges | second_edges)
        similarity = edge_jaccard_similarity(first, second)
        rows.append(
            {
                "first_route_id": first.route_id,
                "second_route_id": second.route_id,
                "shared_edge_count": shared_count,
                "union_edge_count": union_count,
                "jaccard_similarity": similarity,
                "jaccard_diversity": 1.0 - similarity,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_route_diversity(routes: Sequence[CandidateRoute]) -> dict[str, int | float]:
    """Summarize directed-edge diversity across a nonempty candidate set."""

    if not routes:
        raise ValueError("at least one candidate route is required")
    pairwise_table = pairwise_route_diversity_table(routes)
    unique_edges = set().union(*(route_edge_set(route) for route in routes))
    if pairwise_table.empty:
        mean_similarity = 1.0
        mean_diversity = 0.0
        minimum_diversity = 0.0
        maximum_diversity = 0.0
    else:
        mean_similarity = float(pairwise_table["jaccard_similarity"].mean())
        mean_diversity = float(pairwise_table["jaccard_diversity"].mean())
        minimum_diversity = float(pairwise_table["jaccard_diversity"].min())
        maximum_diversity = float(pairwise_table["jaccard_diversity"].max())
    return {
        "route_count": len(routes),
        "pair_count": len(pairwise_table),
        "unique_directed_edge_count": len(unique_edges),
        "mean_jaccard_similarity": mean_similarity,
        "mean_jaccard_diversity": mean_diversity,
        "minimum_pairwise_diversity": minimum_diversity,
        "maximum_pairwise_diversity": maximum_diversity,
    }
