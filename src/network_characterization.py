"""Interpretable, OD-focused structural characterization for road networks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass
import math
from statistics import fmean, pstdev

import networkx as nx
import pandas as pd

from src.candidate_routes import generate_candidate_routes, summarize_route_diversity
from src.network_schema import validate_network


_DENSITY_CAVEAT = (
    "Simple directed density excludes self-loops and is strongly size-dependent; "
    "compare it only among similarly sized networks."
)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentiles require at least one value")
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _distribution_stats(values: Sequence[float]) -> tuple[float, float, float]:
    mean = fmean(values)
    standard_deviation = pstdev(values)
    coefficient_of_variation = standard_deviation / mean if mean > 0.0 else 0.0
    return mean, standard_deviation, coefficient_of_variation


@dataclass(frozen=True)
class NetworkCharacteristics:
    """A compact set of road-structure and OD route-diversity measurements."""

    network_id: str
    topology_family: str
    node_count: int
    directed_edge_count: int
    mean_in_degree: float
    mean_out_degree: float
    mean_undirected_intersection_degree: float
    intersection_degree_min: int
    intersection_degree_p25: float
    intersection_degree_median: float
    intersection_degree_p75: float
    intersection_degree_max: int
    intersection_degree_distribution: dict[int, int]
    directed_simple_density: float
    density_caveat: str
    mean_edge_length_km: float
    std_edge_length_km: float
    cv_edge_length: float
    mean_base_travel_time_min: float
    std_base_travel_time_min: float
    cv_base_travel_time: float
    origin: Hashable
    destination: Hashable
    od_shortest_distance_km: float
    od_shortest_base_travel_time_min: float
    requested_route_k: int
    available_route_count: int
    route_k_complete: bool
    route_unique_directed_edge_count: int
    route_mean_jaccard_similarity: float
    route_mean_jaccard_diversity: float
    route_minimum_pairwise_diversity: float
    route_maximum_pairwise_diversity: float

    def as_record(self) -> dict[str, object]:
        """Return a table-ready record without discarding the degree histogram."""

        return asdict(self)


def characterize_network(
    graph: nx.DiGraph,
    *,
    origin: Hashable | None = None,
    destination: Hashable | None = None,
    route_k: int = 4,
    require_route_k: bool = False,
    route_weight: str = "base_travel_time_min",
) -> NetworkCharacteristics:
    """Characterize one network without any expensive all-pairs computation."""

    validate_network(graph)
    if graph.number_of_edges() == 0:
        raise ValueError("network characterization requires at least one directed edge")
    if origin is None:
        origin = graph.graph.get("origin")
    if destination is None:
        destination = graph.graph.get("destination")
    if origin is None or origin not in graph:
        raise ValueError("a valid origin must be supplied or stored in graph metadata")
    if destination is None or destination not in graph:
        raise ValueError("a valid destination must be supplied or stored in graph metadata")
    if origin == destination:
        raise ValueError("origin and destination must be distinct")

    try:
        shortest_distance = float(
            nx.shortest_path_length(graph, origin, destination, weight="distance_km")
        )
        shortest_base_time = float(
            nx.shortest_path_length(
                graph, origin, destination, weight="base_travel_time_min"
            )
        )
    except nx.NetworkXNoPath as error:
        raise ValueError("origin and destination are not connected by a directed path") from error

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    in_degrees = [degree for _, degree in graph.in_degree()]
    out_degrees = [degree for _, degree in graph.out_degree()]
    undirected_graph = graph.to_undirected(as_view=False)
    undirected_graph.remove_edges_from(nx.selfloop_edges(undirected_graph))
    intersection_degrees = [degree for _, degree in undirected_graph.degree()]
    degree_distribution = dict(sorted(Counter(intersection_degrees).items()))

    edge_lengths = [float(attributes["distance_km"]) for _, _, attributes in graph.edges(data=True)]
    base_times = [
        float(attributes["base_travel_time_min"])
        for _, _, attributes in graph.edges(data=True)
    ]
    mean_length, std_length, cv_length = _distribution_stats(edge_lengths)
    mean_time, std_time, cv_time = _distribution_stats(base_times)

    routes = generate_candidate_routes(
        graph,
        origin,
        destination,
        route_k,
        weight=route_weight,
        require_k=require_route_k,
    )
    route_diversity = summarize_route_diversity(routes)

    non_self_edge_count = sum(1 for source, target in graph.edges if source != target)
    density_denominator = node_count * (node_count - 1)
    directed_density = non_self_edge_count / density_denominator if density_denominator else 0.0

    return NetworkCharacteristics(
        network_id=str(graph.graph.get("network_id", "unnamed-network")),
        topology_family=str(graph.graph.get("topology_family", "unspecified")),
        node_count=node_count,
        directed_edge_count=edge_count,
        mean_in_degree=fmean(in_degrees),
        mean_out_degree=fmean(out_degrees),
        mean_undirected_intersection_degree=fmean(intersection_degrees),
        intersection_degree_min=min(intersection_degrees),
        intersection_degree_p25=_percentile(intersection_degrees, 0.25),
        intersection_degree_median=_percentile(intersection_degrees, 0.50),
        intersection_degree_p75=_percentile(intersection_degrees, 0.75),
        intersection_degree_max=max(intersection_degrees),
        intersection_degree_distribution=degree_distribution,
        directed_simple_density=directed_density,
        density_caveat=_DENSITY_CAVEAT,
        mean_edge_length_km=mean_length,
        std_edge_length_km=std_length,
        cv_edge_length=cv_length,
        mean_base_travel_time_min=mean_time,
        std_base_travel_time_min=std_time,
        cv_base_travel_time=cv_time,
        origin=origin,
        destination=destination,
        od_shortest_distance_km=shortest_distance,
        od_shortest_base_travel_time_min=shortest_base_time,
        requested_route_k=route_k,
        available_route_count=len(routes),
        route_k_complete=len(routes) == route_k,
        route_unique_directed_edge_count=int(route_diversity["unique_directed_edge_count"]),
        route_mean_jaccard_similarity=float(route_diversity["mean_jaccard_similarity"]),
        route_mean_jaccard_diversity=float(route_diversity["mean_jaccard_diversity"]),
        route_minimum_pairwise_diversity=float(
            route_diversity["minimum_pairwise_diversity"]
        ),
        route_maximum_pairwise_diversity=float(
            route_diversity["maximum_pairwise_diversity"]
        ),
    )


def characterizations_to_table(
    characterizations: Sequence[NetworkCharacteristics],
) -> pd.DataFrame:
    """Return one reproducible row per characterized network."""

    return pd.DataFrame([characterization.as_record() for characterization in characterizations])
