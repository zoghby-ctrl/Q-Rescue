"""Deterministic, physically interpretable synthetic road-network families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
import json
import math
import random
from typing import Literal

import networkx as nx

from src.network_schema import EdgeData, NETWORK_SCHEMA_VERSION, NodeData, add_edge, add_node, validate_network


TopologyFamily = Literal[
    "grid",
    "radial",
    "irregular_geometric",
    "sparse_mixed",
    "dense_mixed",
]

SUPPORTED_TOPOLOGIES: tuple[str, ...] = (
    "grid",
    "radial",
    "irregular_geometric",
    "sparse_mixed",
    "dense_mixed",
)


@dataclass(frozen=True)
class SyntheticNetworkSpec:
    """Serializable experiment description for one synthetic network."""

    topology: TopologyFamily | str
    seed: int = 2026
    parameters: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.topology not in SUPPORTED_TOPOLOGIES:
            raise ValueError(
                f"unsupported topology {self.topology!r}; expected one of {SUPPORTED_TOPOLOGIES}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")
        if any(not isinstance(key, str) for key in self.parameters):
            raise ValueError("parameter names must be strings")


def _integer_at_least(value: object, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return numeric_value


def _new_graph(
    topology: str,
    seed: int,
    description: str,
    parameters: Mapping[str, int | float],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.graph.update(
        {
            "schema_version": NETWORK_SCHEMA_VERSION,
            "network_id": f"synthetic-{topology}-seed-{seed}",
            "synthetic": True,
            "topology": topology,
            "topology_family": topology,
            "topology_description": description,
            "seed": seed,
            "coordinate_system": "local_cartesian_km",
            "generator_parameters_json": json.dumps(
                dict(parameters), sort_keys=True, separators=(",", ":")
            ),
        }
    )
    return graph


def _coordinates(graph: nx.DiGraph, node_id: str) -> tuple[float, float]:
    attributes = graph.nodes[node_id]
    return float(attributes["x"]), float(attributes["y"])


_ROAD_SPEEDS_KPH = {
    "local": 32.0,
    "collector": 42.0,
    "arterial": 52.0,
    "radial": 48.0,
    "ring": 44.0,
}


def _add_bidirectional_road(
    graph: nx.DiGraph,
    first: str,
    second: str,
    rng: random.Random,
    road_class: str,
) -> None:
    first_x, first_y = _coordinates(graph, first)
    second_x, second_y = _coordinates(graph, second)
    distance_km = math.hypot(first_x - second_x, first_y - second_y)
    if distance_km <= 0.0:
        raise ValueError(f"road endpoints {first!r} and {second!r} have identical coordinates")

    nominal_speed = _ROAD_SPEEDS_KPH[road_class]
    for source, target in ((first, second), (second, first)):
        speed_kph = nominal_speed * rng.uniform(0.94, 1.06)
        free_flow_minutes = 60.0 * distance_km / speed_kph
        base_travel_time_min = free_flow_minutes * rng.uniform(1.03, 1.16)
        reliability = rng.uniform(0.92, 0.99)
        add_edge(
            graph,
            EdgeData(
                source=source,
                target=target,
                distance_km=distance_km,
                base_travel_time_min=base_travel_time_min,
                speed_kph=speed_kph,
                road_class=road_class,
                reliability=reliability,
            ),
        )


def _finalize_network(graph: nx.DiGraph, origin: str, destination: str) -> nx.DiGraph:
    if origin == destination:
        raise ValueError("origin and destination must be distinct")
    graph.graph["origin"] = origin
    graph.graph["destination"] = destination
    validate_network(graph)
    if not nx.has_path(graph, origin, destination):
        raise ValueError("generated network has no origin-to-destination path")
    return graph


def generate_grid_network(
    *,
    rows: int = 4,
    columns: int = 4,
    spacing_km: float = 0.5,
    seed: int = 2026,
) -> nx.DiGraph:
    """Generate a planned rectangular street grid with two-way roads."""

    rows = _integer_at_least(rows, "rows", 2)
    columns = _integer_at_least(columns, "columns", 2)
    spacing_km = _positive_float(spacing_km, "spacing_km")
    rng = random.Random(seed)
    parameters = {"rows": rows, "columns": columns, "spacing_km": spacing_km}
    graph = _new_graph(
        "grid",
        seed,
        "planned rectangular grid with orthogonal local and collector roads",
        parameters,
    )

    for row in range(rows):
        for column in range(columns):
            add_node(
                graph,
                NodeData(
                    node_id=f"g_{row}_{column}",
                    x=column * spacing_km,
                    y=row * spacing_km,
                ),
            )

    for row in range(rows):
        for column in range(columns):
            current = f"g_{row}_{column}"
            if column + 1 < columns:
                road_class = "collector" if row in {0, rows - 1} else "local"
                _add_bidirectional_road(graph, current, f"g_{row}_{column + 1}", rng, road_class)
            if row + 1 < rows:
                road_class = "collector" if column in {0, columns - 1} else "local"
                _add_bidirectional_road(graph, current, f"g_{row + 1}_{column}", rng, road_class)

    return _finalize_network(graph, "g_0_0", f"g_{rows - 1}_{columns - 1}")


def generate_radial_network(
    *,
    spokes: int = 6,
    rings: int = 3,
    ring_spacing_km: float = 0.6,
    seed: int = 2026,
) -> nx.DiGraph:
    """Generate a hub-and-ring road system with radial and circumferential roads."""

    spokes = _integer_at_least(spokes, "spokes", 4)
    rings = _integer_at_least(rings, "rings", 1)
    ring_spacing_km = _positive_float(ring_spacing_km, "ring_spacing_km")
    rng = random.Random(seed)
    parameters = {
        "spokes": spokes,
        "rings": rings,
        "ring_spacing_km": ring_spacing_km,
    }
    graph = _new_graph(
        "radial",
        seed,
        "central hub joined by radial spokes and concentric ring roads",
        parameters,
    )
    add_node(graph, NodeData("center", x=0.0, y=0.0))

    for ring in range(1, rings + 1):
        radius = ring * ring_spacing_km
        for spoke in range(spokes):
            angle = 2.0 * math.pi * spoke / spokes
            add_node(
                graph,
                NodeData(
                    node_id=f"r_{ring}_s_{spoke}",
                    x=radius * math.cos(angle),
                    y=radius * math.sin(angle),
                ),
            )

    road_pairs: set[tuple[str, str, str]] = set()
    for spoke in range(spokes):
        road_pairs.add(("center", f"r_1_s_{spoke}", "radial"))
        for ring in range(1, rings):
            road_pairs.add((f"r_{ring}_s_{spoke}", f"r_{ring + 1}_s_{spoke}", "radial"))
    for ring in range(1, rings + 1):
        for spoke in range(spokes):
            first = f"r_{ring}_s_{spoke}"
            second = f"r_{ring}_s_{(spoke + 1) % spokes}"
            ordered = tuple(sorted((first, second)))
            road_pairs.add((ordered[0], ordered[1], "ring"))

    for first, second, road_class in sorted(road_pairs):
        _add_bidirectional_road(graph, first, second, rng, road_class)

    origin = f"r_{rings}_s_0"
    destination = f"r_{rings}_s_{spokes // 2}"
    return _finalize_network(graph, origin, destination)


def _random_points(
    node_count: int,
    width_km: float,
    height_km: float,
    rng: random.Random,
) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    minimum_separation = min(width_km, height_km) / (8.0 * math.sqrt(node_count))
    attempts = 0
    while len(points) < node_count:
        attempts += 1
        if attempts > node_count * 1000:
            raise RuntimeError("could not place separated geometric nodes")
        candidate = (rng.uniform(0.0, width_km), rng.uniform(0.0, height_km))
        if all(math.dist(candidate, point) >= minimum_separation for point in points.values()):
            points[f"n_{len(points):03d}"] = candidate
    return points


def _jittered_grid_points(
    node_count: int,
    width_km: float,
    height_km: float,
    rng: random.Random,
) -> dict[str, tuple[float, float]]:
    rows = max(2, int(math.floor(math.sqrt(node_count))))
    columns = int(math.ceil(node_count / rows))
    cell_width = width_km / columns
    cell_height = height_km / rows
    points: dict[str, tuple[float, float]] = {}
    for index in range(node_count):
        row, column = divmod(index, columns)
        jitter_x = rng.uniform(-0.20, 0.20) * cell_width
        jitter_y = rng.uniform(-0.20, 0.20) * cell_height
        points[f"n_{index:03d}"] = (
            (column + 0.5) * cell_width + jitter_x,
            (row + 0.5) * cell_height + jitter_y,
        )
    return points


def _local_road_pairs(
    points: Mapping[str, tuple[float, float]],
    local_neighbors: int,
) -> set[tuple[str, str]]:
    complete_graph = nx.Graph()
    complete_graph.add_nodes_from(points)
    for first, second in combinations(sorted(points), 2):
        complete_graph.add_edge(first, second, distance=math.dist(points[first], points[second]))

    minimum_spanning_tree = nx.minimum_spanning_tree(complete_graph, weight="distance")
    road_pairs = {tuple(sorted((first, second))) for first, second in minimum_spanning_tree.edges}

    for first in sorted(points):
        neighbors = sorted(
            (second for second in points if second != first),
            key=lambda second: (math.dist(points[first], points[second]), second),
        )
        for second in neighbors[:local_neighbors]:
            road_pairs.add(tuple(sorted((first, second))))
    return road_pairs


def _farthest_pair(points: Mapping[str, tuple[float, float]]) -> tuple[str, str]:
    candidates = combinations(sorted(points), 2)
    return max(candidates, key=lambda pair: (math.dist(points[pair[0]], points[pair[1]]), pair))


def _generate_geometric_network(
    *,
    topology: str,
    points: Mapping[str, tuple[float, float]],
    local_neighbors: int,
    seed: int,
    width_km: float,
    height_km: float,
    description: str,
) -> nx.DiGraph:
    local_neighbors = _integer_at_least(local_neighbors, "local_neighbors", 1)
    if local_neighbors >= len(points):
        raise ValueError("local_neighbors must be smaller than node_count")
    parameters = {
        "node_count": len(points),
        "width_km": width_km,
        "height_km": height_km,
        "local_neighbors": local_neighbors,
    }
    graph = _new_graph(topology, seed, description, parameters)
    for node_id, (x, y) in points.items():
        add_node(graph, NodeData(node_id=node_id, x=x, y=y))

    rng = random.Random(seed + 1_000_003)
    typical_spacing = math.sqrt(width_km * height_km / len(points))
    for first, second in sorted(_local_road_pairs(points, local_neighbors)):
        distance = math.dist(points[first], points[second])
        road_class = "collector" if distance > 1.55 * typical_spacing else "local"
        _add_bidirectional_road(graph, first, second, rng, road_class)

    origin, destination = _farthest_pair(points)
    return _finalize_network(graph, origin, destination)


def generate_irregular_geometric_network(
    *,
    node_count: int = 24,
    width_km: float = 5.0,
    height_km: float = 4.0,
    local_neighbors: int = 3,
    seed: int = 2026,
) -> nx.DiGraph:
    """Generate an organic proximity network over irregularly placed junctions."""

    node_count = _integer_at_least(node_count, "node_count", 6)
    width_km = _positive_float(width_km, "width_km")
    height_km = _positive_float(height_km, "height_km")
    rng = random.Random(seed)
    points = _random_points(node_count, width_km, height_km, rng)
    return _generate_geometric_network(
        topology="irregular_geometric",
        points=points,
        local_neighbors=local_neighbors,
        seed=seed,
        width_km=width_km,
        height_km=height_km,
        description="organic geometric network using short proximity roads and a connected backbone",
    )


def _generate_mixed_network(
    *,
    topology: str,
    node_count: int,
    width_km: float,
    height_km: float,
    local_neighbors: int,
    seed: int,
    description: str,
) -> nx.DiGraph:
    node_count = _integer_at_least(node_count, "node_count", 9)
    width_km = _positive_float(width_km, "width_km")
    height_km = _positive_float(height_km, "height_km")
    rng = random.Random(seed)
    points = _jittered_grid_points(node_count, width_km, height_km, rng)
    return _generate_geometric_network(
        topology=topology,
        points=points,
        local_neighbors=local_neighbors,
        seed=seed,
        width_km=width_km,
        height_km=height_km,
        description=description,
    )


def generate_sparse_mixed_network(
    *,
    node_count: int = 25,
    width_km: float = 5.0,
    height_km: float = 5.0,
    local_neighbors: int = 2,
    seed: int = 2026,
) -> nx.DiGraph:
    """Generate a lightly connected, partly planned and partly irregular network."""

    return _generate_mixed_network(
        topology="sparse_mixed",
        node_count=node_count,
        width_km=width_km,
        height_km=height_km,
        local_neighbors=local_neighbors,
        seed=seed,
        description="jittered planned layout with a sparse local-road backbone",
    )


def generate_dense_mixed_network(
    *,
    node_count: int = 25,
    width_km: float = 5.0,
    height_km: float = 5.0,
    local_neighbors: int = 4,
    seed: int = 2026,
) -> nx.DiGraph:
    """Generate a connected mixed-layout network with greater local redundancy."""

    return _generate_mixed_network(
        topology="dense_mixed",
        node_count=node_count,
        width_km=width_km,
        height_km=height_km,
        local_neighbors=local_neighbors,
        seed=seed,
        description="jittered planned layout with dense, locally plausible road redundancy",
    )


_GENERATORS: dict[str, Callable[..., nx.DiGraph]] = {
    "grid": generate_grid_network,
    "radial": generate_radial_network,
    "irregular_geometric": generate_irregular_geometric_network,
    "sparse_mixed": generate_sparse_mixed_network,
    "dense_mixed": generate_dense_mixed_network,
}


def generate_synthetic_network(spec: SyntheticNetworkSpec) -> nx.DiGraph:
    """Generate one validated family from a dataclass experiment spec."""

    parameters = dict(spec.parameters)
    if "seed" in parameters:
        raise ValueError("seed belongs in SyntheticNetworkSpec.seed, not parameters")
    generator = _GENERATORS[str(spec.topology)]
    return generator(seed=spec.seed, **parameters)
