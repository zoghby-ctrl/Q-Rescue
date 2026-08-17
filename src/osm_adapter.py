"""Small, reproducible OpenStreetMap road extracts for offline experiments.

The routing algorithms never branch on a place name.  Region labels and
bounding boxes are experiment data; this module converts any compatible
Overpass road response into the canonical :class:`networkx.DiGraph` schema.

Downloaded extracts are deliberately small.  They represent real road
*topology* only.  Travel times are derived from mapped/default speeds and later
traffic is controlled simulation, never live traffic.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import networkx as nx

from src.candidate_routes import generate_candidate_routes
from src.network_schema import NETWORK_SCHEMA_VERSION, EdgeData, NodeData, add_edge, add_node, validate_network


DEFAULT_OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OSM_ATTRIBUTION = "Map data © OpenStreetMap contributors; ODbL 1.0"
OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"

DEFAULT_HIGHWAY_SPEEDS_KPH: dict[str, float] = {
    "motorway": 90.0,
    "trunk": 70.0,
    "primary": 55.0,
    "secondary": 45.0,
    "tertiary": 38.0,
    "unclassified": 32.0,
    "residential": 28.0,
    "living_street": 15.0,
}

DEFAULT_RELIABILITY_BY_CLASS: dict[str, float] = {
    "motorway": 0.97,
    "trunk": 0.96,
    "primary": 0.95,
    "secondary": 0.95,
    "tertiary": 0.94,
    "unclassified": 0.93,
    "residential": 0.93,
    "living_street": 0.92,
}


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Overpass bounding box in south, west, north, east order."""

    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        values = (self.south, self.west, self.north, self.east)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("bounding-box coordinates must be finite")
        if not -90.0 <= self.south < self.north <= 90.0:
            raise ValueError("bounding-box latitude order/range is invalid")
        if not -180.0 <= self.west < self.east <= 180.0:
            raise ValueError("bounding-box longitude order/range is invalid")

    def overpass_text(self) -> str:
        return ",".join(f"{value:.7f}" for value in (self.south, self.west, self.north, self.east))


@dataclass(frozen=True, slots=True)
class OSMExtractionSpec:
    """Serializable description of one bounded road-topology extract."""

    network_id: str
    region_label: str
    selection_rationale: str
    bounding_box: BoundingBox
    topology_class: str = "real_osm_unclassified"
    highway_classes: tuple[str, ...] = tuple(DEFAULT_HIGHWAY_SPEEDS_KPH)
    candidate_k: int = 4

    def __post_init__(self) -> None:
        if not self.network_id.strip() or not self.region_label.strip():
            raise ValueError("network_id and region_label must be non-empty")
        if not self.selection_rationale.strip():
            raise ValueError("selection_rationale must be non-empty")
        if not self.topology_class.strip():
            raise ValueError("topology_class must be non-empty")
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if not self.highway_classes:
            raise ValueError("at least one highway class is required")
        unknown = sorted(set(self.highway_classes) - set(DEFAULT_HIGHWAY_SPEEDS_KPH))
        if unknown:
            raise ValueError(f"unsupported highway classes: {unknown}")


def build_overpass_query(spec: OSMExtractionSpec) -> str:
    """Build the exact deterministic QL text stored with the cached extract."""

    highway_pattern = "|".join(re.escape(value) for value in spec.highway_classes)
    bbox = spec.bounding_box.overpass_text()
    return (
        "[out:json][timeout:90];"
        f'way["highway"~"^({highway_pattern})$"]({bbox});'
        "(._;>;);out body;"
    )


def fetch_overpass_json(
    spec: OSMExtractionSpec,
    *,
    endpoint: str = DEFAULT_OVERPASS_ENDPOINT,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Fetch once from Overpass; callers should cache the resulting GraphML."""

    query = build_overpass_query(spec)
    request = Request(
        endpoint,
        data=urlencode({"data": query}).encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "QRescue-research/0.2 (offline topology cache)",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - configured HTTPS endpoint
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        raise ValueError("Overpass response does not contain an elements list")
    return payload


_SPEED_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mph|km/h|kph|kmh)?\s*$", re.I)


def _speed_kph(tags: Mapping[str, object], highway: str) -> float:
    value = tags.get("maxspeed")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if math.isfinite(parsed) and parsed > 0.0:
            return parsed
    if isinstance(value, str):
        match = _SPEED_RE.fullmatch(value)
        if match:
            parsed = float(match.group("value"))
            if match.group("unit") and match.group("unit").lower() == "mph":
                parsed *= 1.609344
            if parsed > 0.0:
                return parsed
    return DEFAULT_HIGHWAY_SPEEDS_KPH[highway]


def _haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def _oneway_direction(tags: Mapping[str, object]) -> str:
    value = str(tags.get("oneway", "")).strip().lower()
    if value in {"-1", "reverse"}:
        return "reverse"
    if value in {"yes", "true", "1"} or str(tags.get("junction", "")).lower() == "roundabout":
        return "forward"
    return "both"


def _largest_strong_component(graph: nx.DiGraph) -> set[str]:
    components = list(nx.strongly_connected_components(graph))
    if not components:
        raise ValueError("OSM extract contains no strongly connected road component")
    return set(max(components, key=lambda nodes: (len(nodes), tuple(sorted(map(str, nodes))))))


def _anchor_nodes(graph: nx.DiGraph, limit: int = 28) -> list[str]:
    nodes = sorted(map(str, graph.nodes))
    if len(nodes) <= limit:
        return nodes
    ranked: list[str] = []
    scoring = (
        lambda node: float(graph.nodes[node]["x"]),
        lambda node: float(graph.nodes[node]["y"]),
        lambda node: float(graph.nodes[node]["x"]) + float(graph.nodes[node]["y"]),
        lambda node: float(graph.nodes[node]["x"]) - float(graph.nodes[node]["y"]),
    )
    for score in scoring:
        ordered = sorted(nodes, key=lambda node: (score(node), node))
        ranked.extend((ordered[0], ordered[-1]))
    step = max(1, len(nodes) // max(1, limit - len(set(ranked))))
    ranked.extend(nodes[::step])
    return list(dict.fromkeys(ranked))[:limit]


def select_reproducible_od_pair(
    graph: nx.DiGraph,
    *,
    candidate_k: int = 4,
    maximum_pairs_checked: int = 120,
) -> tuple[str, str, int]:
    """Choose a geographically separated OD pair with as many as K routes."""

    validate_network(graph)
    anchors = _anchor_nodes(graph)
    pairs: list[tuple[float, str, str]] = []
    for index, first in enumerate(anchors):
        for second in anchors[index + 1 :]:
            first_latlon = (float(graph.nodes[first]["y"]), float(graph.nodes[first]["x"]))
            second_latlon = (float(graph.nodes[second]["y"]), float(graph.nodes[second]["x"]))
            pairs.append((_haversine_km(first_latlon, second_latlon), first, second))
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    best: tuple[int, float, str, str] | None = None
    for distance, first, second in pairs[:maximum_pairs_checked]:
        routes = generate_candidate_routes(
            graph,
            first,
            second,
            candidate_k,
            weight="base_travel_time_min",
            require_k=False,
        )
        available = len(routes)
        candidate = (available, distance, first, second)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        if available == candidate_k:
            return first, second, available
    if best is None or best[0] == 0:
        raise ValueError("could not find a reachable OD pair in the canonical OSM component")
    return best[2], best[3], best[0]


def overpass_json_to_canonical_graph(
    payload: Mapping[str, object],
    spec: OSMExtractionSpec,
) -> nx.DiGraph:
    """Convert Overpass JSON to the canonical schema and choose a fixed OD."""

    elements = payload.get("elements")
    if not isinstance(elements, Sequence) or isinstance(elements, (str, bytes)):
        raise ValueError("Overpass payload must contain an elements sequence")
    coordinates: dict[str, tuple[float, float]] = {}
    ways: list[Mapping[str, object]] = []
    for element in elements:
        if not isinstance(element, Mapping):
            continue
        element_type = element.get("type")
        if element_type == "node" and element.get("id") is not None:
            try:
                latitude = float(element["lat"])
                longitude = float(element["lon"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("OSM nodes must contain numeric lat/lon") from error
            coordinates[str(element["id"])] = (latitude, longitude)
        elif element_type == "way":
            ways.append(element)

    alternatives: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    skipped_segments = 0
    for way in ways:
        tags = way.get("tags") if isinstance(way.get("tags"), Mapping) else {}
        assert isinstance(tags, Mapping)
        highway = str(tags.get("highway", ""))
        if highway not in spec.highway_classes:
            continue
        node_ids = [str(value) for value in way.get("nodes", [])]
        direction = _oneway_direction(tags)
        speed = _speed_kph(tags, highway)
        for first, second in zip(node_ids, node_ids[1:]):
            if first not in coordinates or second not in coordinates or first == second:
                skipped_segments += 1
                continue
            distance = _haversine_km(coordinates[first], coordinates[second])
            if distance <= 0.0:
                skipped_segments += 1
                continue
            record = {
                "distance_km": distance,
                "base_travel_time_min": 60.0 * distance / speed,
                "speed_kph": speed,
                "road_class": highway,
                "reliability": DEFAULT_RELIABILITY_BY_CLASS[highway],
                "osm_way_id": str(way.get("id", "unknown")),
                "osm_name": str(tags.get("name", "")),
            }
            if direction in {"both", "forward"}:
                alternatives[(first, second)].append(record)
            if direction in {"both", "reverse"}:
                alternatives[(second, first)].append(dict(record))
    if not alternatives:
        raise ValueError("Overpass extract contains no supported drivable road segments")

    full_graph = nx.DiGraph()
    used_nodes = {node for edge in alternatives for node in edge}
    for node_id in sorted(used_nodes):
        latitude, longitude = coordinates[node_id]
        add_node(full_graph, NodeData(node_id=node_id, x=longitude, y=latitude))

    collapsed_count = 0
    for (source, target), records in sorted(alternatives.items()):
        selected = min(
            records,
            key=lambda row: (
                float(row["base_travel_time_min"]),
                float(row["distance_km"]),
                str(row["osm_way_id"]),
            ),
        )
        add_edge(
            full_graph,
            EdgeData(
                source=source,
                target=target,
                distance_km=float(selected["distance_km"]),
                base_travel_time_min=float(selected["base_travel_time_min"]),
                speed_kph=float(selected["speed_kph"]),
                road_class=str(selected["road_class"]),
                reliability=float(selected["reliability"]),
            ),
        )
        full_graph.edges[source, target].update(
            {
                "osm_way_id": str(selected["osm_way_id"]),
                "osm_name": str(selected["osm_name"]),
                "source_parallel_edge_count": len(records),
                "source_min_distance_km": min(float(row["distance_km"]) for row in records),
                "source_max_reliability": max(float(row["reliability"]) for row in records),
                "parallel_edge_policy": "minimum_base_travel_time",
            }
        )
        collapsed_count += len(records) - 1

    component_nodes = _largest_strong_component(full_graph)
    graph = full_graph.subgraph(component_nodes).copy()
    retained_collapsed_count = sum(
        int(attributes.get("source_parallel_edge_count", 1)) - 1
        for _, _, attributes in graph.edges(data=True)
    )
    source_timestamp = "unknown"
    osm3s = payload.get("osm3s")
    if isinstance(osm3s, Mapping):
        source_timestamp = str(osm3s.get("timestamp_osm_base", "unknown"))
    graph.graph.update(
        {
            "schema_version": NETWORK_SCHEMA_VERSION,
            "network_id": spec.network_id,
            "synthetic": False,
            "topology": spec.topology_class,
            "topology_family": spec.topology_class,
            "topology_source": "selection_motivation_from_config_not_algorithm_logic",
            "region_label": spec.region_label,
            "selection_rationale": spec.selection_rationale,
            "coordinate_system": "WGS84_longitude_latitude",
            "source": "OpenStreetMap via Overpass API",
            "source_timestamp_utc": source_timestamp,
            "attribution": OSM_ATTRIBUTION,
            "copyright_url": OSM_COPYRIGHT_URL,
            "bounding_box": spec.bounding_box.overpass_text(),
            "overpass_query": build_overpass_query(spec),
            "parallel_edge_policy": "minimum_base_travel_time",
            "parallel_edges_collapsed": retained_collapsed_count,
            "source_parallel_edges_collapsed_full_extract": collapsed_count,
            "parallel_edge_projection": retained_collapsed_count > 0,
            "parallel_edge_projection_caveat": (
                "minimum-base-time canonical projection; not a full multigraph "
                "distance, reliability, incident, or closure baseline"
            ),
            "skipped_segments": skipped_segments,
            "source_node_count": len(used_nodes),
            "largest_strong_component_node_count": len(component_nodes),
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
            "traffic_semantics": "controlled_simulated_traffic_not_live_traffic",
            "study_label": "real road topology with controlled simulated traffic",
        }
    )
    validate_network(graph)
    origin, destination, available = select_reproducible_od_pair(
        graph, candidate_k=spec.candidate_k
    )
    graph.graph["origin"] = origin
    graph.graph["destination"] = destination
    graph.graph["od_candidate_route_count"] = available
    return graph


def save_cached_graphml(graph: nx.DiGraph, path: str | Path) -> Path:
    """Persist one canonical real-topology graph for later offline loading."""

    validate_network(graph)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output)
    return output
