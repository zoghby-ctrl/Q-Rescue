"""Offline, location-neutral JSON and GraphML road-network adapters."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

import networkx as nx

from src.network_schema import (
    EdgeData,
    NETWORK_SCHEMA_VERSION,
    NetworkSchemaError,
    NodeData,
    add_edge,
    add_node,
    validate_network,
)


class NetworkLoadError(ValueError):
    """Raised when a local network file cannot be normalized safely."""


CANONICAL_NETWORK_FORMAT = "qrescue-canonical-network-v1"
GRAPHML_PARALLEL_EDGE_POLICY = (
    "fastest_base_travel_time_then_distance_then_source_edge_key"
)
JSON_PARALLEL_EDGE_POLICY = "reject_duplicate_directed_edges"


_PRESERVED_GRAPH_METADATA = {
    "network_id",
    "topology",
    "topology_family",
    "topology_description",
    "seed",
    "synthetic",
    "coordinate_system",
    "generator_parameters_json",
    "origin",
    "destination",
    "od_candidate_route_count",
    "crs",
    "region_label",
    "selection_rationale",
    "topology_source",
    "source",
    "source_timestamp_utc",
    "attribution",
    "copyright_url",
    "bounding_box",
    "overpass_query",
    "overpass_endpoint",
    "raw_overpass_sha256",
    "parallel_edge_policy",
    "parallel_edges_collapsed",
    "source_parallel_edges_collapsed_full_extract",
    "parallel_edge_projection",
    "parallel_edge_projection_caveat",
    "skipped_segments",
    "source_node_count",
    "largest_strong_component_node_count",
    "cached_at_utc",
    "traffic_semantics",
    "study_label",
}


def _node_sort_key(node_id: Hashable) -> tuple[str, str, str]:
    node_type = type(node_id)
    return node_type.__module__, node_type.__qualname__, repr(node_id)


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, str)):
        raise NetworkLoadError(f"{field_name} must be a simple numeric value")
    try:
        numeric_value = float(value.strip() if isinstance(value, str) else value)
    except ValueError as error:
        raise NetworkLoadError(f"{field_name} must be a simple numeric value") from error
    if not math.isfinite(numeric_value):
        raise NetworkLoadError(f"{field_name} must be finite")
    return numeric_value


def _positive_number(value: object, field_name: str) -> float:
    numeric_value = _finite_number(value, field_name)
    if numeric_value <= 0.0:
        raise NetworkLoadError(f"{field_name} must be greater than zero")
    return numeric_value


_SPEED_PATTERN = re.compile(
    r"^\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>km/h|kph|kmh|mph)?\s*$",
    re.IGNORECASE,
)


def _parse_speed_kph(value: object) -> float | None:
    """Parse a scalar speed, an optional km/h suffix, or a simple mph suffix."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Real):
        speed = float(value)
        return speed if math.isfinite(speed) and speed > 0.0 else None
    if not isinstance(value, str):
        return None
    match = _SPEED_PATTERN.fullmatch(value)
    if match is None:
        return None
    speed = float(match.group("value"))
    if match.group("unit") and match.group("unit").lower() == "mph":
        speed *= 1.609344
    return speed if speed > 0.0 else None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
    else:
        return None
    return text or None


def _normalize_node_attributes(
    node_id: Hashable,
    attributes: Mapping[str, object],
    conversions: Counter[str],
) -> dict[str, object]:
    normalized = {key: value for key, value in attributes.items() if key != "node_id"}
    has_x = attributes.get("x") is not None
    has_y = attributes.get("y") is not None
    if has_x != has_y:
        raise NetworkLoadError(f"node {node_id!r} must provide both x and y coordinates")
    if has_x:
        normalized["x"] = _finite_number(attributes["x"], "x")
        normalized["y"] = _finite_number(attributes["y"], "y")
        conversions["coordinate_values_parsed"] += 1
    else:
        normalized.pop("x", None)
        normalized.pop("y", None)
    return normalized


def _normalize_edge_attributes(
    source: Hashable,
    target: Hashable,
    attributes: Mapping[str, object],
    conversions: Counter[str],
    *,
    allow_graphml_aliases: bool,
) -> dict[str, object]:
    edge_label = f"edge {source!r}->{target!r}"
    normalized = dict(attributes)

    if attributes.get("distance_km") is not None:
        distance_km = _positive_number(attributes["distance_km"], "distance_km")
    elif allow_graphml_aliases and attributes.get("length") is not None:
        distance_km = _positive_number(attributes["length"], "length") / 1000.0
        conversions["length_m_to_distance_km"] += 1
    else:
        raise NetworkLoadError(
            f"{edge_label} needs distance_km or GraphML length in meters"
        )

    speed_kph: float | None = None
    speed_source: str | None = None
    for candidate_name in ("speed_kph", "maxspeed"):
        if attributes.get(candidate_name) is not None:
            parsed_speed = _parse_speed_kph(attributes[candidate_name])
            if parsed_speed is not None:
                speed_kph = parsed_speed
                speed_source = candidate_name
                break
    if speed_kph is not None:
        conversions[f"{speed_source}_parsed_as_speed_kph"] += 1
    elif attributes.get("speed_kph") is not None or attributes.get("maxspeed") is not None:
        conversions["unparsed_speed_values_omitted"] += 1

    if attributes.get("base_travel_time_min") is not None:
        base_time_min = _positive_number(
            attributes["base_travel_time_min"], "base_travel_time_min"
        )
    elif allow_graphml_aliases and attributes.get("travel_time") is not None:
        base_time_min = _positive_number(attributes["travel_time"], "travel_time") / 60.0
        conversions["travel_time_s_to_base_travel_time_min"] += 1
    elif allow_graphml_aliases and speed_kph is not None:
        base_time_min = 60.0 * distance_km / speed_kph
        conversions["base_travel_time_derived_from_distance_and_speed"] += 1
    else:
        raise NetworkLoadError(
            f"{edge_label} needs base_travel_time_min, GraphML travel_time in seconds, "
            "or a parseable speed"
        )

    road_class = _optional_text(attributes.get("road_class"))
    if road_class is None and allow_graphml_aliases:
        road_class = _optional_text(attributes.get("highway"))
        if road_class is not None:
            conversions["highway_preserved_as_road_class"] += 1

    reliability: float | None = None
    if attributes.get("reliability") is not None:
        reliability = _finite_number(attributes["reliability"], "reliability")
        if not 0.0 <= reliability <= 1.0:
            raise NetworkLoadError(f"{edge_label}: reliability must be between zero and one")

    try:
        canonical = EdgeData(
            source=source,
            target=target,
            distance_km=distance_km,
            base_travel_time_min=base_time_min,
            speed_kph=speed_kph,
            road_class=road_class,
            reliability=reliability,
        ).as_attributes()
    except NetworkSchemaError as error:
        raise NetworkLoadError(f"{edge_label}: {error}") from error

    for field_name in (
        "distance_km",
        "base_travel_time_min",
        "speed_kph",
        "road_class",
        "reliability",
    ):
        normalized.pop(field_name, None)
    normalized.update(canonical)
    return normalized


def _copy_source_metadata(source_graph: nx.Graph, target_graph: nx.DiGraph) -> None:
    for key in _PRESERVED_GRAPH_METADATA:
        if key in source_graph.graph:
            target_graph.graph[key] = source_graph.graph[key]


def _record_loader_metadata(
    graph: nx.DiGraph,
    *,
    source_format: str,
    source_path: Path,
    original_graph_type: str,
    input_edge_count: int,
    conversions: Counter[str],
    parallel_edge_policy: str,
    parallel_endpoint_count: int = 0,
) -> None:
    collapsed_edge_count = input_edge_count - graph.number_of_edges()
    graph.graph.update(
        {
            "schema_version": NETWORK_SCHEMA_VERSION,
            "loader_source_format": source_format,
            "loader_source_file": source_path.name,
            "loader_source_path": str(source_path.resolve()),
            "loader_original_graph_type": original_graph_type,
            "loader_input_edge_count": input_edge_count,
            "loader_output_edge_count": graph.number_of_edges(),
            "loader_parallel_edges_collapsed": collapsed_edge_count,
            "loader_parallel_endpoint_count": parallel_endpoint_count,
            "loader_parallel_edge_policy": parallel_edge_policy,
            "loader_parallel_edge_projection": collapsed_edge_count > 0,
            "loader_parallel_edge_warning": (
                "Canonical DiGraph retains only the fastest-base-time edge for each "
                "parallel endpoint pair; it is not a full multigraph baseline for "
                "distance, reliability, incidents, or closures. Inspect each selected "
                "edge's parallel_edge_alternatives_json before those analyses."
                if collapsed_edge_count > 0
                else "none"
            ),
            "loader_conversions_json": json.dumps(
                dict(sorted(conversions.items())), sort_keys=True, separators=(",", ":")
            ),
        }
    )


def load_network_graphml(path: str | Path) -> nx.DiGraph:
    """Load and normalize one cached local GraphML road network.

    Ordinary ``DiGraph`` and OSM-style ``MultiDiGraph`` inputs are supported.
    Parallel directed edges are collapsed deterministically by lowest base
    travel time, then distance, then source edge-key representation.

    This is explicitly a fastest-time projection into the canonical simple
    ``DiGraph``.  When alternatives exist, their canonical summaries are saved
    on the retained edge as ``parallel_edge_alternatives_json`` and projection
    metadata is saved on the graph.  The result must not be presented as the
    full multigraph shortest-distance/reliability/closure baseline.
    """

    source_path = Path(path)
    try:
        source_graph = nx.read_graphml(source_path)
    except (OSError, nx.NetworkXError) as error:
        raise NetworkLoadError(f"could not read GraphML file {source_path}") from error
    if not source_graph.is_directed():
        raise NetworkLoadError("GraphML road network must be directed")

    output = nx.DiGraph()
    _copy_source_metadata(source_graph, output)
    conversions: Counter[str] = Counter()

    for node_id, attributes in sorted(
        source_graph.nodes(data=True), key=lambda item: _node_sort_key(item[0])
    ):
        normalized = _normalize_node_attributes(node_id, attributes, conversions)
        x = normalized.get("x")
        y = normalized.get("y")
        add_node(output, NodeData(node_id=node_id, x=x, y=y))
        output.nodes[node_id].update(normalized)

    if source_graph.is_multigraph():
        raw_edges = list(source_graph.edges(keys=True, data=True))
    else:
        raw_edges = [
            (source, target, None, attributes)
            for source, target, attributes in source_graph.edges(data=True)
        ]
    raw_edges.sort(
        key=lambda item: (
            _node_sort_key(item[0]),
            _node_sort_key(item[1]),
            _node_sort_key(item[2]),
        )
    )

    selected_edges: dict[
        tuple[Hashable, Hashable], tuple[tuple[float, float, tuple[str, str, str]], dict[str, object]]
    ] = {}
    edge_candidates: dict[
        tuple[Hashable, Hashable],
        list[tuple[tuple[float, float, tuple[str, str, str]], dict[str, object]]],
    ] = {}
    for source, target, source_key, attributes in raw_edges:
        normalized = _normalize_edge_attributes(
            source,
            target,
            attributes,
            conversions,
            allow_graphml_aliases=True,
        )
        if source_key is not None:
            normalized["source_edge_key"] = str(source_key)
        ranking = (
            float(normalized["base_travel_time_min"]),
            float(normalized["distance_km"]),
            _node_sort_key(source_key),
        )
        endpoint_pair = (source, target)
        edge_candidates.setdefault(endpoint_pair, []).append((ranking, normalized))
        if endpoint_pair not in selected_edges or ranking < selected_edges[endpoint_pair][0]:
            selected_edges[endpoint_pair] = ranking, normalized

    parallel_endpoint_count = 0
    for endpoint_pair, candidates in edge_candidates.items():
        if len(candidates) <= 1:
            continue
        parallel_endpoint_count += 1
        selected_attributes = selected_edges[endpoint_pair][1]
        alternatives: list[dict[str, object]] = []
        for _, candidate in sorted(candidates, key=lambda item: item[0]):
            alternatives.append(
                {
                    "source_edge_key": candidate.get("source_edge_key"),
                    "distance_km": float(candidate["distance_km"]),
                    "base_travel_time_min": float(candidate["base_travel_time_min"]),
                    "speed_kph": candidate.get("speed_kph"),
                    "road_class": candidate.get("road_class"),
                    "reliability": candidate.get("reliability"),
                    "selected": candidate is selected_attributes,
                }
            )
        selected_attributes.update(
            {
                "parallel_edge_candidate_count": len(candidates),
                "parallel_edge_projection_policy": GRAPHML_PARALLEL_EDGE_POLICY,
                "parallel_edge_alternatives_json": json.dumps(
                    alternatives, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    for (source, target), (_, attributes) in sorted(
        selected_edges.items(),
        key=lambda item: (_node_sort_key(item[0][0]), _node_sort_key(item[0][1])),
    ):
        canonical = EdgeData(
            source=source,
            target=target,
            distance_km=float(attributes["distance_km"]),
            base_travel_time_min=float(attributes["base_travel_time_min"]),
            speed_kph=(
                float(attributes["speed_kph"]) if attributes.get("speed_kph") is not None else None
            ),
            road_class=(
                str(attributes["road_class"]) if attributes.get("road_class") is not None else None
            ),
            reliability=(
                float(attributes["reliability"])
                if attributes.get("reliability") is not None
                else None
            ),
        )
        add_edge(output, canonical)
        output.edges[source, target].update(attributes)

    _record_loader_metadata(
        output,
        source_format="graphml",
        source_path=source_path,
        original_graph_type=type(source_graph).__name__,
        input_edge_count=len(raw_edges),
        conversions=conversions,
        parallel_edge_policy=GRAPHML_PARALLEL_EDGE_POLICY,
        parallel_endpoint_count=parallel_endpoint_count,
    )
    try:
        return validate_network(output)
    except NetworkSchemaError as error:
        raise NetworkLoadError(f"normalized GraphML network is invalid: {error}") from error


def _require_json_node_id(value: object, field_name: str) -> Hashable:
    if value is None or isinstance(value, (list, dict)):
        raise NetworkLoadError(f"{field_name} must be a non-null JSON scalar")
    try:
        hash(value)
    except TypeError as error:
        raise NetworkLoadError(f"{field_name} must be hashable") from error
    return value  # type: ignore[return-value]


def load_network_json(path: str | Path) -> nx.DiGraph:
    """Load a local canonical Q-Rescue JSON network file."""

    source_path = Path(path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NetworkLoadError(f"could not read JSON network file {source_path}") from error
    if not isinstance(payload, dict):
        raise NetworkLoadError("JSON network root must be an object")
    if payload.get("format") != CANONICAL_NETWORK_FORMAT:
        raise NetworkLoadError(
            f"JSON network format must be {CANONICAL_NETWORK_FORMAT!r}"
        )
    if payload.get("schema_version") != NETWORK_SCHEMA_VERSION:
        raise NetworkLoadError(
            f"JSON network schema_version must be {NETWORK_SCHEMA_VERSION!r}"
        )
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    metadata = payload.get("metadata", {})
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise NetworkLoadError("JSON network must contain nodes and edges arrays")
    if not isinstance(metadata, dict):
        raise NetworkLoadError("JSON metadata must be an object")

    output = nx.DiGraph()
    output.graph.update(metadata)
    output.graph["loader_json_format"] = CANONICAL_NETWORK_FORMAT
    output.graph["loader_json_schema_version"] = NETWORK_SCHEMA_VERSION
    conversions: Counter[str] = Counter()
    for index, record in enumerate(nodes):
        if not isinstance(record, dict) or "node_id" not in record:
            raise NetworkLoadError(f"nodes[{index}] must be an object containing node_id")
        node_id = _require_json_node_id(record["node_id"], f"nodes[{index}].node_id")
        if node_id in output:
            raise NetworkLoadError(f"duplicate JSON node_id {node_id!r}")
        normalized = _normalize_node_attributes(node_id, record, conversions)
        x = normalized.get("x")
        y = normalized.get("y")
        add_node(output, NodeData(node_id=node_id, x=x, y=y))
        output.nodes[node_id].update(normalized)

    for index, record in enumerate(edges):
        if not isinstance(record, dict) or "source" not in record or "target" not in record:
            raise NetworkLoadError(f"edges[{index}] must contain source and target")
        source = _require_json_node_id(record["source"], f"edges[{index}].source")
        target = _require_json_node_id(record["target"], f"edges[{index}].target")
        if source not in output or target not in output:
            raise NetworkLoadError(f"edges[{index}] references an unknown endpoint")
        if output.has_edge(source, target):
            raise NetworkLoadError(f"duplicate JSON directed edge {source!r}->{target!r}")
        normalized = _normalize_edge_attributes(
            source,
            target,
            record,
            conversions,
            allow_graphml_aliases=False,
        )
        canonical = EdgeData(
            source=source,
            target=target,
            distance_km=float(normalized["distance_km"]),
            base_travel_time_min=float(normalized["base_travel_time_min"]),
            speed_kph=(
                float(normalized["speed_kph"]) if normalized.get("speed_kph") is not None else None
            ),
            road_class=(
                str(normalized["road_class"]) if normalized.get("road_class") is not None else None
            ),
            reliability=(
                float(normalized["reliability"])
                if normalized.get("reliability") is not None
                else None
            ),
        )
        add_edge(output, canonical)
        output.edges[source, target].update(
            {key: value for key, value in normalized.items() if key not in {"source", "target"}}
        )

    _record_loader_metadata(
        output,
        source_format="json",
        source_path=source_path,
        original_graph_type="canonical_json",
        input_edge_count=len(edges),
        conversions=conversions,
        parallel_edge_policy=JSON_PARALLEL_EDGE_POLICY,
    )
    try:
        return validate_network(output)
    except NetworkSchemaError as error:
        raise NetworkLoadError(f"JSON network is invalid: {error}") from error


def _assert_json_serializable(value: object, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise NetworkLoadError(f"{label} is not JSON serializable") from error


def save_network_json(graph: nx.DiGraph, path: str | Path) -> Path:
    """Validate and save a canonical network in the documented JSON shape."""

    validate_network(graph)
    output_path = Path(path)
    node_records: list[dict[str, Any]] = []
    for node_id, attributes in sorted(graph.nodes(data=True), key=lambda item: _node_sort_key(item[0])):
        _require_json_node_id(node_id, "node_id")
        record = {"node_id": node_id, **dict(attributes)}
        _assert_json_serializable(record, f"node {node_id!r}")
        node_records.append(record)

    edge_records: list[dict[str, Any]] = []
    for source, target, attributes in sorted(
        graph.edges(data=True),
        key=lambda item: (_node_sort_key(item[0]), _node_sort_key(item[1])),
    ):
        _require_json_node_id(source, "source")
        _require_json_node_id(target, "target")
        record = {"source": source, "target": target, **dict(attributes)}
        _assert_json_serializable(record, f"edge {source!r}->{target!r}")
        edge_records.append(record)

    metadata = dict(graph.graph)
    _assert_json_serializable(metadata, "graph metadata")
    payload = {
        "format": CANONICAL_NETWORK_FORMAT,
        "schema_version": NETWORK_SCHEMA_VERSION,
        "metadata": metadata,
        "nodes": node_records,
        "edges": edge_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_network(path: str | Path) -> nx.DiGraph:
    """Dispatch a local path to the JSON or GraphML adapter by suffix."""

    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".json":
        return load_network_json(source_path)
    if suffix in {".graphml", ".xml"}:
        return load_network_graphml(source_path)
    raise NetworkLoadError(f"unsupported network file suffix {source_path.suffix!r}")
