"""Canonical, location-independent road-network records and validation.

NetworkX node keys are the canonical ``node_id`` values. Coordinates are
optional local Cartesian values; routing and validation do not depend on them.
Unknown attributes are deliberately preserved so later traffic-state features
can be attached without changing this base schema.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
import math
from numbers import Real

import networkx as nx


NETWORK_SCHEMA_VERSION = "1.0"


class NetworkSchemaError(ValueError):
    """Raised when a graph does not satisfy the canonical road schema."""


def _validate_node_id(node_id: Hashable, field_name: str = "node_id") -> None:
    if node_id is None:
        raise NetworkSchemaError(f"{field_name} cannot be None")
    if not isinstance(node_id, Hashable):
        raise NetworkSchemaError(f"{field_name} must be hashable")


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise NetworkSchemaError(f"{field_name} must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise NetworkSchemaError(f"{field_name} must be finite")
    return numeric_value


def _positive_number(value: object, field_name: str) -> float:
    numeric_value = _finite_number(value, field_name)
    if numeric_value <= 0.0:
        raise NetworkSchemaError(f"{field_name} must be greater than zero")
    return numeric_value


@dataclass(frozen=True)
class NodeData:
    """Canonical node input.

    ``x`` and ``y`` are optional coordinates in an explicitly documented
    coordinate system. Both must be supplied together. The node identifier is
    stored as the NetworkX node key rather than duplicated as an attribute.
    """

    node_id: Hashable
    x: float | None = None
    y: float | None = None

    def __post_init__(self) -> None:
        _validate_node_id(self.node_id)
        if (self.x is None) != (self.y is None):
            raise NetworkSchemaError("x and y coordinates must be supplied together")
        if self.x is not None:
            _finite_number(self.x, "x")
            _finite_number(self.y, "y")

    def as_attributes(self) -> dict[str, float]:
        """Return NetworkX attributes, omitting absent coordinates."""

        if self.x is None:
            return {}
        return {"x": float(self.x), "y": float(self.y)}


@dataclass(frozen=True)
class EdgeData:
    """Canonical directed-road input with optional descriptive attributes."""

    source: Hashable
    target: Hashable
    distance_km: float
    base_travel_time_min: float
    speed_kph: float | None = None
    road_class: str | None = None
    reliability: float | None = None

    def __post_init__(self) -> None:
        _validate_node_id(self.source, "source")
        _validate_node_id(self.target, "target")
        _positive_number(self.distance_km, "distance_km")
        _positive_number(self.base_travel_time_min, "base_travel_time_min")
        if self.speed_kph is not None:
            _positive_number(self.speed_kph, "speed_kph")
        if self.road_class is not None:
            if not isinstance(self.road_class, str) or not self.road_class.strip():
                raise NetworkSchemaError("road_class must be a non-empty string")
        if self.reliability is not None:
            reliability = _finite_number(self.reliability, "reliability")
            if not 0.0 <= reliability <= 1.0:
                raise NetworkSchemaError("reliability must be between zero and one")

    def as_attributes(self) -> dict[str, float | str]:
        """Return canonical NetworkX edge attributes."""

        attributes: dict[str, float | str] = {
            "distance_km": float(self.distance_km),
            "base_travel_time_min": float(self.base_travel_time_min),
        }
        if self.speed_kph is not None:
            attributes["speed_kph"] = float(self.speed_kph)
        if self.road_class is not None:
            attributes["road_class"] = self.road_class
        if self.reliability is not None:
            attributes["reliability"] = float(self.reliability)
        return attributes


def _require_directed_simple_graph(graph: object) -> nx.DiGraph:
    if not isinstance(graph, nx.DiGraph) or not graph.is_directed():
        raise NetworkSchemaError("network must be a directed networkx.DiGraph")
    if graph.is_multigraph():
        raise NetworkSchemaError("parallel edges are not supported by the canonical schema")
    return graph


def add_node(graph: nx.DiGraph, node: NodeData) -> None:
    """Add one canonical node to a directed graph."""

    _require_directed_simple_graph(graph)
    graph.add_node(node.node_id, **node.as_attributes())


def add_edge(graph: nx.DiGraph, edge: EdgeData) -> None:
    """Add one canonical edge, requiring both endpoint nodes to exist."""

    _require_directed_simple_graph(graph)
    missing = [node_id for node_id in (edge.source, edge.target) if node_id not in graph]
    if missing:
        raise NetworkSchemaError(f"edge endpoints must already exist; missing {missing!r}")
    graph.add_edge(edge.source, edge.target, **edge.as_attributes())


def validate_network(graph: nx.DiGraph, *, require_nonempty: bool = True) -> nx.DiGraph:
    """Validate and return an arbitrary directed graph against the schema.

    Validation permits extra graph, node, and edge attributes. This keeps the
    canonical layer compatible with later dynamic traffic and context state.
    """

    _require_directed_simple_graph(graph)
    errors: list[str] = []

    if require_nonempty and graph.number_of_nodes() == 0:
        errors.append("network must contain at least one node")

    for node_id, attributes in graph.nodes(data=True):
        try:
            _validate_node_id(node_id)
        except NetworkSchemaError as error:
            errors.append(f"node {node_id!r}: {error}")

        if "node_id" in attributes and attributes["node_id"] != node_id:
            errors.append(f"node {node_id!r}: node_id attribute does not match its graph key")

        has_x = attributes.get("x") is not None
        has_y = attributes.get("y") is not None
        if has_x != has_y:
            errors.append(f"node {node_id!r}: x and y coordinates must be supplied together")
        elif has_x:
            for field_name in ("x", "y"):
                try:
                    _finite_number(attributes[field_name], field_name)
                except NetworkSchemaError as error:
                    errors.append(f"node {node_id!r}: {error}")

    for source, target, attributes in graph.edges(data=True):
        edge_label = f"edge {source!r}->{target!r}"
        for field_name in ("distance_km", "base_travel_time_min"):
            if field_name not in attributes:
                errors.append(f"{edge_label}: missing required {field_name}")
                continue
            try:
                _positive_number(attributes[field_name], field_name)
            except NetworkSchemaError as error:
                errors.append(f"{edge_label}: {error}")

        if attributes.get("speed_kph") is not None:
            try:
                _positive_number(attributes["speed_kph"], "speed_kph")
            except NetworkSchemaError as error:
                errors.append(f"{edge_label}: {error}")

        if attributes.get("road_class") is not None:
            road_class = attributes["road_class"]
            if not isinstance(road_class, str) or not road_class.strip():
                errors.append(f"{edge_label}: road_class must be a non-empty string")

        if attributes.get("reliability") is not None:
            try:
                reliability = _finite_number(attributes["reliability"], "reliability")
                if not 0.0 <= reliability <= 1.0:
                    raise NetworkSchemaError("reliability must be between zero and one")
            except NetworkSchemaError as error:
                errors.append(f"{edge_label}: {error}")

    if errors:
        formatted_errors = "\n".join(f"- {error}" for error in errors)
        raise NetworkSchemaError(f"invalid canonical road network:\n{formatted_errors}")

    return graph
