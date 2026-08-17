"""Tests for offline JSON and GraphML network adapters."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.network_loader import (
    CANONICAL_NETWORK_FORMAT,
    GRAPHML_PARALLEL_EDGE_POLICY,
    NetworkLoadError,
    load_network,
    load_network_graphml,
    load_network_json,
    save_network_json,
)
from src.network_schema import NETWORK_SCHEMA_VERSION, validate_network
from src.synthetic_networks import generate_grid_network


def test_canonical_json_round_trip_preserves_network_and_metadata(tmp_path: Path) -> None:
    original = generate_grid_network(rows=3, columns=3, seed=14)
    json_path = save_network_json(original, tmp_path / "network.json")
    loaded = load_network_json(json_path)

    assert validate_network(loaded) is loaded
    assert set(loaded.nodes) == set(original.nodes)
    assert set(loaded.edges) == set(original.edges)
    assert loaded.graph["origin"] == original.graph["origin"]
    assert loaded.graph["destination"] == original.graph["destination"]
    assert loaded.graph["loader_source_format"] == "json"
    assert loaded.graph["loader_source_file"] == "network.json"
    assert loaded.graph["loader_parallel_edges_collapsed"] == 0
    assert loaded.graph["loader_parallel_edge_policy"] == "reject_duplicate_directed_edges"
    assert loaded.graph["loader_json_format"] == CANONICAL_NETWORK_FORMAT
    assert loaded.graph["loader_json_schema_version"] == NETWORK_SCHEMA_VERSION
    assert load_network(json_path).number_of_edges() == original.number_of_edges()


def test_graphml_normalizes_osm_style_units_and_preserves_optional_fields(
    tmp_path: Path,
) -> None:
    source = nx.DiGraph()
    source.add_node("a", x="1.25", y="2.5")
    source.add_node("b", x="2.25", y="3.5")
    source.add_edge(
        "a",
        "b",
        length="1500",
        travel_time="120",
        maxspeed="45 km/h",
        highway="primary",
    )
    graphml_path = tmp_path / "ordinary.graphml"
    nx.write_graphml(source, graphml_path)

    loaded = load_network_graphml(graphml_path)
    edge = loaded.edges["a", "b"]
    conversions = json.loads(loaded.graph["loader_conversions_json"])

    assert isinstance(loaded, nx.DiGraph)
    assert not loaded.is_multigraph()
    assert np.isclose(loaded.nodes["a"]["x"], 1.25)
    assert np.isclose(edge["distance_km"], 1.5)
    assert np.isclose(edge["base_travel_time_min"], 2.0)
    assert np.isclose(edge["speed_kph"], 45.0)
    assert edge["road_class"] == "primary"
    assert conversions["length_m_to_distance_km"] == 1
    assert conversions["travel_time_s_to_base_travel_time_min"] == 1
    assert conversions["highway_preserved_as_road_class"] == 1


def test_graphml_prefers_existing_canonical_attributes(tmp_path: Path) -> None:
    source = nx.DiGraph()
    source.add_nodes_from(["a", "b"])
    source.add_edge(
        "a",
        "b",
        distance_km=2.0,
        base_travel_time_min=3.0,
        length=9000.0,
        travel_time=9000.0,
    )
    path = tmp_path / "canonical.graphml"
    nx.write_graphml(source, path)

    loaded = load_network_graphml(path)
    assert loaded.edges["a", "b"]["distance_km"] == 2.0
    assert loaded.edges["a", "b"]["base_travel_time_min"] == 3.0
    conversions = json.loads(loaded.graph["loader_conversions_json"])
    assert "length_m_to_distance_km" not in conversions
    assert "travel_time_s_to_base_travel_time_min" not in conversions


def test_multidigraph_parallel_edges_collapse_to_lowest_base_time(tmp_path: Path) -> None:
    source = nx.MultiDiGraph()
    source.add_nodes_from(["a", "b"])
    source.add_edge("a", "b", key="slow", length=1000.0, travel_time=180.0)
    source.add_edge("a", "b", key="fast", length=1200.0, travel_time=60.0)
    source.add_edge("b", "a", key="return", length=1000.0, travel_time=90.0)
    path = tmp_path / "parallel.graphml"
    nx.write_graphml(source, path)

    loaded = load_network_graphml(path)

    assert not loaded.is_multigraph()
    assert loaded.number_of_edges() == 2
    assert np.isclose(loaded.edges["a", "b"]["base_travel_time_min"], 1.0)
    assert np.isclose(loaded.edges["a", "b"]["distance_km"], 1.2)
    assert loaded.graph["loader_input_edge_count"] == 3
    assert loaded.graph["loader_parallel_edges_collapsed"] == 1
    assert loaded.graph["loader_parallel_endpoint_count"] == 1
    assert loaded.graph["loader_parallel_edge_projection"] is True
    assert loaded.graph["loader_parallel_edge_policy"] == GRAPHML_PARALLEL_EDGE_POLICY
    assert "not a full multigraph baseline" in loaded.graph["loader_parallel_edge_warning"]
    assert loaded.edges["a", "b"]["parallel_edge_candidate_count"] == 2
    alternatives = json.loads(
        loaded.edges["a", "b"]["parallel_edge_alternatives_json"]
    )
    assert [alternative["source_edge_key"] for alternative in alternatives] == [
        "fast",
        "slow",
    ]
    assert [alternative["distance_km"] for alternative in alternatives] == [1.2, 1.0]
    assert [alternative["selected"] for alternative in alternatives] == [True, False]


def test_graphml_can_derive_base_time_from_simple_mph_speed(tmp_path: Path) -> None:
    source = nx.DiGraph()
    source.add_nodes_from(["a", "b"])
    source.add_edge("a", "b", length=1000.0, maxspeed="30 mph")
    path = tmp_path / "speed.graphml"
    nx.write_graphml(source, path)

    loaded = load_network_graphml(path)
    expected_speed_kph = 30.0 * 1.609344
    assert np.isclose(loaded.edges["a", "b"]["speed_kph"], expected_speed_kph)
    assert np.isclose(
        loaded.edges["a", "b"]["base_travel_time_min"],
        60.0 / expected_speed_kph,
    )


def test_json_adapter_rejects_duplicate_edges(tmp_path: Path) -> None:
    payload = {
        "format": CANONICAL_NETWORK_FORMAT,
        "schema_version": NETWORK_SCHEMA_VERSION,
        "nodes": [{"node_id": "a"}, {"node_id": "b"}],
        "edges": [
            {
                "source": "a",
                "target": "b",
                "distance_km": 1.0,
                "base_travel_time_min": 2.0,
            },
            {
                "source": "a",
                "target": "b",
                "distance_km": 1.1,
                "base_travel_time_min": 2.2,
            },
        ],
    }
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NetworkLoadError, match="duplicate JSON directed edge"):
        load_network_json(path)


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("format", "some-other-format", "JSON network format"),
        ("schema_version", "999", "schema_version"),
        ("format", None, "JSON network format"),
        ("schema_version", None, "schema_version"),
    ],
)
def test_json_adapter_validates_format_and_schema_version(
    tmp_path: Path,
    field_name: str,
    field_value: str | None,
    message: str,
) -> None:
    payload = {
        "format": CANONICAL_NETWORK_FORMAT,
        "schema_version": NETWORK_SCHEMA_VERSION,
        "nodes": [{"node_id": "a"}, {"node_id": "b"}],
        "edges": [
            {
                "source": "a",
                "target": "b",
                "distance_km": 1.0,
                "base_travel_time_min": 2.0,
            }
        ],
    }
    if field_value is None:
        payload.pop(field_name)
    else:
        payload[field_name] = field_value
    path = tmp_path / f"invalid-{field_name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NetworkLoadError, match=message):
        load_network_json(path)


def test_graphml_rejects_missing_units_and_undirected_networks(tmp_path: Path) -> None:
    missing_time = nx.DiGraph()
    missing_time.add_edge("a", "b", length=1000.0)
    missing_path = tmp_path / "missing.graphml"
    nx.write_graphml(missing_time, missing_path)
    with pytest.raises(NetworkLoadError, match="needs base_travel_time_min"):
        load_network_graphml(missing_path)

    undirected = nx.Graph()
    undirected.add_edge("a", "b", length=1000.0, travel_time=60.0)
    undirected_path = tmp_path / "undirected.graphml"
    nx.write_graphml(undirected, undirected_path)
    with pytest.raises(NetworkLoadError, match="must be directed"):
        load_network_graphml(undirected_path)


def test_loader_dispatch_rejects_unknown_file_types(tmp_path: Path) -> None:
    with pytest.raises(NetworkLoadError, match="unsupported network file suffix"):
        load_network(tmp_path / "network.csv")
