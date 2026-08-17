import networkx as nx
import pytest

from src.candidate_routes import generate_candidate_routes
from src.network_loader import load_network_graphml
from src.osm_adapter import (
    BoundingBox,
    OSMExtractionSpec,
    build_overpass_query,
    overpass_json_to_canonical_graph,
    save_cached_graphml,
)


def _grid_payload():
    elements = []
    node_id = 1
    ids = {}
    for row in range(3):
        for column in range(3):
            ids[row, column] = node_id
            elements.append(
                {
                    "type": "node",
                    "id": node_id,
                    "lat": 30.0 + row * 0.001,
                    "lon": 31.0 + column * 0.001,
                }
            )
            node_id += 1
    way_id = 100
    for row in range(3):
        elements.append(
            {
                "type": "way",
                "id": way_id,
                "nodes": [ids[row, column] for column in range(3)],
                "tags": {"highway": "residential", "maxspeed": "30"},
            }
        )
        way_id += 1
    for column in range(3):
        elements.append(
            {
                "type": "way",
                "id": way_id,
                "nodes": [ids[row, column] for row in range(3)],
                "tags": {"highway": "tertiary", "maxspeed": "20 mph"},
            }
        )
        way_id += 1
    # A slower parallel way proves that collapse is explicit and traceable.
    elements.append(
        {
            "type": "way",
            "id": 999,
            "nodes": [ids[0, 0], ids[0, 1]],
            "tags": {"highway": "residential", "maxspeed": "10"},
        }
    )
    return {
        "osm3s": {"timestamp_osm_base": "2026-08-01T00:00:00Z"},
        "elements": elements,
    }


def _spec():
    return OSMExtractionSpec(
        network_id="fixture-grid",
        region_label="unit-test region",
        selection_rationale="small orthogonal topology for conversion testing",
        bounding_box=BoundingBox(29.9, 30.9, 30.1, 31.1),
        topology_class="orthogonal_grid",
        highway_classes=("tertiary", "residential"),
        candidate_k=4,
    )


def test_query_is_location_data_not_algorithm_branch():
    query = build_overpass_query(_spec())
    assert "29.9000000,30.9000000,30.1000000,31.1000000" in query
    assert "tertiary|residential" in query
    assert "unit-test region" not in query


@pytest.mark.parametrize(
    "values",
    [(-91, 0, 1, 2), (2, 0, 1, 2), (0, 5, 1, 4)],
)
def test_bounding_box_validation(values):
    with pytest.raises(ValueError):
        BoundingBox(*values)


def test_overpass_conversion_is_canonical_and_traceable():
    graph = overpass_json_to_canonical_graph(_grid_payload(), _spec())
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 9
    assert graph.graph["source_timestamp_utc"] == "2026-08-01T00:00:00Z"
    assert graph.graph["traffic_semantics"] == "controlled_simulated_traffic_not_live_traffic"
    assert graph.graph["study_label"] == "real road topology with controlled simulated traffic"
    assert graph.graph["topology"] == "orthogonal_grid"
    # The bidirectional duplicate contributes one collapsed alternative in
    # each direction.
    assert graph.graph["parallel_edges_collapsed"] == 2
    assert graph.graph["od_candidate_route_count"] == 4
    assert all(data["distance_km"] > 0 for _, _, data in graph.edges(data=True))
    assert all(data["base_travel_time_min"] > 0 for _, _, data in graph.edges(data=True))
    assert graph.edges["1", "2"]["source_parallel_edge_count"] == 2
    assert len(
        generate_candidate_routes(
            graph,
            graph.graph["origin"],
            graph.graph["destination"],
            4,
            require_k=True,
        )
    ) == 4


def test_cached_graphml_replays_offline(tmp_path):
    graph = overpass_json_to_canonical_graph(_grid_payload(), _spec())
    path = save_cached_graphml(graph, tmp_path / "fixture.graphml")
    loaded = load_network_graphml(path)
    assert loaded.graph["network_id"] == "fixture-grid"
    assert loaded.graph["origin"] == graph.graph["origin"]
    assert loaded.graph["destination"] == graph.graph["destination"]
    assert loaded.graph["source_timestamp_utc"] == "2026-08-01T00:00:00Z"
    assert loaded.graph["attribution"] == graph.graph["attribution"]
    assert loaded.graph["overpass_query"] == graph.graph["overpass_query"]
    assert loaded.graph["parallel_edges_collapsed"] == 2
    assert loaded.graph["parallel_edge_projection"] is True
    assert loaded.graph["study_label"] == "real road topology with controlled simulated traffic"
    assert loaded.number_of_nodes() == graph.number_of_nodes()
    assert loaded.number_of_edges() == graph.number_of_edges()
