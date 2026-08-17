"""Tests for the generalized canonical network layer and synthetic families."""

from collections.abc import Callable

import networkx as nx
import pytest

from src.network_schema import (
    EdgeData,
    NetworkSchemaError,
    NodeData,
    add_edge,
    add_node,
    validate_network,
)
from src.synthetic_networks import (
    SyntheticNetworkSpec,
    generate_dense_mixed_network,
    generate_grid_network,
    generate_irregular_geometric_network,
    generate_radial_network,
    generate_sparse_mixed_network,
    generate_synthetic_network,
)


def _graph_signature(graph: nx.DiGraph) -> tuple[object, ...]:
    nodes = tuple(
        sorted((str(node_id), tuple(sorted(attributes.items()))) for node_id, attributes in graph.nodes(data=True))
    )
    edges = tuple(
        sorted(
            (str(source), str(target), tuple(sorted(attributes.items())))
            for source, target, attributes in graph.edges(data=True)
        )
    )
    return tuple(sorted(graph.graph.items())), nodes, edges


def test_canonical_records_validate_an_arbitrary_directed_graph() -> None:
    graph = nx.DiGraph()
    add_node(graph, NodeData("origin", x=0.0, y=0.0))
    add_node(graph, NodeData("destination"))
    add_edge(
        graph,
        EdgeData(
            source="origin",
            target="destination",
            distance_km=1.25,
            base_travel_time_min=2.5,
            speed_kph=35.0,
            road_class="collector",
            reliability=0.97,
        ),
    )
    graph.edges["origin", "destination"]["incident"] = False

    assert validate_network(graph) is graph
    assert graph.nodes["destination"] == {}
    assert graph.edges["origin", "destination"]["distance_km"] == 1.25


def test_node_coordinates_must_be_complete() -> None:
    with pytest.raises(NetworkSchemaError, match="supplied together"):
        NodeData("incomplete", x=1.0)

    graph = nx.DiGraph()
    graph.add_node("incomplete", x=1.0)
    with pytest.raises(NetworkSchemaError, match="supplied together"):
        validate_network(graph)


def test_schema_rejects_undirected_and_multigraph_inputs() -> None:
    with pytest.raises(NetworkSchemaError, match="directed"):
        validate_network(nx.Graph())  # type: ignore[arg-type]
    with pytest.raises(NetworkSchemaError, match="parallel edges"):
        validate_network(nx.MultiDiGraph())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "attributes, expected_message",
    [
        ({"base_travel_time_min": 2.0}, "missing required distance_km"),
        ({"distance_km": 0.0, "base_travel_time_min": 2.0}, "greater than zero"),
        ({"distance_km": 1.0, "base_travel_time_min": float("nan")}, "must be finite"),
        (
            {"distance_km": 1.0, "base_travel_time_min": 2.0, "speed_kph": 0.0},
            "speed_kph must be greater than zero",
        ),
        (
            {"distance_km": 1.0, "base_travel_time_min": 2.0, "reliability": 1.1},
            "reliability must be between zero and one",
        ),
    ],
)
def test_schema_rejects_invalid_edge_attributes(
    attributes: dict[str, float],
    expected_message: str,
) -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b"])
    graph.add_edge("a", "b", **attributes)
    with pytest.raises(NetworkSchemaError, match=expected_message):
        validate_network(graph)


GENERATOR_CASES: tuple[tuple[str, Callable[[], nx.DiGraph]], ...] = (
    ("grid", lambda: generate_grid_network(rows=3, columns=4, seed=17)),
    ("radial", lambda: generate_radial_network(spokes=5, rings=2, seed=17)),
    (
        "irregular_geometric",
        lambda: generate_irregular_geometric_network(node_count=12, seed=17),
    ),
    ("sparse_mixed", lambda: generate_sparse_mixed_network(node_count=16, seed=17)),
    ("dense_mixed", lambda: generate_dense_mixed_network(node_count=16, seed=17)),
)


@pytest.mark.parametrize("topology,factory", GENERATOR_CASES, ids=[case[0] for case in GENERATOR_CASES])
def test_every_generator_is_deterministic_valid_and_routable(
    topology: str,
    factory: Callable[[], nx.DiGraph],
) -> None:
    first = factory()
    second = factory()

    assert _graph_signature(first) == _graph_signature(second)
    assert validate_network(first) is first
    assert first.graph["topology_family"] == topology
    assert first.graph["seed"] == 17
    assert first.graph["coordinate_system"] == "local_cartesian_km"
    assert first.graph["origin"] != first.graph["destination"]
    assert nx.has_path(first, first.graph["origin"], first.graph["destination"])
    assert nx.is_strongly_connected(first)
    assert all("x" in attributes and "y" in attributes for _, attributes in first.nodes(data=True))
    assert all(
        attributes["distance_km"] > 0.0 and attributes["base_travel_time_min"] > 0.0
        for _, _, attributes in first.edges(data=True)
    )


def test_grid_and_radial_have_interpretable_structures() -> None:
    grid = generate_grid_network(rows=3, columns=4, seed=2)
    radial = generate_radial_network(spokes=5, rings=2, seed=2)

    assert grid.number_of_nodes() == 12
    assert grid.number_of_edges() == 34
    assert radial.number_of_nodes() == 11
    assert "center" in radial
    assert {data["road_class"] for _, _, data in radial.edges(data=True)} == {"radial", "ring"}


def test_dense_mixed_has_more_road_redundancy_than_sparse_mixed() -> None:
    sparse = generate_sparse_mixed_network(node_count=25, seed=91)
    dense = generate_dense_mixed_network(node_count=25, seed=91)

    assert set(sparse.nodes) == set(dense.nodes)
    assert sparse.number_of_edges() < dense.number_of_edges()


def test_dataclass_spec_dispatches_without_location_specific_logic() -> None:
    spec = SyntheticNetworkSpec(
        topology="grid",
        seed=44,
        parameters={"rows": 2, "columns": 3, "spacing_km": 0.4},
    )
    graph = generate_synthetic_network(spec)

    assert graph.graph["topology"] == "grid"
    assert graph.graph["seed"] == 44
    assert graph.number_of_nodes() == 6


def test_spec_rejects_unknown_topology_and_duplicate_seed() -> None:
    with pytest.raises(ValueError, match="unsupported topology"):
        SyntheticNetworkSpec(topology="named_city")

    spec = SyntheticNetworkSpec(topology="grid", parameters={"seed": 7})
    with pytest.raises(ValueError, match="belongs in SyntheticNetworkSpec.seed"):
        generate_synthetic_network(spec)
