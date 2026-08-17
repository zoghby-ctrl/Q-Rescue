"""Tests for interpretable network characterization without all-pairs work."""

from collections.abc import Callable

import networkx as nx
import numpy as np
import pytest

from src.candidate_routes import InsufficientRoutesError
from src.network_characterization import characterize_network, characterizations_to_table
from src.network_schema import EdgeData, NodeData, add_edge, add_node
from src.synthetic_networks import (
    generate_dense_mixed_network,
    generate_grid_network,
    generate_irregular_geometric_network,
    generate_radial_network,
    generate_sparse_mixed_network,
)


FAMILY_CASES: tuple[tuple[str, Callable[[], nx.DiGraph]], ...] = (
    ("grid", lambda: generate_grid_network(rows=4, columns=4, seed=23)),
    ("radial", lambda: generate_radial_network(spokes=6, rings=2, seed=23)),
    (
        "irregular_geometric",
        lambda: generate_irregular_geometric_network(node_count=14, seed=23),
    ),
    ("sparse_mixed", lambda: generate_sparse_mixed_network(node_count=16, seed=23)),
    ("dense_mixed", lambda: generate_dense_mixed_network(node_count=16, seed=23)),
)


@pytest.mark.parametrize("family,factory", FAMILY_CASES, ids=[case[0] for case in FAMILY_CASES])
def test_five_synthetic_families_have_interpretable_characteristics(
    family: str,
    factory: Callable[[], nx.DiGraph],
) -> None:
    graph = factory()
    result = characterize_network(graph, route_k=4, require_route_k=True)

    assert result.topology_family == family
    assert result.node_count == graph.number_of_nodes()
    assert result.directed_edge_count == graph.number_of_edges()
    assert np.isclose(result.mean_in_degree, graph.number_of_edges() / graph.number_of_nodes())
    assert np.isclose(result.mean_out_degree, graph.number_of_edges() / graph.number_of_nodes())
    assert sum(result.intersection_degree_distribution.values()) == result.node_count
    assert result.intersection_degree_min <= result.intersection_degree_median
    assert result.intersection_degree_median <= result.intersection_degree_max
    assert result.mean_edge_length_km > 0.0
    assert result.std_edge_length_km >= 0.0
    assert result.cv_edge_length >= 0.0
    assert result.mean_base_travel_time_min > 0.0
    assert result.od_shortest_distance_km > 0.0
    assert result.od_shortest_base_travel_time_min > 0.0
    assert result.available_route_count == 4
    assert result.route_k_complete is True
    assert 0.0 <= result.route_mean_jaccard_diversity <= 1.0
    assert 0.0 <= result.directed_simple_density <= 1.0
    assert "size-dependent" in result.density_caveat


def test_known_path_graph_metrics_are_exact() -> None:
    graph = nx.DiGraph()
    for node_id in ("a", "b", "c"):
        add_node(graph, NodeData(node_id))
    for source, target, distance, travel_time in (
        ("a", "b", 1.0, 2.0),
        ("b", "a", 1.0, 2.0),
        ("b", "c", 2.0, 4.0),
        ("c", "b", 2.0, 4.0),
    ):
        add_edge(graph, EdgeData(source, target, distance, travel_time))
    graph.graph.update(network_id="known", topology_family="path", origin="a", destination="c")

    result = characterize_network(graph, route_k=1, require_route_k=True)

    assert result.node_count == 3
    assert result.directed_edge_count == 4
    assert np.isclose(result.mean_in_degree, 4 / 3)
    assert np.isclose(result.mean_out_degree, 4 / 3)
    assert np.isclose(result.mean_undirected_intersection_degree, 4 / 3)
    assert result.intersection_degree_distribution == {1: 2, 2: 1}
    assert np.isclose(result.mean_edge_length_km, 1.5)
    assert np.isclose(result.mean_base_travel_time_min, 3.0)
    assert np.isclose(result.od_shortest_distance_km, 3.0)
    assert np.isclose(result.od_shortest_base_travel_time_min, 6.0)
    assert result.available_route_count == 1
    assert result.route_mean_jaccard_diversity == 0.0


def test_route_k_completeness_is_explicit_and_configurable() -> None:
    graph = generate_grid_network(rows=2, columns=2, seed=4)
    partial = characterize_network(graph, route_k=8, require_route_k=False)

    assert partial.requested_route_k == 8
    assert partial.available_route_count == 2
    assert partial.route_k_complete is False
    with pytest.raises(InsufficientRoutesError):
        characterize_network(graph, route_k=8, require_route_k=True)


def test_characterization_table_retains_one_row_per_network() -> None:
    grid = characterize_network(generate_grid_network(rows=3, columns=3, seed=5), route_k=4)
    dense = characterize_network(generate_dense_mixed_network(node_count=12, seed=5), route_k=8)
    table = characterizations_to_table([grid, dense])

    assert len(table) == 2
    assert table["topology_family"].tolist() == ["grid", "dense_mixed"]
    assert table["requested_route_k"].tolist() == [4, 8]
    assert "intersection_degree_distribution" in table.columns


def test_missing_od_metadata_and_disconnected_od_are_rejected() -> None:
    graph = nx.DiGraph()
    add_node(graph, NodeData("a"))
    add_node(graph, NodeData("b"))
    add_edge(graph, EdgeData("a", "b", 1.0, 1.0))

    with pytest.raises(ValueError, match="valid origin"):
        characterize_network(graph, route_k=1)
    with pytest.raises(ValueError, match="not connected"):
        characterize_network(graph, origin="b", destination="a", route_k=1)
