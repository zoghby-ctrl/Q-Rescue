"""Tests for generalized, deterministic candidate-route generation."""

import networkx as nx
import numpy as np
import pytest

from src.candidate_routes import (
    CandidateRoute,
    InsufficientRoutesError,
    edge_jaccard_similarity,
    generate_candidate_routes,
    pairwise_route_diversity_table,
    route_edge_set,
    routes_to_table,
    summarize_route_diversity,
)
from src.network_schema import EdgeData, NetworkSchemaError, NodeData, add_edge, add_node
from src.synthetic_networks import generate_dense_mixed_network, generate_grid_network


@pytest.mark.parametrize("k", [4, 6, 8])
def test_generated_network_supports_requested_candidate_counts(k: int) -> None:
    graph = generate_dense_mixed_network(node_count=16, seed=305)
    routes = generate_candidate_routes(
        graph,
        graph.graph["origin"],
        graph.graph["destination"],
        k,
    )

    assert len(routes) == k
    assert [route.route_id for route in routes] == [f"x_{index}" for index in range(k)]
    assert all(len(route.nodes) == len(set(route.nodes)) for route in routes)
    assert all(route.edge_count == len(route.edges) == len(route.nodes) - 1 for route in routes)
    assert all(route.nodes[0] == graph.graph["origin"] for route in routes)
    assert all(route.nodes[-1] == graph.graph["destination"] for route in routes)
    assert [route.base_travel_time_min for route in routes] == sorted(
        route.base_travel_time_min for route in routes
    )


def test_route_summaries_equal_edge_attribute_sums() -> None:
    graph = generate_grid_network(rows=4, columns=4, seed=12)
    for source, target, attributes in graph.edges(data=True):
        attributes["expected_time_min"] = 1.25 * attributes["base_travel_time_min"]

    routes = generate_candidate_routes(
        graph,
        graph.graph["origin"],
        graph.graph["destination"],
        4,
        weight="expected_time_min",
        expected_time_attribute="expected_time_min",
    )
    table = routes_to_table(routes)

    assert table["route_id"].tolist() == ["x_0", "x_1", "x_2", "x_3"]
    assert list(table.columns) == [
        "route_id",
        "ordered_nodes",
        "edges",
        "distance_km",
        "base_travel_time_min",
        "expected_travel_time_min",
        "edge_count",
    ]
    for route in routes:
        expected_distance = sum(graph.edges[edge]["distance_km"] for edge in route.edges)
        expected_base_time = sum(
            graph.edges[edge]["base_travel_time_min"] for edge in route.edges
        )
        assert np.isclose(route.distance_km, expected_distance)
        assert np.isclose(route.base_travel_time_min, expected_base_time)
        assert np.isclose(route.expected_travel_time_min, 1.25 * expected_base_time)


def test_route_order_is_deterministic_across_graph_insertion_orders() -> None:
    graph = generate_grid_network(rows=4, columns=4, seed=77)
    reversed_graph = nx.DiGraph()
    reversed_graph.graph.update(graph.graph)
    for node_id, attributes in reversed(list(graph.nodes(data=True))):
        reversed_graph.add_node(node_id, **attributes)
    for source, target, attributes in reversed(list(graph.edges(data=True))):
        reversed_graph.add_edge(source, target, **attributes)

    first = generate_candidate_routes(
        graph, graph.graph["origin"], graph.graph["destination"], 8
    )
    second = generate_candidate_routes(
        reversed_graph,
        reversed_graph.graph["origin"],
        reversed_graph.graph["destination"],
        8,
    )

    assert [route.nodes for route in first] == [route.nodes for route in second]
    assert routes_to_table(first).equals(routes_to_table(second))


def test_equal_cost_ties_use_canonical_node_order() -> None:
    graph = nx.DiGraph()
    for node_id in ["destination", "branch_b", "branch_a", "origin"]:
        add_node(graph, NodeData(node_id))
    for source, target in [
        ("origin", "branch_b"),
        ("branch_b", "destination"),
        ("origin", "branch_a"),
        ("branch_a", "destination"),
    ]:
        add_edge(
            graph,
            EdgeData(source, target, distance_km=1.0, base_travel_time_min=1.0),
        )

    routes = generate_candidate_routes(graph, "origin", "destination", 2)
    assert routes[0].nodes == ("origin", "branch_a", "destination")
    assert routes[1].nodes == ("origin", "branch_b", "destination")


def test_insufficient_routes_are_never_silently_reported_as_k() -> None:
    graph = generate_grid_network(rows=2, columns=2, seed=8)
    origin = graph.graph["origin"]
    destination = graph.graph["destination"]

    with pytest.raises(InsufficientRoutesError) as caught:
        generate_candidate_routes(graph, origin, destination, 4)
    assert caught.value.requested_k == 4
    assert caught.value.available_count == 2

    available = generate_candidate_routes(graph, origin, destination, 4, require_k=False)
    assert len(available) == 2
    assert [route.route_id for route in available] == ["x_0", "x_1"]


def test_edge_set_and_jaccard_diversity_helpers() -> None:
    graph = generate_grid_network(rows=4, columns=4, seed=56)
    routes = generate_candidate_routes(
        graph, graph.graph["origin"], graph.graph["destination"], 4
    )

    first_edges = route_edge_set(routes[0])
    second_edges = route_edge_set(routes[1])
    expected_similarity = len(first_edges & second_edges) / len(first_edges | second_edges)
    pairwise = pairwise_route_diversity_table(routes)
    summary = summarize_route_diversity(routes)

    assert edge_jaccard_similarity(routes[0], routes[1]) == expected_similarity
    assert len(pairwise) == 6
    assert np.allclose(
        pairwise["jaccard_similarity"] + pairwise["jaccard_diversity"], 1.0
    )
    assert summary["route_count"] == 4
    assert summary["pair_count"] == 6
    assert summary["unique_directed_edge_count"] == len(
        set().union(*(route_edge_set(route) for route in routes))
    )
    assert np.isclose(
        summary["mean_jaccard_diversity"], pairwise["jaccard_diversity"].mean()
    )


@pytest.mark.parametrize("bad_k", [0, -1, 1.5, True])
def test_invalid_k_is_rejected(bad_k: object) -> None:
    graph = generate_grid_network(rows=3, columns=3, seed=1)
    with pytest.raises(ValueError, match="positive integer"):
        generate_candidate_routes(
            graph,
            graph.graph["origin"],
            graph.graph["destination"],
            bad_k,  # type: ignore[arg-type]
        )


def test_invalid_nodes_and_missing_custom_weight_are_rejected() -> None:
    graph = generate_grid_network(rows=3, columns=3, seed=1)
    with pytest.raises(ValueError, match="origin node"):
        generate_candidate_routes(graph, "missing", graph.graph["destination"], 2)
    with pytest.raises(ValueError, match="missing weight attribute"):
        generate_candidate_routes(
            graph,
            graph.graph["origin"],
            graph.graph["destination"],
            2,
            weight="predicted_time_min",
        )


def test_invalid_graph_schema_and_repeated_route_nodes_are_rejected() -> None:
    invalid_graph = nx.DiGraph()
    invalid_graph.add_edge("a", "b", distance_km=1.0)
    with pytest.raises(NetworkSchemaError, match="base_travel_time_min"):
        generate_candidate_routes(invalid_graph, "a", "b", 1)

    with pytest.raises(ValueError, match="loopless"):
        CandidateRoute(
            route_id="x_0",
            nodes=("a", "b", "a"),
            edges=(("a", "b"), ("b", "a")),
            distance_km=2.0,
            base_travel_time_min=3.0,
            expected_travel_time_min=3.0,
            edge_count=2,
        )
