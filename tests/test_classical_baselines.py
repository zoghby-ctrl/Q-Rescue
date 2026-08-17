"""Tests for deterministic classical route-selection foundations."""

import networkx as nx
import pandas as pd
import pytest

from src.classical_baselines import (
    RESULT_COLUMNS,
    UnreachablePathError,
    baseline_results_to_table,
    dijkstra_expected_time_path,
    dijkstra_shortest_distance_path,
    evaluate_candidate_baselines,
    select_exact_risk_candidate,
    select_shortest_distance_candidate,
    select_shortest_expected_time_candidate,
    select_upper_bound_heuristic,
)


def _candidate_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route_id": "x_0",
                "ordered_nodes": ("s", "a", "t"),
                "distance_km": 4.0,
                "expected_travel_time_min": 12.0,
                "upper_bound": 15.0,
                "risk_score": 14.0,
            },
            {
                "route_id": "x_1",
                "ordered_nodes": ("s", "b", "t"),
                "distance_km": 5.0,
                "expected_travel_time_min": 8.0,
                "upper_bound": 14.0,
                "risk_score": 13.0,
            },
            {
                "route_id": "x_2",
                "ordered_nodes": ("s", "c", "t"),
                "distance_km": 6.0,
                "expected_travel_time_min": 10.0,
                "upper_bound": 12.0,
                "risk_score": 9.0,
            },
            {
                "route_id": "x_3",
                "ordered_nodes": ("s", "d", "t"),
                "distance_km": 7.0,
                "expected_travel_time_min": 11.0,
                "upper_bound": 11.5,
                "risk_score": 10.0,
            },
        ]
    )


def test_candidate_baselines_can_select_different_routes() -> None:
    """Each baseline must optimize its stated column, not a shared shortcut."""

    candidates = _candidate_table()
    distance = select_shortest_distance_candidate(candidates)
    expected = select_shortest_expected_time_candidate(candidates)
    exact_risk = select_exact_risk_candidate(candidates)
    upper = select_upper_bound_heuristic(candidates)

    assert distance.route_id == "x_0"
    assert expected.route_id == "x_1"
    assert exact_risk.route_id == "x_2"
    assert upper.route_id == "x_3"
    assert exact_risk.objective_value == 9.0
    assert exact_risk.candidate_count == 4
    assert exact_risk.path == ("s", "c", "t")

    comparison = evaluate_candidate_baselines(candidates)
    assert tuple(comparison.columns) == RESULT_COLUMNS
    assert comparison["route_id"].tolist() == ["x_0", "x_1", "x_2", "x_3"]


def test_ties_use_numeric_decision_variable_order_independent_of_rows() -> None:
    """x_2 must precede x_10, not depend on lexical or DataFrame order."""

    tied = pd.DataFrame(
        [
            {
                "route_id": "x_10",
                "ordered_nodes": ("s", "z", "t"),
                "distance_km": 5.0,
                "expected_travel_time_min": 7.0,
                "upper_bound": 9.0,
                "risk_score": 8.0,
            },
            {
                "route_id": "x_2",
                "ordered_nodes": ("s", "a", "t"),
                "distance_km": 5.0,
                "expected_travel_time_min": 7.0,
                "upper_bound": 9.0,
                "risk_score": 8.0,
            },
        ]
    )

    assert select_exact_risk_candidate(tied).route_id == "x_2"
    assert select_exact_risk_candidate(tied.iloc[::-1]).route_id == "x_2"


@pytest.mark.parametrize("candidate_count", [1, 4, 6, 8, 11])
def test_candidate_selectors_support_arbitrary_k(candidate_count: int) -> None:
    rows = [
        {
            "route_id": f"x_{index}",
            "ordered_nodes": ("s", f"n_{index}", "t"),
            "distance_km": float(index + 1),
            "expected_travel_time_min": float(candidate_count - index + 1),
            "upper_bound": float(candidate_count - index + 2),
            "risk_score": float(index + 3),
        }
        for index in range(candidate_count)
    ]
    candidates = pd.DataFrame(rows)

    assert select_shortest_distance_candidate(candidates).candidate_count == candidate_count
    assert select_shortest_expected_time_candidate(candidates).route_id == f"x_{candidate_count - 1}"
    assert select_exact_risk_candidate(candidates).route_id == "x_0"


def _dijkstra_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    # Insert branch b first to ensure graph insertion order is not the tie break.
    graph.add_edge(
        "s", "b", distance_km=1.0, base_travel_time_min=2.0, predicted_time=3.0
    )
    graph.add_edge(
        "b", "t", distance_km=1.0, base_travel_time_min=2.0, predicted_time=3.0
    )
    graph.add_edge(
        "s", "a", distance_km=2.0, base_travel_time_min=2.0, predicted_time=3.0
    )
    graph.add_edge(
        "a", "t", distance_km=2.0, base_travel_time_min=2.0, predicted_time=3.0
    )
    return graph


def test_dijkstra_expected_time_path_has_deterministic_tie_break_and_trace() -> None:
    result = dijkstra_expected_time_path(
        _dijkstra_graph(), "s", "t", weight="predicted_time"
    )

    assert result.path == ("s", "a", "t")
    assert result.objective_value == 6.0
    assert result.expected_travel_time_min == 6.0
    assert result.distance_km == 4.0
    assert result.route_id is None


def test_graph_wide_shortest_distance_reports_distance_and_travel_time() -> None:
    graph = nx.DiGraph()
    graph.add_edge(
        "s", "short", distance_km=1.0, base_travel_time_min=10.0
    )
    graph.add_edge(
        "short", "t", distance_km=1.0, base_travel_time_min=10.0
    )
    graph.add_edge(
        "s", "fast", distance_km=3.0, base_travel_time_min=2.0
    )
    graph.add_edge(
        "fast", "t", distance_km=3.0, base_travel_time_min=2.0
    )

    distance = dijkstra_shortest_distance_path(graph, "s", "t")
    expected_time = dijkstra_expected_time_path(graph, "s", "t")

    assert distance.method == "dijkstra_shortest_distance_path"
    assert distance.path == ("s", "short", "t")
    assert distance.objective_name == "distance_km"
    assert distance.objective_value == 2.0
    assert distance.distance_km == 2.0
    assert distance.expected_travel_time_min == 20.0
    assert expected_time.path == ("s", "fast", "t")
    assert expected_time.objective_value == 4.0
    assert expected_time.distance_km == 6.0


def test_dijkstra_tie_handling_does_not_enumerate_all_paths(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("all_shortest_paths must not be enumerated")

    monkeypatch.setattr(nx, "all_shortest_paths", fail_if_called)
    result = dijkstra_expected_time_path(
        _dijkstra_graph(), "s", "t", weight="predicted_time"
    )
    assert result.path == ("s", "a", "t")


@pytest.mark.parametrize(
    "metadata_field",
    ["loader_parallel_edges_collapsed", "parallel_edges_collapsed"],
)
def test_distance_baseline_requires_explicit_parallel_projection_opt_in(
    metadata_field: str,
) -> None:
    graph = _dijkstra_graph()
    graph.graph[metadata_field] = 1

    with pytest.raises(ValueError, match="fastest-base-time projection"):
        dijkstra_shortest_distance_path(graph, "s", "t")

    projected = dijkstra_shortest_distance_path(
        graph, "s", "t", allow_parallel_edge_projection=True
    )
    assert projected.method.endswith("on_fastest_time_projection")


def test_unreachable_dijkstra_path_is_reported_explicitly() -> None:
    graph = _dijkstra_graph()
    graph.add_node("isolated")
    with pytest.raises(UnreachablePathError, match="no directed path"):
        dijkstra_expected_time_path(graph, "s", "isolated")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns="risk_score"), "missing required columns"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "must be unique"),
        (lambda frame: frame.assign(risk_score=float("nan")), "must be finite"),
        (
            lambda frame: frame.assign(
                ordered_nodes=[("s", "a", "s")] * len(frame)
            ),
            "loopless",
        ),
    ],
)
def test_invalid_candidate_tables_are_rejected(mutator, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select_exact_risk_candidate(mutator(_candidate_table()))


def test_missing_graph_weight_and_empty_result_table_are_rejected() -> None:
    graph = _dijkstra_graph()
    with pytest.raises(ValueError, match="missing weight"):
        dijkstra_expected_time_path(graph, "s", "t", weight="not_present")
    with pytest.raises(ValueError, match="at least one"):
        baseline_results_to_table([])
