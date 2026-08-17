from __future__ import annotations

import numpy as np

from ui.backend import (
    METHOD_LABELS,
    available_k_values,
    load_model_bundles,
    load_network,
    load_network_catalog,
    run_dynamic_demo,
    run_pipeline,
)
from ui.network_view import build_network_figure


def test_network_catalog_exposes_five_synthetic_and_cached_real_cases() -> None:
    catalog = load_network_catalog()
    synthetic = [item for item in catalog.values() if not item.is_real]
    real = [item for item in catalog.values() if item.is_real]
    assert {item.topology for item in synthetic} == {
        "grid",
        "radial",
        "irregular_geometric",
        "sparse_mixed",
        "dense_mixed",
    }
    assert len(real) == 4
    assert all(item.study_label == "Real road topology + simulated traffic" for item in real)


def test_default_pipeline_calls_existing_backend_end_to_end() -> None:
    graph = load_network("synthetic:grid")
    bundle = load_model_bundles()["synthetic"]
    result = run_pipeline(
        graph,
        bundle,
        origin=graph.graph["origin"],
        destination=graph.graph["destination"],
        scenario_name="rain",
        deadline_minutes=8.0,
        k=4,
        selected_method="feasible_xy",
        qaoa_depth=1,
        qaoa_iterations=2,
    )

    assert len(result.routes) == 4
    assert len(result.exact_assignments) == 16
    assert set(result.comparison_table["method"]) == set(METHOD_LABELS.values())
    assert result.selected_route_id in set(result.route_table["route_id"])
    assert result.route_table["selected"].sum() == 1
    assert {
        "prediction",
        "lower_bound",
        "upper_bound",
        "uncertainty_width",
        "deadline_margin",
        "risk_score",
    }.issubset(result.route_table.columns)
    assert np.isclose(result.quantum_results["feasible_xy"].valid_probability, 1.0)
    assert 0.0 <= result.quantum_results["standard_x"].valid_probability <= 1.0
    assert result.lowest_risk_route_id == result.route_table.sort_values(
        ["risk_score", "route_id"], kind="stable"
    ).iloc[0]["route_id"]


def test_larger_k_pipeline_and_dynamic_closure_use_existing_backend() -> None:
    graph = load_network("synthetic:grid")
    supported = available_k_values(
        graph, graph.graph["origin"], graph.graph["destination"]
    )
    assert 4 in supported
    assert all(value in {4, 6, 8} for value in supported)

    larger_k = max(supported)
    bundle = load_model_bundles()["synthetic"]
    result = run_pipeline(
        graph,
        bundle,
        origin=graph.graph["origin"],
        destination=graph.graph["destination"],
        scenario_name="normal",
        deadline_minutes=8.0,
        k=larger_k,
        selected_method="exact_risk",
        qaoa_depth=1,
        qaoa_iterations=1,
    )
    assert len(result.routes) == larger_k
    assert len(result.exact_assignments) == 2**larger_k

    dynamic = run_dynamic_demo(
        result,
        bundle,
        event_name="closure",
        qaoa_iterations=1,
    )
    assert dynamic.disruption_edge is not None
    assert dynamic.route_before_event is not None
    assert dynamic.updated_route is not None
    assert dynamic.reroute_count >= 0


def test_cached_real_network_preserves_geometry_and_renders() -> None:
    graph = load_network("real:real_orthogonal_grid_00")
    assert graph.graph["synthetic"] is False
    assert graph.graph["traffic_semantics"] == "controlled_simulated_traffic_not_live_traffic"
    assert all("x" in values and "y" in values for _, values in graph.nodes(data=True))
    figure = build_network_figure(
        graph,
        origin=graph.graph["origin"],
        destination=graph.graph["destination"],
    )
    assert figure.axes
    figure.clear()
