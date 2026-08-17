import networkx as nx
import pandas as pd

from src.deadline_risk import RiskWeights
from src.dynamic_rerouting import (
    DynamicEvent,
    apply_dynamic_event,
    choose_dynamic_route,
)
from src.network_schema import EdgeData, NodeData, add_edge, add_node


def _diamond() -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in ("o", "a", "b", "d"):
        add_node(graph, NodeData(node))
    for source, target, time in (
        ("o", "a", 1.0),
        ("a", "d", 1.0),
        ("o", "b", 1.2),
        ("b", "d", 1.2),
    ):
        add_edge(graph, EdgeData(source, target, 0.5, time, road_class="local"))
    return graph


def test_dynamic_event_removes_closure_and_adds_expected_edge_time() -> None:
    graph = _diamond()
    event = DynamicEvent(
        "closure", global_congestion=0.5, closed_edges=frozenset({("o", "a")})
    )
    dynamic = apply_dynamic_event(graph, event)
    assert not dynamic.has_edge("o", "a")
    assert dynamic.edges["o", "b"]["dynamic_expected_time_min"] > 1.2
    assert graph.has_edge("o", "a")


def test_policy_selection_and_quantum_callback_are_explicit() -> None:
    candidates = pd.DataFrame(
        {
            "route_id": ["x_0", "x_1"],
            "prediction": [10.0, 11.0],
            "risk_score": [14.0, 12.0],
        }
    )
    assert choose_dynamic_route(candidates, policy="expected_time_rerouting") == "x_0"
    assert choose_dynamic_route(candidates, policy="uncertainty_aware_classical") == "x_1"
    assert (
        choose_dynamic_route(
            candidates,
            policy="uncertainty_aware_qaoa",
            quantum_selector=lambda table: "x_1",
        )
        == "x_1"
    )
