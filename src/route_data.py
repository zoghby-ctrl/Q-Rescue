"""Synthetic route data and a small directed road network."""

from collections.abc import Sequence

import networkx as nx
import pandas as pd


def create_route_dataframe() -> pd.DataFrame:
    """Return the four candidate routes and their manually simulated inputs."""

    route_rows = [
        {"route": "Route A", "expected_time_minutes": 18.0, "uncertainty_minutes": 10.0},
        {"route": "Route B", "expected_time_minutes": 21.0, "uncertainty_minutes": 2.0},
        {"route": "Route C", "expected_time_minutes": 24.0, "uncertainty_minutes": 3.0},
        {"route": "Route D", "expected_time_minutes": 20.0, "uncertainty_minutes": 7.0},
    ]
    return pd.DataFrame(route_rows)


def create_road_network() -> nx.DiGraph:
    """Return a directed graph whose four path times match the route table."""

    graph = nx.DiGraph()
    graph.add_edge("Ambulance Station", "Junction A", time_minutes=8.0)
    graph.add_edge("Junction A", "Hospital", time_minutes=10.0)
    graph.add_edge("Ambulance Station", "Junction B", time_minutes=10.0)
    graph.add_edge("Junction B", "Hospital", time_minutes=11.0)
    graph.add_edge("Ambulance Station", "Junction C", time_minutes=11.0)
    graph.add_edge("Junction C", "Hospital", time_minutes=13.0)
    graph.add_edge("Ambulance Station", "Junction D", time_minutes=9.0)
    graph.add_edge("Junction D", "Hospital", time_minutes=11.0)
    return graph


def get_candidate_paths() -> dict[str, list[str]]:
    """Return each route name and its ordered list of graph nodes."""

    return {
        "Route A": ["Ambulance Station", "Junction A", "Hospital"],
        "Route B": ["Ambulance Station", "Junction B", "Hospital"],
        "Route C": ["Ambulance Station", "Junction C", "Hospital"],
        "Route D": ["Ambulance Station", "Junction D", "Hospital"],
    }


def calculate_path_time(graph: nx.DiGraph, path: Sequence[str]) -> float:
    """Add the edge travel times along one ordered path."""

    total_minutes = 0.0
    for start_node, end_node in zip(path[:-1], path[1:]):
        total_minutes += float(graph.edges[start_node, end_node]["time_minutes"])
    return total_minutes

