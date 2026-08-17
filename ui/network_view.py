"""Geometry-preserving road-network visualization for the Streamlit console."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import networkx as nx

from src.candidate_routes import CandidateRoute
from ui.theme import CYAN, EMBER, GRAPHITE, MUTED, TEXT


def _positions(graph: nx.DiGraph) -> dict[Hashable, tuple[float, float]]:
    if all("x" in values and "y" in values for _, values in graph.nodes(data=True)):
        return {
            node: (float(values["x"]), float(values["y"]))
            for node, values in graph.nodes(data=True)
        }
    generated = nx.spring_layout(graph, seed=2026, weight="distance_km")
    return {node: (float(point[0]), float(point[1])) for node, point in generated.items()}


def _route_line(
    axis: plt.Axes,
    positions: dict[Hashable, tuple[float, float]],
    route: CandidateRoute,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    selected: bool,
) -> None:
    x_values = [positions[node][0] for node in route.nodes]
    y_values = [positions[node][1] for node in route.nodes]
    line = axis.plot(
        x_values,
        y_values,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=6 if selected else 3,
    )[0]
    if selected:
        line.set_path_effects(
            [
                path_effects.Stroke(linewidth=linewidth + 5.0, foreground="#0e596b", alpha=0.32),
                path_effects.Normal(),
            ]
        )


def build_network_figure(
    graph: nx.DiGraph,
    routes: Sequence[CandidateRoute] = (),
    *,
    selected_route_id: str | None = None,
    origin: Hashable | None = None,
    destination: Hashable | None = None,
    disrupted_edge: tuple[Hashable, Hashable] | None = None,
) -> Figure:
    """Build a quiet base graph with candidate and selected-route hierarchy."""

    positions = _positions(graph)
    figure, axis = plt.subplots(figsize=(10.2, 6.25), dpi=130)
    figure.patch.set_facecolor(GRAPHITE)
    axis.set_facecolor(GRAPHITE)

    node_count = graph.number_of_nodes()
    base_width = 0.36 if node_count > 400 else 0.62
    base_alpha = 0.20 if node_count > 400 else 0.30
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        arrows=False,
        edge_color="#65717d",
        width=base_width,
        alpha=base_alpha,
    )

    for route in routes:
        if route.route_id == selected_route_id:
            continue
        _route_line(
            axis,
            positions,
            route,
            color="#8b98a5",
            linewidth=1.25,
            alpha=0.44,
            selected=False,
        )

    selected_route = next(
        (route for route in routes if route.route_id == selected_route_id), None
    )
    if selected_route is not None:
        _route_line(
            axis,
            positions,
            selected_route,
            color=CYAN,
            linewidth=2.7,
            alpha=0.98,
            selected=True,
        )

    if disrupted_edge is not None and all(node in positions for node in disrupted_edge):
        first, second = disrupted_edge
        x_values = [positions[first][0], positions[second][0]]
        y_values = [positions[first][1], positions[second][1]]
        axis.plot(
            x_values,
            y_values,
            color=EMBER,
            linewidth=3.2,
            linestyle=(0, (2, 2)),
            zorder=8,
        )
        axis.scatter(
            [(x_values[0] + x_values[1]) / 2.0],
            [(y_values[0] + y_values[1]) / 2.0],
            marker="x",
            s=62,
            linewidths=2.0,
            color=EMBER,
            zorder=9,
        )

    origin = graph.graph.get("origin") if origin is None else origin
    destination = graph.graph.get("destination") if destination is None else destination
    if origin in positions:
        axis.scatter(
            [positions[origin][0]],
            [positions[origin][1]],
            s=82,
            marker="o",
            facecolor=GRAPHITE,
            edgecolor=CYAN,
            linewidth=2.0,
            zorder=10,
        )
        axis.annotate(
            "ORIGIN",
            positions[origin],
            xytext=(7, 7),
            textcoords="offset points",
            color=TEXT,
            fontsize=7,
            weight="bold",
            zorder=11,
        )
    if destination in positions:
        axis.scatter(
            [positions[destination][0]],
            [positions[destination][1]],
            s=78,
            marker="s",
            facecolor=GRAPHITE,
            edgecolor="#f2f5f7",
            linewidth=1.8,
            zorder=10,
        )
        axis.annotate(
            "DESTINATION",
            positions[destination],
            xytext=(7, 7),
            textcoords="offset points",
            color=TEXT,
            fontsize=7,
            weight="bold",
            zorder=11,
        )

    handles = [
        Line2D([0], [0], color="#65717d", lw=1.0, alpha=0.65, label="Road"),
        Line2D([0], [0], color="#8b98a5", lw=1.5, alpha=0.8, label="Candidate route"),
        Line2D([0], [0], color=CYAN, lw=2.8, label="Recommended route"),
        Line2D(
            [0],
            [0],
            color=EMBER,
            lw=2.8,
            linestyle=(0, (2, 2)),
            label="Disrupted / closed edge",
        ),
    ]
    legend = axis.legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="#101318",
        edgecolor="#2b333d",
        labelcolor=TEXT,
        fontsize=7,
        borderpad=0.65,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)

    title = "Road network and route decision"
    if selected_route_id is not None:
        title += f" · selected {selected_route_id}"
    axis.set_title(title, loc="left", color=TEXT, fontsize=10, weight="bold", pad=8)
    axis.text(
        1.0,
        1.01,
        f"{graph.number_of_nodes():,} nodes · {graph.number_of_edges():,} directed edges",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=7,
    )
    axis.margins(0.035)
    axis.set_axis_off()
    if bool(graph.graph.get("synthetic", False)):
        axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout(pad=0.45)
    return figure


def build_probability_figure(probability_table, *, title: str) -> Figure:
    """Plot one-hot route probabilities with the decoded state labeled."""

    table = probability_table.copy()
    labels = [
        f"{row.route_id}  |  {row.bitstring}" for row in table.itertuples()
    ]
    colors = [CYAN if bool(value) else "#596673" for value in table["selected"]]
    figure, axis = plt.subplots(figsize=(6.6, max(2.3, 0.42 * len(table) + 1.25)), dpi=130)
    figure.patch.set_facecolor(GRAPHITE)
    axis.set_facecolor(GRAPHITE)
    bars = axis.barh(labels[::-1], table["probability"].to_numpy()[::-1], color=colors[::-1])
    axis.set_xlim(0.0, max(1.0, float(table["probability"].max()) * 1.12))
    axis.set_xlabel("Analytic probability", color=MUTED, fontsize=8)
    axis.set_title(title, loc="left", color=TEXT, fontsize=9, weight="bold")
    axis.tick_params(colors=TEXT, labelsize=7)
    axis.grid(axis="x", color="#303842", alpha=0.4, linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, table["probability"].to_numpy()[::-1], strict=True):
        axis.text(
            min(float(value) + 0.015, 0.94),
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.1%}",
            va="center",
            color=TEXT,
            fontsize=7,
        )
    figure.tight_layout(pad=0.8)
    return figure
