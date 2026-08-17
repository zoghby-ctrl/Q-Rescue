"""Plotting functions for the Q-Rescue prototype artifacts."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

from src.route_data import calculate_path_time


ROUTE_COLORS = {
    "Route A": "#2563eb",
    "Route B": "#16a34a",
    "Route C": "#d97706",
    "Route D": "#dc2626",
}


def _save_figure(figure: Figure, output_path: Path) -> None:
    """Save one figure with consistent resolution and close it."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def save_road_network_figure(
    graph: nx.DiGraph,
    candidate_paths: Mapping[str, Sequence[str]],
    output_path: Path,
) -> None:
    """Save four subplots, each highlighting one candidate route."""

    positions = {
        "Ambulance Station": (0.0, 0.0),
        "Junction A": (1.0, 1.5),
        "Junction B": (1.0, 0.5),
        "Junction C": (1.0, -0.5),
        "Junction D": (1.0, -1.5),
        "Hospital": (2.0, 0.0),
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    edge_labels = nx.get_edge_attributes(graph, "time_minutes")

    for axis, (route_name, path) in zip(axes.flat, candidate_paths.items()):
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx(
            graph,
            positions,
            ax=axis,
            node_color="#e2e8f0",
            node_size=1750,
            font_size=8,
            arrows=True,
            edge_color="#cbd5e1",
            width=1.5,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=path_edges,
            edge_color=ROUTE_COLORS[route_name],
            width=4.0,
            arrows=True,
            arrowsize=20,
        )
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            ax=axis,
            edge_labels={edge: f"{time:g} min" for edge, time in edge_labels.items()},
            font_size=7,
        )
        total_time = calculate_path_time(graph, path)
        axis.set_title(f"{route_name}: {total_time:g} expected minutes")
        axis.set_axis_off()

    figure.suptitle("Q-Rescue synthetic directed road network", fontsize=16)
    figure.tight_layout()
    _save_figure(figure, output_path)


def save_route_comparison_figure(route_data: pd.DataFrame, output_path: Path) -> None:
    """Save grouped bars comparing expected time and risk score by route."""

    positions = np.arange(len(route_data))
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.5))
    expected_bars = axis.bar(
        positions - bar_width / 2,
        route_data["expected_time_minutes"],
        bar_width,
        label="Expected time",
        color="#60a5fa",
    )
    risk_bars = axis.bar(
        positions + bar_width / 2,
        route_data["risk_score"],
        bar_width,
        label="Risk score",
        color="#f59e0b",
    )
    axis.bar_label(expected_bars, fmt="%.0f")
    axis.bar_label(risk_bars, fmt="%.0f")
    axis.set_xticks(positions, route_data["route"])
    axis.set_ylabel("Minutes-equivalent score")
    axis.set_title("Expected-time baseline and risk-aware score")
    axis.legend()
    axis.set_ylim(0, float(route_data["risk_score"].max()) + 6)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, output_path)


def save_qubo_matrix_figure(qubo_matrix: pd.DataFrame, output_path: Path) -> None:
    """Save an annotated heatmap of the symmetric QUBO matrix."""

    values = qubo_matrix.to_numpy(dtype=float)
    color_limit = float(np.abs(values).max())
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="coolwarm", vmin=-color_limit, vmax=color_limit)
    axis.set_xticks(range(len(qubo_matrix.columns)), qubo_matrix.columns)
    axis.set_yticks(range(len(qubo_matrix.index)), qubo_matrix.index)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                color="black",
            )
    axis.set_title("Symmetric QUBO matrix Q in C = constant + xᵀQx")
    figure.colorbar(image, ax=axis, label="Coefficient")
    figure.tight_layout()
    _save_figure(figure, output_path)


def save_qaoa_circuit_figure(
    qnode: qml.QNode,
    gammas: pnp.ndarray,
    betas: pnp.ndarray,
    output_path: Path,
) -> None:
    """Draw and save the PennyLane QAOA circuit for supplied angles."""

    figure, axis = qml.draw_mpl(qnode, decimals=None, fontsize=15)(gammas, betas)
    figure.set_size_inches(14, 5)
    axis.set_title(
        "Analytic p=1 QAOA: H → cost evolution U_C(γ) → mixer evolution U_M(β) → ⟨H_C⟩",
        pad=22,
    )
    figure.subplots_adjust(top=0.82, bottom=0.08)
    _save_figure(figure, output_path)


def save_training_history_figure(
    histories: Mapping[int, pd.DataFrame],
    output_path: Path,
) -> None:
    """Save the analytic cost-expectation history for each QAOA depth."""

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for depth, history in histories.items():
        axis.plot(
            history["iteration"],
            history["cost_expectation"],
            label=f"p = {depth}",
            linewidth=2,
        )
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Cost Hamiltonian expectation (offset omitted)")
    axis.set_title("QAOA classical parameter optimization")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def save_measurement_probabilities_figure(
    summaries: Mapping[int, Mapping[str, object]],
    optimal_bitstring: str,
    output_path: Path,
) -> None:
    """Save all measured bitstring probabilities for each QAOA depth."""

    number_of_plots = len(summaries)
    figure, axes = plt.subplots(number_of_plots, 1, figsize=(13, 4.5 * number_of_plots))
    axes_array = np.atleast_1d(axes)

    for axis, (depth, summary) in zip(axes_array, summaries.items()):
        counts = summary["counts"]
        if not isinstance(counts, dict):
            raise TypeError("summary['counts'] must be a dictionary")
        bitstrings = list(counts.keys())
        total_shots = sum(int(count) for count in counts.values())
        probabilities = [int(counts[bitstring]) / total_shots for bitstring in bitstrings]
        colors = []
        for bitstring in bitstrings:
            if bitstring == optimal_bitstring:
                colors.append("#16a34a")
            elif bitstring.count("1") == 1:
                colors.append("#f59e0b")
            else:
                colors.append("#94a3b8")
        axis.bar(bitstrings, probabilities, color=colors)
        axis.set_ylabel("Measured probability")
        axis.set_title(f"Finite-shot QAOA distribution at p = {depth}")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.25)

    legend_items = [
        Patch(color="#16a34a", label="Optimal valid bitstring"),
        Patch(color="#f59e0b", label="Other valid bitstring"),
        Patch(color="#94a3b8", label="Invalid bitstring"),
    ]
    axes_array[0].legend(handles=legend_items)
    figure.tight_layout()
    _save_figure(figure, output_path)
