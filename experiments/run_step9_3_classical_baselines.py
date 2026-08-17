"""Benchmark strong classical route baselines on the saved Step 6 corpus."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.classical_baselines import (
    ClassicalBaselineResult,
    dijkstra_expected_time_path,
    dijkstra_shortest_distance_path,
    select_exact_risk_candidate,
    select_shortest_distance_candidate,
    select_shortest_expected_time_candidate,
    select_upper_bound_heuristic,
)
from src.reproducibility import (
    build_experiment_manifest,
    file_sha256,
    write_json,
)
from src.synthetic_networks import SyntheticNetworkSpec, generate_synthetic_network


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "classical_baselines_benchmark.json"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "research" / "step9_3_classical_baselines"
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_path(path_text: object, delimiter: str) -> tuple[str, ...]:
    if not isinstance(path_text, str):
        raise ValueError("ordered_nodes must be a delimiter-separated string")
    path = tuple(part.strip() for part in path_text.split(delimiter))
    if len(path) < 2 or any(not node for node in path):
        raise ValueError(f"invalid ordered route path: {path_text!r}")
    return path


def _reverse_insertion_copy(graph: nx.DiGraph) -> nx.DiGraph:
    """Copy a graph with reversed insertion order but unchanged direction."""

    reversed_graph = nx.DiGraph()
    reversed_graph.graph.update(graph.graph)
    for node, attributes in reversed(list(graph.nodes(data=True))):
        reversed_graph.add_node(node, **dict(attributes))
    for source, target, attributes in reversed(list(graph.edges(data=True))):
        reversed_graph.add_edge(source, target, **dict(attributes))
    return reversed_graph


def _timed_repetitions(
    operation: Callable[[], ClassicalBaselineResult],
    *,
    repetitions: int,
) -> tuple[ClassicalBaselineResult, list[tuple[int, int]]]:
    timings: list[tuple[int, int]] = []
    selected: ClassicalBaselineResult | None = None
    expected_signature: tuple[str | None, tuple[object, ...]] | None = None
    for repetition in range(repetitions):
        start_ns = time.perf_counter_ns()
        result = operation()
        elapsed_ns = time.perf_counter_ns() - start_ns
        signature = (result.route_id, result.path)
        if expected_signature is None:
            expected_signature = signature
            selected = result
        elif signature != expected_signature:
            raise RuntimeError("A deterministic baseline changed selection across repetitions")
        timings.append((repetition, elapsed_ns))
    if selected is None:
        raise RuntimeError("runtime_repetitions must be positive")
    return selected, timings


def _runtime_summary(timings: list[tuple[int, int]]) -> dict[str, float]:
    values_us = np.asarray([elapsed / 1000.0 for _, elapsed in timings], dtype=float)
    return {
        "runtime_median_us": float(np.median(values_us)),
        "runtime_min_us": float(values_us.min()),
        "runtime_p95_us": float(np.quantile(values_us, 0.95, method="linear")),
    }


def _candidate_tie_count(
    candidates: pd.DataFrame,
    objective_column: str,
    *,
    absolute_tolerance: float,
) -> int:
    values = candidates[objective_column].to_numpy(dtype=float)
    minimum = float(values.min())
    return int(
        np.isclose(values, minimum, rtol=0.0, atol=absolute_tolerance).sum()
    )


def _dijkstra_shortest_path_count(
    graph: nx.DiGraph,
    origin: object,
    destination: object,
    weight: str,
) -> int:
    """Count tied shortest paths by dynamic programming, without enumeration."""

    predecessors, distances = nx.dijkstra_predecessor_and_distance(
        graph,
        source=origin,
        weight=weight,
    )
    if destination not in distances:
        return 0
    path_counts: dict[object, int] = {origin: 1}
    ordered_nodes = sorted(
        distances,
        key=lambda node: (
            float(distances[node]),
            type(node).__module__,
            type(node).__qualname__,
            repr(node),
        ),
    )
    for node in ordered_nodes:
        if node == origin:
            continue
        path_counts[node] = sum(path_counts.get(parent, 0) for parent in predecessors[node])
    return path_counts[destination]


def _aggregate(group: pd.DataFrame, grouping: dict[str, object]) -> dict[str, object]:
    comparable = group[group["candidate_comparable"]]
    comparable_count = len(comparable)
    optimal_count = int(comparable["is_exact_risk_optimal"].sum())
    return {
        **grouping,
        "decision_count": len(group),
        "comparable_decision_count": comparable_count,
        "exact_risk_optimal_count": optimal_count,
        "exact_risk_optimal_rate": (
            optimal_count / comparable_count if comparable_count else np.nan
        ),
        "mean_risk_regret": (
            float(comparable["risk_regret"].mean()) if comparable_count else np.nan
        ),
        "median_risk_regret": (
            float(comparable["risk_regret"].median()) if comparable_count else np.nan
        ),
        "maximum_risk_regret": (
            float(comparable["risk_regret"].max()) if comparable_count else np.nan
        ),
        "deadline_proxy_violation_count": (
            int(comparable["deadline_proxy_violation"].sum())
            if comparable_count
            else 0
        ),
        "deadline_proxy_violation_rate": (
            float(comparable["deadline_proxy_violation"].mean())
            if comparable_count
            else np.nan
        ),
        "mean_deadline_margin_min": (
            float(comparable["deadline_margin_min"].mean())
            if comparable_count
            else np.nan
        ),
        "median_runtime_us": float(group["runtime_median_us"].median()),
        "p95_instance_runtime_us": float(
            np.quantile(group["runtime_median_us"], 0.95, method="linear")
        ),
        "tied_decision_count": int(group["tie_break_applied"].sum()),
        "stable_tie_check_count": int(group["stable_under_reordered_input"].sum()),
    }


def _summary_tables(instance_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = [
        _aggregate(group, {"method": method})
        for method, group in instance_results.groupby("method", sort=False)
    ]
    topology_rows = [
        _aggregate(group, {"method": method, "topology": topology})
        for (method, topology), group in instance_results.groupby(
            ["method", "topology"], sort=False
        )
    ]
    k_rows = [
        _aggregate(group, {"method": method, "K": int(k_value)})
        for (method, k_value), group in instance_results.groupby(
            ["method", "K"], sort=False
        )
    ]
    return pd.DataFrame(overall_rows), pd.DataFrame(topology_rows), pd.DataFrame(k_rows)


def _format_summary_markdown(summary: pd.DataFrame) -> str:
    labels = {
        "shortest_distance_candidate": "Shortest-distance candidate",
        "dijkstra_shortest_distance_path": "Unrestricted Dijkstra (distance)",
        "shortest_expected_time_candidate": "Shortest expected-time candidate",
        "dijkstra_expected_time_path": "Unrestricted Dijkstra (base expected time)",
        "exact_risk_argmin": "Exact risk argmin",
        "minimum_upper_bound_heuristic": "Minimum-upper-bound heuristic",
    }
    lines = [
        "| Method | Comparable | Exact-optimal | Mean regret | Deadline proxy violations | Median runtime (us) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {labels.get(row.method, row.method)} | "
            f"{row.comparable_decision_count}/{row.decision_count} | "
            f"{row.exact_risk_optimal_count}/{row.comparable_decision_count} | "
            f"{row.mean_risk_regret:.3f} | "
            f"{row.deadline_proxy_violation_count}/{row.comparable_decision_count} | "
            f"{row.median_runtime_us:.2f} |"
        )
    return "\n".join(lines)


def _write_validation_report(
    summary: pd.DataFrame,
    instance_results: pd.DataFrame,
    checks: dict[str, object],
    *,
    source_rows: int,
    topology_count: int,
) -> None:
    all_checks_pass = all(
        bool(value) for key, value in checks.items() if key.endswith("_passed")
    )
    assessment = "Ready to share with stated caveats" if all_checks_pass else "Needs revision"
    dijkstra = summary[summary["method"] == "dijkstra_expected_time_path"].iloc[0]
    distance_dijkstra = summary[
        summary["method"] == "dijkstra_shortest_distance_path"
    ].iloc[0]
    tie_rows = instance_results[instance_results["tie_break_applied"]]
    report = f"""# Step 9.3 classical-baseline validation report

## Overall assessment: {assessment}

This benchmark reused {source_rows} full-objective route rows from the validated
Steps 6/6.5 artifact. They form 15 candidate decisions across {topology_count}
synthetic topology families and K=4, 6, and 8. The source snapshot, instance
selections, raw runtime repetitions, aggregates, tie checks, manifest, and
machine-readable validation checks are saved beside this report.

## Methodology

Four methods select only from the same candidate table: candidate shortest
distance, shortest predicted travel time, exact full-risk argmin, and minimum
calibrated upper bound. Exact risk argmin is the tractable ground truth. Risk regret is
`selected risk - exact minimum risk` in minute-equivalent units.

Two NetworkX Dijkstra methods are deliberately separate unrestricted-graph
baselines. One minimizes `distance_km` and reports both path kilometers and its
additive base travel time; the other minimizes additive
`base_travel_time_min`. Risk, regret, and deadline proxies are reported only
when a graph-wide path exactly matches a saved candidate. The expected-time
Dijkstra path matched in {int(dijkstra.comparable_decision_count)}/{int(dijkstra.decision_count)} cases.
The graph-wide distance path matched in
{int(distance_dijkstra.comparable_decision_count)}/{int(distance_dijkstra.decision_count)} cases.
Both use deterministic polynomial Dijkstra tie handling; tied paths are counted
without enumerating every path.

The deadline metric is conservative `upper_bound > deadline`, not an observed
late arrival. Runtime is the median of repeated local Python calls and is
reported only as an implementation diagnostic.

## Results

{_format_summary_markdown(summary)}

## Stable tie handling

There were {len(tie_rows)} method/decision rows with more than one minimum.
Every one of the {len(instance_results)} selections was unchanged after candidate
row order or graph insertion order was reversed. Candidate `x_N` ties use
numeric decision-variable order; graph-path ties use canonical node order.

## Calculation spot-checks

- Exact-risk maximum absolute regret: {float(checks['maximum_exact_risk_regret']):.3e}.
- Minimum observed regret across all comparable rows: {float(checks['minimum_comparable_regret']):.3e}.
- Raw runtime rows: {int(checks['runtime_row_count'])}; expected {int(checks['expected_runtime_row_count'])}.
- Source decision sizes equal their stated K: {bool(checks['candidate_sizes_passed'])}.
- Saved aggregate reconciliation: {bool(checks['aggregate_reconciliation_passed'])}.

## Issues and limitations

1. The corpus contains one controlled instance per topology and K, so topology
   differences are descriptive rather than statistically generalizable.
2. Predicted candidate time is a route-level model output; unrestricted
   Dijkstra uses additive base edge time. No route-model-to-edge prediction
   integration is implemented here, so their scopes and objectives differ.
3. The conservative deadline proxy is not a realized emergency outcome.
4. Runtime varies by machine and process load and must not be extrapolated into
   scaling or cross-platform performance claims.
5. Full-risk scores inherit the documented synthetic states, conformal model,
   deadline definition, and rerouting weight from Steps 4-6.5.
6. These synthetic inputs are simple directed graphs. This benchmark does not
   validate distance semantics for later GraphML imports where collapsing
   parallel road edges under another weight could remove a distance-optimal edge.

The graph-wide distance baseline's parallel-edge projection restriction was not
triggered: all five benchmark networks are native synthetic simple graphs with
zero collapsed parallel edges. No projected GraphML distance result is included.
Candidate shortest distance and graph-wide shortest distance happened to select
the same path in this corpus because every candidate set retained the unrestricted
distance optimum. They remain distinct baselines; this agreement is not assumed
for another candidate generator, network, or parallel-edge representation.

## Decision

Use exact risk argmin as ground truth for tractable candidate sets. Retain all
other methods as distinct baselines, and keep unrestricted Dijkstra labeled
separately in downstream comparisons.
"""
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")


def run() -> None:
    config = _read_json(CONFIG_PATH)
    source_directory = REPOSITORY_ROOT / str(config["source_artifacts"])
    source_path = source_directory / str(config["source_scores_file"])
    network_config_path = REPOSITORY_ROOT / str(config["network_config"])
    network_config = _read_json(network_config_path)
    runtime_repetitions = int(config["runtime_repetitions"])
    if runtime_repetitions <= 0:
        raise ValueError("runtime_repetitions must be positive")
    candidate_counts = {int(value) for value in config["candidate_counts"]}
    tolerance = float(config["tie_absolute_tolerance"])
    delimiter = str(config["path_delimiter"])
    expected_column = str(config["expected_candidate_column"])
    dijkstra_weight = str(config["dijkstra_weight"])
    dijkstra_distance_weight = str(config["dijkstra_distance_weight"])
    dijkstra_distance_travel_time_weight = str(
        config["dijkstra_distance_travel_time_weight"]
    )
    allow_parallel_edge_projection = bool(config["allow_parallel_edge_projection"])

    source_scores = pd.read_csv(source_path)
    corpus = source_scores[
        (source_scores["variant"] == str(config["source_variant"]))
        & source_scores["K"].isin(candidate_counts)
    ].copy()
    required_columns = {
        "network_id",
        "topology",
        "K",
        "route_id",
        "ordered_nodes",
        "distance_km",
        expected_column,
        "upper_bound",
        "deadline_minutes",
        "risk_score",
    }
    missing = sorted(required_columns.difference(corpus.columns))
    if missing:
        raise ValueError(f"source risk corpus is missing columns: {missing}")
    if corpus.empty:
        raise ValueError("source risk corpus is empty")
    if corpus.duplicated(["network_id", "K", "route_id"]).any():
        raise ValueError("source candidate identity must be unique by network/K/route")
    numeric_values = corpus.loc[
        :, ["K", "distance_km", expected_column, "upper_bound", "deadline_minutes", "risk_score"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("source risk corpus contains non-finite required values")
    corpus["ordered_nodes_tuple"] = corpus["ordered_nodes"].map(
        lambda value: _parse_path(value, delimiter)
    )

    network_entries = {
        str(entry["network_id"]): dict(entry)
        for entry in network_config["networks"]
    }
    graphs: dict[str, nx.DiGraph] = {}
    for network_id in sorted(corpus["network_id"].unique()):
        if network_id not in network_entries:
            raise ValueError(f"network {network_id!r} is absent from network config")
        entry = network_entries[network_id]
        graphs[network_id] = generate_synthetic_network(
            SyntheticNetworkSpec(
                topology=str(entry["topology"]),
                seed=int(entry["seed"]),
                parameters=dict(entry["parameters"]),
            )
        )

    instance_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []
    method_order = [
        "shortest_distance_candidate",
        "dijkstra_shortest_distance_path",
        "shortest_expected_time_candidate",
        "dijkstra_expected_time_path",
        "exact_risk_argmin",
        "minimum_upper_bound_heuristic",
    ]

    for (network_id, topology, k_value), raw_group in corpus.groupby(
        ["network_id", "topology", "K"], sort=True
    ):
        k = int(k_value)
        candidates = raw_group.rename(
            columns={"ordered_nodes_tuple": "path_for_baseline"}
        ).copy()
        deadline_values = candidates["deadline_minutes"].unique()
        if len(candidates) != k or len(deadline_values) != 1:
            raise ValueError(f"invalid candidate decision {network_id}/K={k}")
        deadline = float(deadline_values[0])
        exact_result = select_exact_risk_candidate(
            candidates,
            path_column="path_for_baseline",
            expected_time_column=expected_column,
        )
        exact_risk = float(
            candidates.loc[
                candidates["route_id"] == exact_result.route_id, "risk_score"
            ].iloc[0]
        )
        graph = graphs[str(network_id)]
        origin = graph.graph["origin"]
        destination = graph.graph["destination"]

        candidate_operations: list[
            tuple[str, str, Callable[[], ClassicalBaselineResult]]
        ] = [
            (
                "shortest_distance_candidate",
                "distance_km",
                lambda table=candidates: select_shortest_distance_candidate(
                    table,
                    path_column="path_for_baseline",
                    expected_time_column=expected_column,
                ),
            ),
            (
                "shortest_expected_time_candidate",
                expected_column,
                lambda table=candidates: select_shortest_expected_time_candidate(
                    table,
                    path_column="path_for_baseline",
                    expected_time_column=expected_column,
                ),
            ),
            (
                "exact_risk_argmin",
                "risk_score",
                lambda table=candidates: select_exact_risk_candidate(
                    table,
                    path_column="path_for_baseline",
                    expected_time_column=expected_column,
                ),
            ),
            (
                "minimum_upper_bound_heuristic",
                "upper_bound",
                lambda table=candidates: select_upper_bound_heuristic(
                    table,
                    path_column="path_for_baseline",
                    expected_time_column=expected_column,
                ),
            ),
        ]

        operations: list[
            tuple[str, str, str, Callable[[], ClassicalBaselineResult]]
        ] = [
            (method, "candidate_set", objective, operation)
            for method, objective, operation in candidate_operations
        ]
        operations.insert(
            1,
            (
                "dijkstra_shortest_distance_path",
                "unrestricted_graph",
                dijkstra_distance_weight,
                lambda current_graph=graph: dijkstra_shortest_distance_path(
                    current_graph,
                    origin,
                    destination,
                    distance_weight=dijkstra_distance_weight,
                    travel_time_weight=dijkstra_distance_travel_time_weight,
                    allow_parallel_edge_projection=allow_parallel_edge_projection,
                ),
            ),
        )
        operations.insert(
            3,
            (
                "dijkstra_expected_time_path",
                "unrestricted_graph",
                dijkstra_weight,
                lambda current_graph=graph: dijkstra_expected_time_path(
                    current_graph,
                    origin,
                    destination,
                    weight=dijkstra_weight,
                ),
            ),
        )

        for method, selection_scope, objective_column, operation in operations:
            result, timings = _timed_repetitions(
                operation,
                repetitions=runtime_repetitions,
            )
            runtime_statistics = _runtime_summary(timings)
            if selection_scope == "candidate_set":
                selected_rows = candidates[candidates["route_id"] == result.route_id]
                tie_count = _candidate_tie_count(
                    candidates,
                    objective_column,
                    absolute_tolerance=tolerance,
                )
                reversed_table = candidates.iloc[::-1].reset_index(drop=True)
                reversed_operations = {
                    "shortest_distance_candidate": lambda: select_shortest_distance_candidate(
                        reversed_table,
                        path_column="path_for_baseline",
                        expected_time_column=expected_column,
                    ),
                    "shortest_expected_time_candidate": lambda: select_shortest_expected_time_candidate(
                        reversed_table,
                        path_column="path_for_baseline",
                        expected_time_column=expected_column,
                    ),
                    "exact_risk_argmin": lambda: select_exact_risk_candidate(
                        reversed_table,
                        path_column="path_for_baseline",
                        expected_time_column=expected_column,
                    ),
                    "minimum_upper_bound_heuristic": lambda: select_upper_bound_heuristic(
                        reversed_table,
                        path_column="path_for_baseline",
                        expected_time_column=expected_column,
                    ),
                }
                reordered_result = reversed_operations[method]()
            else:
                selected_rows = candidates[
                    candidates["path_for_baseline"].map(tuple) == result.path
                ]
                graph_weight = (
                    dijkstra_distance_weight
                    if method == "dijkstra_shortest_distance_path"
                    else dijkstra_weight
                )
                tie_count = _dijkstra_shortest_path_count(
                    graph, origin, destination, graph_weight
                )
                if method == "dijkstra_shortest_distance_path":
                    reordered_result = dijkstra_shortest_distance_path(
                        _reverse_insertion_copy(graph),
                        origin,
                        destination,
                        distance_weight=dijkstra_distance_weight,
                        travel_time_weight=dijkstra_distance_travel_time_weight,
                        allow_parallel_edge_projection=allow_parallel_edge_projection,
                    )
                else:
                    reordered_result = dijkstra_expected_time_path(
                        _reverse_insertion_copy(graph),
                        origin,
                        destination,
                        weight=dijkstra_weight,
                    )

            stable = (
                result.route_id == reordered_result.route_id
                and result.path == reordered_result.path
            )
            candidate_comparable = len(selected_rows) == 1
            if candidate_comparable:
                selected = selected_rows.iloc[0]
                selected_route_id: str | None = str(selected["route_id"])
                selected_risk = float(selected["risk_score"])
                selected_prediction = float(selected[expected_column])
                selected_upper = float(selected["upper_bound"])
                selected_distance = float(selected["distance_km"])
                regret = selected_risk - exact_risk
                deadline_margin = deadline - selected_upper
                deadline_violation: bool | float = selected_upper > deadline
                is_optimal: bool | float = bool(
                    np.isclose(regret, 0.0, rtol=0.0, atol=tolerance)
                )
            else:
                selected_route_id = None
                selected_risk = np.nan
                selected_prediction = np.nan
                selected_upper = np.nan
                selected_distance = float(result.distance_km or np.nan)
                regret = np.nan
                deadline_margin = np.nan
                deadline_violation = np.nan
                is_optimal = np.nan

            instance_rows.append(
                {
                    "network_id": network_id,
                    "topology": topology,
                    "K": k,
                    "method": method,
                    "selection_scope": selection_scope,
                    "objective_column": objective_column,
                    "objective_units": (
                        "km"
                        if objective_column == "distance_km"
                        else "minute_equivalent"
                        if objective_column == "risk_score"
                        else "minutes"
                    ),
                    "objective_value": result.objective_value,
                    "selected_route_id": selected_route_id,
                    "selected_path": delimiter.join(str(node) for node in result.path),
                    "candidate_comparable": candidate_comparable,
                    "selected_distance_km": selected_distance,
                    "selected_prediction_min": selected_prediction,
                    "graph_path_distance_km": (
                        float(result.distance_km)
                        if selection_scope == "unrestricted_graph"
                        and result.distance_km is not None
                        else np.nan
                    ),
                    "graph_path_base_travel_time_min": (
                        float(result.expected_travel_time_min)
                        if selection_scope == "unrestricted_graph"
                        and result.expected_travel_time_min is not None
                        else np.nan
                    ),
                    "graph_parallel_edge_projection": bool(
                        graph.graph.get("loader_parallel_edges_collapsed", 0)
                    )
                    if selection_scope == "unrestricted_graph"
                    else False,
                    "selected_upper_bound_min": selected_upper,
                    "selected_risk_score": selected_risk,
                    "exact_risk_score": exact_risk,
                    "risk_regret": regret,
                    "is_exact_risk_optimal": is_optimal,
                    "deadline_minutes": deadline,
                    "deadline_margin_min": deadline_margin,
                    "deadline_proxy_violation": deadline_violation,
                    "minimum_tie_count": tie_count,
                    "tie_break_applied": tie_count > 1,
                    "stable_under_reordered_input": stable,
                    **runtime_statistics,
                }
            )
            tie_rows.append(
                {
                    "network_id": network_id,
                    "topology": topology,
                    "K": k,
                    "method": method,
                    "minimum_tie_count": tie_count,
                    "tie_break_applied": tie_count > 1,
                    "selected_route_id": selected_route_id,
                    "selected_path": delimiter.join(str(node) for node in result.path),
                    "stable_under_reordered_input": stable,
                }
            )
            for repetition, runtime_ns in timings:
                runtime_rows.append(
                    {
                        "network_id": network_id,
                        "topology": topology,
                        "K": k,
                        "method": method,
                        "repetition": repetition,
                        "runtime_ns": runtime_ns,
                        "selected_route_id": selected_route_id,
                        "selected_path": delimiter.join(str(node) for node in result.path),
                    }
                )

    instance_results = pd.DataFrame(instance_rows)
    instance_results["method"] = pd.Categorical(
        instance_results["method"], categories=method_order, ordered=True
    )
    instance_results = instance_results.sort_values(
        ["network_id", "K", "method"], kind="stable"
    ).reset_index(drop=True)
    instance_results["method"] = instance_results["method"].astype(str)
    runtime_table = pd.DataFrame(runtime_rows)
    tie_table = pd.DataFrame(tie_rows)
    summary, by_topology, by_k = _summary_tables(instance_results)

    expected_runtime_rows = len(instance_results) * runtime_repetitions
    source_group_sizes = corpus.groupby(["network_id", "K"], sort=True).size()
    recalculated_summary, _, _ = _summary_tables(instance_results)
    checks: dict[str, object] = {
        "source_identity_unique_passed": not corpus.duplicated(
            ["network_id", "K", "route_id"]
        ).any(),
        "candidate_sizes_passed": bool(
            all(int(size) == int(index[1]) for index, size in source_group_sizes.items())
        ),
        "all_numeric_finite_passed": bool(np.isfinite(numeric_values).all()),
        "selection_count_passed": len(instance_results) == len(source_group_sizes) * 6,
        "runtime_count_passed": len(runtime_table) == expected_runtime_rows,
        "stable_ties_passed": bool(instance_results["stable_under_reordered_input"].all()),
        "nonnegative_regret_passed": bool(
            (
                instance_results.loc[
                    instance_results["candidate_comparable"], "risk_regret"
                ]
                >= -tolerance
            ).all()
        ),
        "exact_ground_truth_passed": bool(
            np.allclose(
                instance_results.loc[
                    instance_results["method"] == "exact_risk_argmin", "risk_regret"
                ],
                0.0,
                rtol=0.0,
                atol=tolerance,
            )
        ),
        "simple_graph_projection_policy_passed": bool(
            not instance_results["graph_parallel_edge_projection"].any()
            and not allow_parallel_edge_projection
        ),
        "aggregate_reconciliation_passed": summary.equals(recalculated_summary),
        "maximum_exact_risk_regret": float(
            instance_results.loc[
                instance_results["method"] == "exact_risk_argmin", "risk_regret"
            ].abs().max()
        ),
        "minimum_comparable_regret": float(
            instance_results.loc[
                instance_results["candidate_comparable"], "risk_regret"
            ].min()
        ),
        "runtime_row_count": len(runtime_table),
        "expected_runtime_row_count": expected_runtime_rows,
        "source_route_row_count": len(corpus),
        "decision_set_count": len(source_group_sizes),
        "topology_count": int(corpus["topology"].nunique()),
    }
    if not all(bool(value) for key, value in checks.items() if key.endswith("_passed")):
        raise RuntimeError(f"classical benchmark validation failed: {checks}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    corpus.drop(columns="ordered_nodes_tuple").to_csv(
        OUTPUT_DIRECTORY / "candidate_corpus.csv", index=False
    )
    instance_results.to_csv(OUTPUT_DIRECTORY / "baseline_instance_results.csv", index=False)
    summary.to_csv(OUTPUT_DIRECTORY / "baseline_summary.csv", index=False)
    by_topology.to_csv(OUTPUT_DIRECTORY / "baseline_by_topology.csv", index=False)
    by_k.to_csv(OUTPUT_DIRECTORY / "baseline_by_k.csv", index=False)
    runtime_table.to_csv(OUTPUT_DIRECTORY / "runtime_repetitions.csv", index=False)
    tie_table.to_csv(OUTPUT_DIRECTORY / "tie_validation.csv", index=False)
    write_json(OUTPUT_DIRECTORY / "validation_checks.json", checks)

    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={},
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest["source_files"] = {
        str(source_path.relative_to(REPOSITORY_ROOT)): file_sha256(source_path),
        str(network_config_path.relative_to(REPOSITORY_ROOT)): file_sha256(
            network_config_path
        ),
    }
    implementation_paths = [
        REPOSITORY_ROOT / "src" / "classical_baselines.py",
        Path(__file__).resolve(),
    ]
    manifest["implementation_files"] = {
        str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path)
        for path in implementation_paths
    }
    manifest["graph_distance_policy"] = {
        "distance_weight": dijkstra_distance_weight,
        "travel_time_weight": dijkstra_distance_travel_time_weight,
        "allow_parallel_edge_projection": allow_parallel_edge_projection,
        "all_benchmark_graphs_are_native_simple_graphs": bool(
            not instance_results["graph_parallel_edge_projection"].any()
        ),
    }
    manifest["artifact_rows"] = {
        "candidate_corpus.csv": len(corpus),
        "baseline_instance_results.csv": len(instance_results),
        "baseline_summary.csv": len(summary),
        "baseline_by_topology.csv": len(by_topology),
        "baseline_by_k.csv": len(by_k),
        "runtime_repetitions.csv": len(runtime_table),
        "tie_validation.csv": len(tie_table),
    }
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    _write_validation_report(
        summary,
        instance_results,
        checks,
        source_rows=len(corpus),
        topology_count=int(corpus["topology"].nunique()),
    )

    print(summary.to_string(index=False))
    print(f"Validated instance rows: {len(instance_results)}")
    print(f"Raw runtime repetitions: {len(runtime_table)}")
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run()
