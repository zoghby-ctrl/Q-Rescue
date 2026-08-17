"""Thin orchestration layer connecting the Streamlit MVP to Q-Rescue.

Scientific calculations stay in :mod:`src`.  This module only loads validated
artifacts, assembles supported scenarios, and packages backend outputs for the
presentation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Hashable, Mapping, Sequence

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from src.candidate_routes import CandidateRoute, generate_candidate_routes
from src.classical_baselines import (
    select_exact_risk_candidate,
    select_shortest_expected_time_candidate,
)
from src.conformal import LocallyScaledSplitConformalRegressor
from src.deadline_risk import RiskWeights
from src.dynamic_rerouting import (
    DynamicEvent,
    DynamicSimulationResult,
    apply_dynamic_event,
    prepare_dynamic_candidates,
    simulate_dynamic_route,
)
from src.feasibility_qaoa import (
    circuit_resource_record,
    create_expectation_qnode,
    create_probability_qnode,
    one_hot_basis_indices,
    optimize_qaoa_angles,
    summarize_probabilities,
)
from src.network_loader import load_network_graphml
from src.network_schema import validate_network
from src.risk_qubo import (
    RiskQUBO,
    brute_force_risk_qubo,
    build_risk_qubo,
    risk_qubo_to_ising,
)
from src.synthetic_networks import SyntheticNetworkSpec, generate_synthetic_network
from src.traffic_scenarios import get_scenario


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS: Mapping[str, str] = {
    "expected_time": "Expected-time baseline",
    "exact_risk": "Exact risk-aware classical",
    "standard_x": "Standard QAOA",
    "feasible_xy": "Feasible-XY QAOA",
}

QAOA_METHOD_LABELS: Mapping[str, str] = {
    "standard_x": "Standard X-Mixer QAOA",
    "feasible_xy": "Feasible XY QAOA",
}

SCENARIO_LABELS: Mapping[str, str] = {
    "normal": "Normal",
    "rush": "Rush hour",
    "rain": "Rain",
    "accident": "Accident",
    "closure": "Road closure",
    "prediction_noise": "Prediction noise",
}

SUPPORTED_K = (4, 6, 8)
SUPPORTED_DEPTHS = (1, 2, 3)
DEFAULT_RISK_WEIGHTS = RiskWeights(
    lambda_uncertainty=1.0,
    lambda_deadline=2.0,
    lambda_reroute=0.0,
)


class PipelineConfigurationError(ValueError):
    """Raised when a selected UI scenario is outside an executable scope."""


@dataclass(frozen=True)
class NetworkDefinition:
    key: str
    label: str
    network_id: str
    topology: str
    origin: str
    destination: str
    is_real: bool
    region_label: str | None = None
    study_label: str | None = None


@dataclass(frozen=True)
class ModelBundle:
    point_model: object
    interval_model: LocallyScaledSplitConformalRegressor
    distance_normalization_km: float
    model_name: str
    calibration_method: str
    alpha: float
    calibration_quantile: float
    n_calibration: int
    artifact_scope: str


@dataclass(frozen=True)
class QuantumResult:
    method: str
    method_label: str
    number_of_qubits: int
    depth: int
    iterations: int
    valid_probability: float
    optimal_probability: float
    most_probable_valid_bitstring: str
    selected_route_index: int
    selected_route_id: str
    expected_qubo_energy: float
    circuit_depth: int
    gate_count: int
    probability_table: pd.DataFrame
    training_history: pd.DataFrame


@dataclass(frozen=True)
class PipelineResult:
    network_id: str
    topology: str
    is_real_topology: bool
    origin: Hashable
    destination: Hashable
    scenario_name: str
    scenario_label: str
    deadline_minutes: float
    k: int
    qaoa_depth: int
    selected_method: str
    selected_method_label: str
    selected_route_id: str
    fastest_route_id: str
    lowest_risk_route_id: str
    graph: nx.DiGraph
    routes: tuple[CandidateRoute, ...]
    route_table: pd.DataFrame
    comparison_table: pd.DataFrame
    qubo: RiskQUBO
    exact_assignments: pd.DataFrame
    quantum_results: Mapping[str, QuantumResult]
    model_name: str
    model_features: tuple[str, ...]
    calibration_method: str
    calibration_alpha: float
    calibration_quantile: float
    calibration_size: int
    model_artifact_scope: str
    risk_weights: RiskWeights
    disruption_edge: tuple[Hashable, Hashable] | None
    runtime_seconds: float

    @property
    def selected_row(self) -> pd.Series:
        return self.route_table.loc[
            self.route_table["route_id"] == self.selected_route_id
        ].iloc[0]

    @property
    def is_quantum(self) -> bool:
        return self.selected_method in QAOA_METHOD_LABELS


@dataclass(frozen=True)
class DynamicDemoResult:
    event_name: str
    event_label: str
    route_before_event: str | None
    updated_route: str | None
    rerouting_occurred: bool
    reroute_count: int
    reroute_cost_minutes: float
    updated_deadline_margin: float | None
    disruption_edge: tuple[Hashable, Hashable] | None
    simulation: DynamicSimulationResult


def _read_json(relative_path: str) -> dict[str, object]:
    return json.loads((REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"))


def _single_distance_scale(relative_path: str) -> float:
    values = pd.read_csv(
        REPOSITORY_ROOT / relative_path,
        usecols=["distance_normalization_km"],
    )["distance_normalization_km"].drop_duplicates()
    if len(values) != 1:
        raise RuntimeError(f"{relative_path} must contain one frozen distance scale")
    return float(values.iloc[0])


def load_model_bundles() -> dict[str, ModelBundle]:
    """Load the validated synthetic and frozen real-transfer inference bundles."""

    synthetic_root = REPOSITORY_ROOT / "results/research/steps8_25_8_75_topology_benchmark"
    synthetic_point = joblib.load(synthetic_root / "best_model.joblib")
    synthetic_scale = joblib.load(synthetic_root / "residual_scale_model.joblib")
    synthetic_calibration = _read_json(
        "results/research/steps8_25_8_75_topology_benchmark/conformal_calibration.json"
    )
    synthetic_interval = LocallyScaledSplitConformalRegressor(
        base_model=synthetic_point,
        scale_model=synthetic_scale,
        alpha=float(synthetic_calibration["alpha"]),
        calibration_quantile=float(synthetic_calibration["calibration_quantile"]),
        quantile_level=float(synthetic_calibration["finite_sample_quantile_level"]),
        n_calibration=int(synthetic_calibration["n_calibration"]),
    )

    real_config = _read_json("configs/real_topology_benchmark.json")
    real_point = joblib.load(REPOSITORY_ROOT / str(real_config["frozen_step4_model"]))
    real_scale = joblib.load(REPOSITORY_ROOT / str(real_config["frozen_step5_scale_model"]))
    quantiles = pd.read_csv(REPOSITORY_ROOT / str(real_config["frozen_step5_quantiles"]))
    selected = quantiles.loc[
        (quantiles["method"] == str(real_config["conformal_method"]))
        & np.isclose(quantiles["alpha"], float(real_config["conformal_alpha"]))
    ]
    if len(selected) != 1:
        raise RuntimeError("frozen real-topology conformal quantile is ambiguous")
    real_row = selected.iloc[0]
    real_interval = LocallyScaledSplitConformalRegressor(
        base_model=real_point,
        scale_model=real_scale,
        alpha=float(real_row["alpha"]),
        calibration_quantile=float(real_row["calibration_residual_quantile"]),
        quantile_level=float(real_row["finite_sample_quantile_level"]),
        n_calibration=int(real_row["n_calibration"]),
    )

    return {
        "synthetic": ModelBundle(
            point_model=synthetic_point,
            interval_model=synthetic_interval,
            distance_normalization_km=_single_distance_scale(
                "results/research/steps8_25_8_75_topology_benchmark/traffic_observations.csv"
            ),
            model_name=str(synthetic_point.name),
            calibration_method="Locally scaled split conformal",
            alpha=synthetic_interval.alpha,
            calibration_quantile=synthetic_interval.calibration_quantile,
            n_calibration=synthetic_interval.n_calibration,
            artifact_scope="Repeated synthetic topology benchmark",
        ),
        "real": ModelBundle(
            point_model=real_point,
            interval_model=real_interval,
            distance_normalization_km=_single_distance_scale(
                str(real_config["frozen_step4_train"])
            ),
            model_name=str(real_point.name),
            calibration_method="Locally scaled split conformal (frozen Step 5)",
            alpha=real_interval.alpha,
            calibration_quantile=real_interval.calibration_quantile,
            n_calibration=real_interval.n_calibration,
            artifact_scope="Frozen Step 4/5 real-topology transfer",
        ),
    }


def load_network_catalog() -> dict[str, NetworkDefinition]:
    """Return presentation labels for five synthetic and four cached real cases."""

    synthetic_config = _read_json("configs/topology_repeated_benchmark.json")
    topology_labels = {
        "grid": "Synthetic grid",
        "radial": "Synthetic radial",
        "irregular_geometric": "Synthetic irregular",
        "sparse_mixed": "Synthetic sparse / mixed",
        "dense_mixed": "Synthetic dense / mixed",
    }
    catalog: dict[str, NetworkDefinition] = {}
    for entry_object in synthetic_config["instances"]:
        entry = dict(entry_object)
        if int(entry["replicate"]) != 0:
            continue
        topology = str(entry["topology"])
        graph = _build_synthetic_graph(entry)
        key = f"synthetic:{topology}"
        catalog[key] = NetworkDefinition(
            key=key,
            label=topology_labels[topology],
            network_id=str(entry["network_id"]),
            topology=topology,
            origin=str(graph.graph["origin"]),
            destination=str(graph.graph["destination"]),
            is_real=False,
        )

    registry = pd.read_csv(REPOSITORY_ROOT / "data/real_networks/cache_registry.csv")
    for row in registry.sort_values("network_id").itertuples():
        key = f"real:{row.network_id}"
        catalog[key] = NetworkDefinition(
            key=key,
            label=f"{row.region_label} · {str(row.topology_class).replace('_', ' ')}",
            network_id=str(row.network_id),
            topology=str(row.topology_class),
            origin=str(row.origin),
            destination=str(row.destination),
            is_real=True,
            region_label=str(row.region_label),
            study_label="Real road topology + simulated traffic",
        )
    return catalog


def _build_synthetic_graph(entry: Mapping[str, object]) -> nx.DiGraph:
    graph = generate_synthetic_network(
        SyntheticNetworkSpec(
            topology=str(entry["topology"]),
            seed=int(entry["seed"]),
            parameters=dict(entry["parameters"]),
        )
    )
    multiplier = float(entry["base_time_multiplier"])
    for _, _, attributes in graph.edges(data=True):
        attributes["base_travel_time_min"] = (
            float(attributes["base_travel_time_min"]) * multiplier
        )
        if attributes.get("speed_kph") is not None:
            attributes["speed_kph"] = float(attributes["speed_kph"]) / multiplier
    graph.graph.update(
        network_id=str(entry["network_id"]),
        topology_family=str(entry["topology"]),
        replicate=int(entry["replicate"]),
        base_time_multiplier=multiplier,
        ui_network_key=f"synthetic:{entry['topology']}",
    )
    return validate_network(graph)


def load_network(network_key: str) -> nx.DiGraph:
    """Load one canonical graph without changing its scientific attributes."""

    if network_key.startswith("synthetic:"):
        topology = network_key.split(":", 1)[1]
        config = _read_json("configs/topology_repeated_benchmark.json")
        selected = [
            dict(entry)
            for entry in config["instances"]
            if str(entry["topology"]) == topology and int(entry["replicate"]) == 0
        ]
        if len(selected) != 1:
            raise KeyError(f"unknown synthetic network {network_key!r}")
        return _build_synthetic_graph(selected[0])

    if network_key.startswith("real:"):
        network_id = network_key.split(":", 1)[1]
        path = REPOSITORY_ROOT / "data/real_networks" / f"{network_id}.graphml"
        if not path.exists():
            raise KeyError(f"cached real network {network_id!r} is unavailable")
        graph = load_network_graphml(path)
        graph.graph.update(ui_network_key=network_key, real_topology=True)
        return validate_network(graph)

    raise KeyError(f"unknown network key {network_key!r}")


def _central_first(edges: Sequence[tuple[Hashable, Hashable]]) -> list[tuple[Hashable, Hashable]]:
    if not edges:
        return []
    middle = (len(edges) - 1) / 2.0
    return sorted(edges, key=lambda item: abs(edges.index(item) - middle))


def _base_route_pool(
    graph: nx.DiGraph,
    origin: Hashable,
    destination: Hashable,
    k: int,
) -> list[tuple[Hashable, Hashable]]:
    routes = generate_candidate_routes(
        graph,
        origin,
        destination,
        k,
        require_k=False,
    )
    if not routes:
        raise PipelineConfigurationError("No route exists for the selected origin and destination.")
    ordered: list[tuple[Hashable, Hashable]] = []
    for route in routes:
        for edge in _central_first(list(route.edges)):
            if edge not in ordered:
                ordered.append(edge)
    return ordered


def build_scenario_event(
    graph: nx.DiGraph,
    origin: Hashable,
    destination: Hashable,
    scenario_name: str,
    *,
    k: int,
) -> tuple[DynamicEvent, tuple[Hashable, Hashable] | None]:
    """Translate a supported traffic preset into the existing dynamic event API."""

    if scenario_name not in SCENARIO_LABELS:
        raise PipelineConfigurationError(f"Unsupported traffic scenario {scenario_name!r}.")
    scenario = get_scenario(scenario_name)
    edge_pool = _base_route_pool(graph, origin, destination, max(1, k))
    disrupted_edge: tuple[Hashable, Hashable] | None = None
    rain = int(scenario_name == "rain")
    incident_edges: frozenset[tuple[Hashable, Hashable]] = frozenset()
    localized_edges: frozenset[tuple[Hashable, Hashable]] = frozenset()
    closed_edges: frozenset[tuple[Hashable, Hashable]] = frozenset()

    if scenario_name == "accident":
        disrupted_edge = edge_pool[0]
        incident_edges = frozenset({disrupted_edge})
        localized_edges = frozenset({disrupted_edge})

    if scenario_name == "closure":
        for candidate_edge in edge_pool:
            candidate_event = DynamicEvent(
                name=scenario_name,
                global_congestion=scenario.congestion_mean,
                closed_edges=frozenset({candidate_edge}),
                actual_noise_fraction=scenario.prediction_noise_fraction,
            )
            candidate_graph = apply_dynamic_event(graph, candidate_event)
            remaining = generate_candidate_routes(
                candidate_graph,
                origin,
                destination,
                k,
                weight="dynamic_expected_time_min",
                expected_time_attribute="dynamic_expected_time_min",
                require_k=False,
            )
            if len(remaining) == k:
                disrupted_edge = candidate_edge
                closed_edges = frozenset({candidate_edge})
                break
        if disrupted_edge is None:
            raise PipelineConfigurationError(
                f"Road closure leaves fewer than K={k} routes for this O/D pair. "
                "Choose a smaller K or another origin/destination."
            )

    event = DynamicEvent(
        name=scenario_name,
        global_congestion=scenario.congestion_mean,
        rain_indicator=rain,
        localized_edges=localized_edges,
        localized_congestion_boost=0.25 if localized_edges else 0.0,
        incident_edges=incident_edges,
        closed_edges=closed_edges,
        actual_noise_fraction=scenario.prediction_noise_fraction,
    )
    return event, disrupted_edge


def available_k_values(
    graph: nx.DiGraph,
    origin: Hashable,
    destination: Hashable,
) -> tuple[int, ...]:
    """Return validated UI K values supported by the current graph and O/D pair."""

    routes = generate_candidate_routes(
        graph,
        origin,
        destination,
        max(SUPPORTED_K),
        require_k=False,
    )
    return tuple(k for k in SUPPORTED_K if len(routes) >= k)


def _solve_quantum(
    qubo: RiskQUBO,
    route_ids: Sequence[str],
    *,
    method: str,
    depth: int,
    iterations: int,
    seed: int,
) -> QuantumResult:
    hamiltonian, _, _, _ = risk_qubo_to_ising(qubo)
    expectation_qnode = create_expectation_qnode(
        hamiltonian,
        number_of_qubits=qubo.number_of_variables,
        depth=depth,
        method=method,
    )
    probability_qnode = create_probability_qnode(
        hamiltonian,
        number_of_qubits=qubo.number_of_variables,
        depth=depth,
        method=method,
    )
    optimized = optimize_qaoa_angles(
        expectation_qnode,
        depth=depth,
        seed=seed,
        iterations=iterations,
        step_size=0.05,
    )
    probabilities = np.asarray(
        probability_qnode(optimized.gammas, optimized.betas), dtype=float
    )
    optimal_route_index = int(np.argmin(np.asarray(qubo.risk_scores)))
    summary = summarize_probabilities(
        probabilities,
        qubo,
        optimal_route_index=optimal_route_index,
    )
    resources = circuit_resource_record(
        probability_qnode, optimized.gammas, optimized.betas
    )
    valid_basis = one_hot_basis_indices(qubo.number_of_variables)
    selected_index = int(summary["most_probable_valid_route_index"])
    probability_table = pd.DataFrame(
        {
            "route_id": list(route_ids),
            "bitstring": [
                format(index, f"0{qubo.number_of_variables}b") for index in valid_basis
            ],
            "probability": [float(probabilities[index]) for index in valid_basis],
            "selected": [index == selected_index for index in range(len(route_ids))],
        }
    )
    return QuantumResult(
        method=method,
        method_label=QAOA_METHOD_LABELS[method],
        number_of_qubits=qubo.number_of_variables,
        depth=depth,
        iterations=iterations,
        valid_probability=float(np.clip(summary["valid_probability"], 0.0, 1.0)),
        optimal_probability=float(np.clip(summary["optimal_probability"], 0.0, 1.0)),
        most_probable_valid_bitstring=str(summary["most_probable_valid_bitstring"]),
        selected_route_index=selected_index,
        selected_route_id=str(route_ids[selected_index]),
        expected_qubo_energy=float(summary["expected_qubo_energy"]),
        circuit_depth=int(resources["circuit_depth"]),
        gate_count=int(resources["gate_count"]),
        probability_table=probability_table,
        training_history=optimized.history.copy(),
    )


def run_pipeline(
    graph: nx.DiGraph,
    model_bundle: ModelBundle,
    *,
    origin: Hashable,
    destination: Hashable,
    scenario_name: str,
    deadline_minutes: float,
    k: int,
    selected_method: str,
    qaoa_depth: int,
    qaoa_iterations: int = 12,
    qaoa_seed: int = 825_600,
) -> PipelineResult:
    """Run the complete existing Q-Rescue decision pipeline for one scenario."""

    started = time.perf_counter()
    validate_network(graph)
    if selected_method not in METHOD_LABELS:
        raise PipelineConfigurationError(f"Unknown routing method {selected_method!r}.")
    if k not in SUPPORTED_K:
        raise PipelineConfigurationError(f"K must be one of {SUPPORTED_K}.")
    if qaoa_depth not in SUPPORTED_DEPTHS:
        raise PipelineConfigurationError(f"QAOA depth must be one of {SUPPORTED_DEPTHS}.")
    if deadline_minutes <= 0.0:
        raise PipelineConfigurationError("Emergency deadline must be positive.")

    event, disruption_edge = build_scenario_event(
        graph, origin, destination, scenario_name, k=k
    )
    dynamic_graph = apply_dynamic_event(graph, event)
    scored, routes = prepare_dynamic_candidates(
        dynamic_graph,
        origin,
        destination,
        k=k,
        event=event,
        point_interval_model=model_bundle.interval_model,
        distance_normalization_km=model_bundle.distance_normalization_km,
        remaining_deadline_minutes=float(deadline_minutes),
        current_plan=None,
        reroute_penalty_minutes=0.0,
        risk_weights=DEFAULT_RISK_WEIGHTS,
    )
    if len(routes) != k:
        raise PipelineConfigurationError(
            f"The selected scenario produced {len(routes)} routes, not K={k}."
        )

    route_lookup = {route.route_id: route for route in routes}
    route_table = scored.copy()
    route_table["ordered_nodes"] = [
        route_lookup[str(route_id)].nodes for route_id in route_table["route_id"]
    ]
    route_table["edges"] = [
        route_lookup[str(route_id)].edges for route_id in route_table["route_id"]
    ]
    route_table["expected_travel_time_min"] = route_table["prediction"]
    route_table["uncertainty_width"] = (
        route_table["upper_bound"] - route_table["lower_bound"]
    )
    route_table["conservative_deadline_violation"] = (
        route_table["upper_bound"] > float(deadline_minutes)
    )

    fastest = select_shortest_expected_time_candidate(
        route_table,
        expected_time_column="prediction",
    )
    lowest_risk = select_exact_risk_candidate(route_table)

    qubo = build_risk_qubo(route_table["risk_score"].to_numpy(dtype=float))
    exact_assignments = brute_force_risk_qubo(qubo)
    exact_valid = exact_assignments.loc[exact_assignments["valid"]]
    exact_assignment = exact_valid.sort_values(
        ["qubo_cost", "selected_route_index"], kind="stable"
    ).iloc[0]
    exact_route_id = str(
        route_table.iloc[int(exact_assignment["selected_route_index"])]["route_id"]
    )
    if exact_route_id != lowest_risk.route_id:
        raise RuntimeError("exact QUBO optimum and exact risk selector disagree")

    route_ids = route_table["route_id"].astype(str).tolist()
    quantum_results = {
        "standard_x": _solve_quantum(
            qubo,
            route_ids,
            method="standard_x",
            depth=qaoa_depth,
            iterations=qaoa_iterations,
            seed=qaoa_seed,
        ),
        "feasible_xy": _solve_quantum(
            qubo,
            route_ids,
            method="feasible_xy",
            depth=qaoa_depth,
            iterations=qaoa_iterations,
            seed=qaoa_seed + 1,
        ),
    }
    selections = {
        "expected_time": str(fastest.route_id),
        "exact_risk": exact_route_id,
        "standard_x": quantum_results["standard_x"].selected_route_id,
        "feasible_xy": quantum_results["feasible_xy"].selected_route_id,
    }
    selected_route_id = selections[selected_method]

    status_values: list[str] = []
    for row in route_table.itertuples():
        labels: list[str] = []
        if row.route_id == fastest.route_id:
            labels.append("Fastest expected")
        if row.route_id == lowest_risk.route_id:
            labels.append("Lowest risk")
        if row.route_id == selected_route_id:
            labels.append("Recommended")
        status_values.append(" · ".join(labels) if labels else "Candidate")
    route_table["selected"] = route_table["route_id"] == selected_route_id
    route_table["status"] = status_values

    comparison_rows: list[dict[str, object]] = []
    for method, route_id in selections.items():
        row = route_table.loc[route_table["route_id"] == route_id].iloc[0]
        quantum = quantum_results.get(method)
        comparison_rows.append(
            {
                "method": METHOD_LABELS[method],
                "route_id": route_id,
                "predicted_time": float(row["prediction"]),
                "deadline_margin": float(row["deadline_margin"]),
                "risk_score": float(row["risk_score"]),
                "valid_probability": (
                    float(quantum.valid_probability) if quantum is not None else np.nan
                ),
                "optimal_probability": (
                    float(quantum.optimal_probability) if quantum is not None else np.nan
                ),
                "is_selected_method": method == selected_method,
            }
        )

    point_features = tuple(getattr(model_bundle.point_model, "feature_columns", ()))
    return PipelineResult(
        network_id=str(graph.graph.get("network_id", "network")),
        topology=str(graph.graph.get("topology_family", graph.graph.get("topology", "unknown"))),
        is_real_topology=not bool(graph.graph.get("synthetic", False)),
        origin=origin,
        destination=destination,
        scenario_name=scenario_name,
        scenario_label=SCENARIO_LABELS[scenario_name],
        deadline_minutes=float(deadline_minutes),
        k=k,
        qaoa_depth=qaoa_depth,
        selected_method=selected_method,
        selected_method_label=METHOD_LABELS[selected_method],
        selected_route_id=selected_route_id,
        fastest_route_id=str(fastest.route_id),
        lowest_risk_route_id=str(lowest_risk.route_id),
        graph=dynamic_graph,
        routes=tuple(routes),
        route_table=route_table.reset_index(drop=True),
        comparison_table=pd.DataFrame(comparison_rows),
        qubo=qubo,
        exact_assignments=exact_assignments,
        quantum_results=quantum_results,
        model_name=model_bundle.model_name,
        model_features=point_features,
        calibration_method=model_bundle.calibration_method,
        calibration_alpha=model_bundle.alpha,
        calibration_quantile=model_bundle.calibration_quantile,
        calibration_size=model_bundle.n_calibration,
        model_artifact_scope=model_bundle.artifact_scope,
        risk_weights=DEFAULT_RISK_WEIGHTS,
        disruption_edge=disruption_edge,
        runtime_seconds=time.perf_counter() - started,
    )


def run_dynamic_demo(
    pipeline: PipelineResult,
    model_bundle: ModelBundle,
    *,
    event_name: str,
    qaoa_iterations: int = 8,
    qaoa_seed: int = 826_100,
    reroute_penalty_minutes: float = 1.5,
) -> DynamicDemoResult:
    """Exercise the existing edge-by-edge rerouting backend for one disruption."""

    if event_name not in {"accident", "congestion", "closure", "rain"}:
        raise PipelineConfigurationError("Unsupported dynamic demonstration event.")
    graph = pipeline.graph.copy()
    # Remove static event mutations by reloading the canonical graph when possible.
    network_key = str(graph.graph.get("ui_network_key", ""))
    if network_key:
        graph = load_network(network_key)

    normal_scenario = get_scenario("normal")
    initial_event = DynamicEvent(
        name="normal",
        global_congestion=normal_scenario.congestion_mean,
        actual_noise_fraction=normal_scenario.prediction_noise_fraction,
    )
    event_scenario_name = "rush" if event_name == "congestion" else event_name
    disruption_event, disruption_edge = build_scenario_event(
        graph,
        pipeline.origin,
        pipeline.destination,
        event_scenario_name,
        k=pipeline.k,
    )
    if event_name == "congestion":
        scenario = get_scenario("congestion")
        disruption_event = DynamicEvent(
            name="congestion",
            global_congestion=scenario.congestion_mean,
            localized_edges=(
                frozenset({pipeline.routes[0].edges[len(pipeline.routes[0].edges) // 2]})
                if pipeline.routes[0].edges
                else frozenset()
            ),
            localized_congestion_boost=0.15,
            actual_noise_fraction=scenario.prediction_noise_fraction,
        )
        disruption_edge = next(iter(disruption_event.localized_edges), None)

    if pipeline.selected_method == "expected_time":
        policy = "expected_time_rerouting"
        quantum_selector = None
    elif pipeline.selected_method == "exact_risk":
        policy = "uncertainty_aware_classical"
        quantum_selector = None
    else:
        policy = "uncertainty_aware_qaoa"
        calls = {"count": 0}

        def quantum_selector(candidates: pd.DataFrame) -> str:
            calls["count"] += 1
            qubo = build_risk_qubo(candidates["risk_score"].to_numpy(dtype=float))
            result = _solve_quantum(
                qubo,
                candidates["route_id"].astype(str).tolist(),
                method=pipeline.selected_method,
                depth=pipeline.qaoa_depth,
                iterations=qaoa_iterations,
                seed=qaoa_seed + calls["count"],
            )
            return result.selected_route_id

    simulation = simulate_dynamic_route(
        graph,
        pipeline.origin,
        pipeline.destination,
        [initial_event, disruption_event],
        policy=policy,
        k=pipeline.k,
        deadline_minutes=pipeline.deadline_minutes,
        point_interval_model=model_bundle.interval_model,
        distance_normalization_km=model_bundle.distance_normalization_km,
        risk_weights=RiskWeights(),
        reroute_penalty_minutes=reroute_penalty_minutes,
        random_seed=2026,
        quantum_selector=quantum_selector,
    )
    decisions = simulation.decisions
    before = None
    updated = None
    if not decisions.empty and "selected_path" in decisions:
        before_value = decisions.iloc[0].get("selected_path")
        before = None if pd.isna(before_value) else str(before_value)
    if len(decisions) > 1 and "selected_path" in decisions:
        updated_value = decisions.iloc[1].get("selected_path")
        updated = None if pd.isna(updated_value) else str(updated_value)

    return DynamicDemoResult(
        event_name=event_name,
        event_label=("Congestion" if event_name == "congestion" else SCENARIO_LABELS.get(event_scenario_name, event_name.title())),
        route_before_event=before,
        updated_route=updated,
        rerouting_occurred=bool(simulation.reroute_count > 0),
        reroute_count=int(simulation.reroute_count),
        reroute_cost_minutes=float(simulation.reroute_count * reroute_penalty_minutes),
        updated_deadline_margin=simulation.deadline_margin_minutes,
        disruption_edge=disruption_edge,
        simulation=simulation,
    )
