"""Run repeated synthetic-topology and OD-characterization benchmarks.

This experiment is deliberately offline and simulator-only.  It evaluates
held-out observations from known synthetic networks; it is not an unseen-
network generalization, real-traffic, quantum-speedup, or hardware study.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.candidate_routes import generate_candidate_routes
from src.conformal import (
    calibrate_locally_scaled_split_conformal,
    evaluate_conformal,
    train_residual_scale_model,
)
from src.deadline_risk import RiskWeights, score_route_table
from src.feasibility_qaoa import (
    circuit_resource_record,
    create_expectation_qnode,
    create_probability_qnode,
    optimize_qaoa_angles,
    summarize_probabilities,
)
from src.network_characterization import characterize_network
from src.network_schema import validate_network
from src.reproducibility import (
    build_experiment_manifest,
    file_sha256,
    write_json,
)
from src.risk_qubo import build_risk_qubo, risk_qubo_to_ising
from src.route_features import (
    RouteTrafficState,
    build_route_feature_table,
    estimate_pretrip_speed_kph,
)
from src.synthetic_networks import SyntheticNetworkSpec, generate_synthetic_network
from src.traffic_data import (
    MODEL_FEATURE_COLUMNS,
    generate_historical_trip_observations,
    split_traffic_observations,
    validate_split_separation,
)
from src.traffic_model import (
    FEATURE_COLUMNS,
    evaluate_traffic_model,
    select_best_by_validation_rmse,
    train_all_traffic_models,
)
from src.traffic_scenarios import get_scenario


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "topology_repeated_benchmark.json"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "research"
    / "steps8_25_8_75_topology_benchmark"
)
EXPECTED_FAMILIES = {
    "grid",
    "radial",
    "irregular_geometric",
    "sparse_mixed",
    "dense_mixed",
}


def _load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _build_scenario_registry(config: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_name in config["traffic_scenarios"]:
        row = asdict(get_scenario(str(scenario_name)))
        row["hours_json"] = json.dumps(row.pop("hours"))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_networks(
    config: dict[str, object],
) -> tuple[dict[str, object], dict[str, str], pd.DataFrame]:
    networks: dict[str, object] = {}
    topology_labels: dict[str, str] = {}
    registry_rows: list[dict[str, object]] = []
    entries = list(config["instances"])
    for entry in entries:
        values = dict(entry)
        network_id = str(values["network_id"])
        topology = str(values["topology"])
        replicate = int(values["replicate"])
        seed = int(values["seed"])
        parameters = dict(values["parameters"])
        multiplier = float(values["base_time_multiplier"])
        if network_id in networks:
            raise ValueError(f"duplicate network_id {network_id!r}")
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("base_time_multiplier must be positive and finite")
        graph = generate_synthetic_network(
            SyntheticNetworkSpec(topology=topology, seed=seed, parameters=parameters)
        )
        for _, _, attributes in graph.edges(data=True):
            attributes["base_travel_time_min"] = (
                float(attributes["base_travel_time_min"]) * multiplier
            )
            if attributes.get("speed_kph") is not None:
                attributes["speed_kph"] = float(attributes["speed_kph"]) / multiplier
        graph.graph.update(
            network_id=network_id,
            topology_family=topology,
            replicate=replicate,
            base_time_multiplier=multiplier,
        )
        validate_network(graph)
        networks[network_id] = graph
        topology_labels[network_id] = topology
        registry_rows.append(
            {
                "network_id": network_id,
                "topology": topology,
                "replicate": replicate,
                "seed": seed,
                "base_time_multiplier": multiplier,
                "generator_parameters_json": json.dumps(parameters, sort_keys=True),
                "node_count": graph.number_of_nodes(),
                "directed_edge_count": graph.number_of_edges(),
                "origin": graph.graph["origin"],
                "destination": graph.graph["destination"],
            }
        )

    registry = pd.DataFrame(registry_rows).sort_values(
        ["topology", "replicate"], kind="stable"
    ).reset_index(drop=True)
    family_counts = registry.groupby("topology").size().to_dict()
    if set(family_counts) != EXPECTED_FAMILIES or any(count < 3 for count in family_counts.values()):
        raise ValueError("benchmark requires at least three instances of every synthetic family")
    if registry["seed"].duplicated().any():
        raise ValueError("every synthetic instance must use a distinct seed")
    for topology, group in registry.groupby("topology"):
        if group["node_count"].nunique() < 2:
            raise ValueError(f"{topology} instances must vary network size")
        if group["base_time_multiplier"].nunique() < 3:
            raise ValueError(f"{topology} instances must vary base-time scaling")
    return networks, topology_labels, registry


def _characterize_instances(
    networks: dict[str, object],
    registry: pd.DataFrame,
    candidate_k: int,
) -> pd.DataFrame:
    registry_lookup = registry.set_index("network_id")
    rows: list[dict[str, object]] = []
    for network_id in sorted(networks):
        graph = networks[network_id]
        result = characterize_network(
            graph, route_k=candidate_k, require_route_k=False
        ).as_record()
        degree_distribution = result.pop("intersection_degree_distribution")
        registry_row = registry_lookup.loc[network_id]
        weak_components = list(nx.weakly_connected_components(graph))
        strong_components = list(nx.strongly_connected_components(graph))
        rows.append(
            {
                "network_id": network_id,
                "topology": registry_row["topology"],
                "replicate": int(registry_row["replicate"]),
                "seed": int(registry_row["seed"]),
                "base_time_multiplier": float(registry_row["base_time_multiplier"]),
                **result,
                "weak_component_count": len(weak_components),
                "strong_component_count": len(strong_components),
                "largest_weak_component_fraction": max(map(len, weak_components))
                / graph.number_of_nodes(),
                "largest_strong_component_fraction": max(map(len, strong_components))
                / graph.number_of_nodes(),
                "intersection_degree_distribution_json": json.dumps(
                    degree_distribution, sort_keys=True
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["topology", "replicate"], kind="stable"
    ).reset_index(drop=True)
    if not table["route_k_complete"].all():
        incomplete = table.loc[~table["route_k_complete"], "network_id"].tolist()
        raise ValueError(f"K={candidate_k} route generation incomplete for {incomplete}")
    return table


def _route_state(route: object, seed: int) -> RouteTrafficState:
    rng = np.random.default_rng(seed)
    congestion = float(rng.uniform(0.18, 0.82))
    rain = int(rng.random() < 0.30)
    incident = int(rng.random() < 0.18)
    historical_delay = float(rng.gamma(shape=1.5, scale=0.65))
    free_flow_speed = 60.0 * route.distance_km / route.base_travel_time_min
    speed = estimate_pretrip_speed_kph(
        route.distance_km,
        route.base_travel_time_min,
        congestion,
        rain,
        incident,
        sensor_adjustment_kph=float(rng.normal(0.0, 0.03 * free_flow_speed)),
    )
    return RouteTrafficState(
        hour_of_day=8,
        day_of_week=1,
        congestion=congestion,
        average_speed_kph=speed,
        rain_indicator=rain,
        incident_indicator=incident,
        historical_delay_minutes=historical_delay,
    )


def _risk_routing(
    config: dict[str, object],
    networks: dict[str, object],
    topology_labels: dict[str, str],
    conformal_model: object,
    distance_normalization_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_k = int(config["candidate_k"])
    weights = RiskWeights(**dict(config["risk_weights"]))
    score_tables: list[pd.DataFrame] = []
    decision_rows: list[dict[str, object]] = []
    inference_tables: list[pd.DataFrame] = []
    for network_index, network_id in enumerate(sorted(networks)):
        graph = networks[network_id]
        routes = generate_candidate_routes(
            graph,
            graph.graph["origin"],
            graph.graph["destination"],
            candidate_k,
            require_k=True,
        )
        states = {
            route.route_id: _route_state(
                route,
                int(config["risk_state_seed"]) + network_index * 100 + route.route_index,
            )
            for route in routes
        }
        features = build_route_feature_table(
            graph,
            routes,
            states,
            distance_normalization_km=distance_normalization_km,
        )
        intervals = conformal_model.predict(features)
        deadline = float(
            np.quantile(
                intervals["upper_bound"],
                float(config["deadline_upper_bound_quantile"]),
                method="linear",
            )
        )
        intervals.insert(0, "network_id", network_id)
        intervals.insert(1, "topology", topology_labels[network_id])
        intervals.insert(2, "replicate", int(graph.graph["replicate"]))
        intervals["deadline_minutes"] = deadline
        inference_tables.append(intervals.copy())

        for variant in config["risk_variants"]:
            scored = score_route_table(
                intervals,
                deadline,
                weights=weights,
                variant=str(variant),
            )
            scored.insert(3, "variant", str(variant))
            scored["conservative_deadline_violation"] = scored["upper_bound"] > deadline
            score_tables.append(scored)
            chosen = scored.sort_values(["risk_score", "route_id"], kind="stable").iloc[0]
            decision_rows.append(
                {
                    "network_id": network_id,
                    "topology": topology_labels[network_id],
                    "replicate": int(graph.graph["replicate"]),
                    "variant": str(variant),
                    "selected_route_id": str(chosen["route_id"]),
                    "prediction": float(chosen["prediction"]),
                    "interval_half_width": float(chosen["interval_half_width"]),
                    "upper_bound": float(chosen["upper_bound"]),
                    "deadline_minutes": deadline,
                    "deadline_margin": float(chosen["deadline_margin"]),
                    "conservative_deadline_violation": bool(
                        chosen["conservative_deadline_violation"]
                    ),
                    "risk_score": float(chosen["risk_score"]),
                }
            )

    decisions = pd.DataFrame(decision_rows)
    expected_choices = decisions[decisions["variant"] == "A"].set_index("network_id")[
        "selected_route_id"
    ]
    decisions["changed_from_expected_time"] = [
        row.selected_route_id != expected_choices.loc[row.network_id]
        for row in decisions.itertuples()
    ]
    return (
        pd.concat(inference_tables, ignore_index=True),
        pd.concat(score_tables, ignore_index=True),
        decisions,
    )


def _run_qaoa(
    config: dict[str, object],
    registry: pd.DataFrame,
    risk_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qaoa_config = dict(config["qaoa"])
    if not bool(qaoa_config["enabled"]):
        return pd.DataFrame(), pd.DataFrame()
    representative_ids = registry.loc[
        registry["replicate"] == int(qaoa_config["representative_replicate"]),
        "network_id",
    ].tolist()
    if len(representative_ids) != len(EXPECTED_FAMILIES):
        raise ValueError("QAOA study requires one representative per topology")
    d_scores = risk_scores[risk_scores["variant"] == "D"].copy()
    metric_rows: list[dict[str, object]] = []
    history_tables: list[pd.DataFrame] = []
    run_index = 0
    for network_id in sorted(representative_ids):
        group = d_scores[d_scores["network_id"] == network_id].copy()
        group["route_index"] = group["route_id"].str.removeprefix("x_").astype(int)
        group = group.sort_values("route_index", kind="stable")
        k = int(config["candidate_k"])
        if len(group) != k:
            raise ValueError(f"expected K={k} D-risk rows for {network_id}")
        risk_vector = group["risk_score"].to_numpy(dtype=float)
        qubo = build_risk_qubo(risk_vector)
        cost_hamiltonian, offset, _, _ = risk_qubo_to_ising(qubo)
        optimal_index = int(np.argmin(risk_vector))
        for method in qaoa_config["methods"]:
            run_index += 1
            expectation_qnode = create_expectation_qnode(
                cost_hamiltonian,
                number_of_qubits=k,
                depth=int(qaoa_config["depth"]),
                method=str(method),
            )
            probability_qnode = create_probability_qnode(
                cost_hamiltonian,
                number_of_qubits=k,
                depth=int(qaoa_config["depth"]),
                method=str(method),
            )
            optimized = optimize_qaoa_angles(
                expectation_qnode,
                depth=int(qaoa_config["depth"]),
                seed=int(qaoa_config["optimization_seed"]),
                iterations=int(qaoa_config["iterations"]),
                step_size=float(qaoa_config["step_size"]),
            )
            probabilities = np.asarray(
                probability_qnode(optimized.gammas, optimized.betas), dtype=float
            )
            summary = summarize_probabilities(
                probabilities, qubo, optimal_route_index=optimal_index
            )
            resources = circuit_resource_record(
                probability_qnode, optimized.gammas, optimized.betas
            )
            run_id = f"topology_qaoa_{run_index:03d}"
            metric_rows.append(
                {
                    "run_id": run_id,
                    "network_id": network_id,
                    "topology": str(group.iloc[0]["topology"]),
                    "replicate": int(group.iloc[0]["replicate"]),
                    "method": str(method),
                    "K": k,
                    "depth": int(qaoa_config["depth"]),
                    "optimization_seed": int(qaoa_config["optimization_seed"]),
                    "iterations": int(qaoa_config["iterations"]),
                    "step_size": float(qaoa_config["step_size"]),
                    "exact_optimal_route_index": optimal_index,
                    "exact_optimal_route_id": f"x_{optimal_index}",
                    "exact_optimal_risk": float(risk_vector[optimal_index]),
                    "initial_expected_qubo_energy": float(
                        optimized.history.iloc[0]["hamiltonian_expectation"] + offset
                    ),
                    "final_expected_qubo_energy": float(summary["expected_qubo_energy"]),
                    "analytic_valid_probability": float(summary["valid_probability"]),
                    "analytic_optimal_probability": float(summary["optimal_probability"]),
                    "most_probable_valid_route_index": int(
                        summary["most_probable_valid_route_index"]
                    ),
                    "most_probable_valid_is_optimal": int(
                        summary["most_probable_valid_route_index"]
                    )
                    == optimal_index,
                    "gammas_json": json.dumps(
                        [float(value) for value in optimized.gammas]
                    ),
                    "betas_json": json.dumps([float(value) for value in optimized.betas]),
                    **resources,
                }
            )
            history = optimized.history.copy()
            history.insert(0, "run_id", run_id)
            history.insert(1, "network_id", network_id)
            history.insert(2, "topology", str(group.iloc[0]["topology"]))
            history.insert(3, "method", str(method))
            history["expected_qubo_energy"] = history["hamiltonian_expectation"] + offset
            history_tables.append(history)
    return pd.DataFrame(metric_rows), pd.concat(history_tables, ignore_index=True)


def _exploratory_correlations(
    characteristics: pd.DataFrame,
    qaoa_runs: pd.DataFrame,
) -> pd.DataFrame:
    if qaoa_runs.empty:
        return pd.DataFrame()
    structure_metrics = [
        "node_count",
        "directed_edge_count",
        "mean_undirected_intersection_degree",
        "cv_base_travel_time",
        "route_mean_jaccard_diversity",
    ]
    outcomes = ["analytic_valid_probability", "analytic_optimal_probability"]
    representative = characteristics.merge(qaoa_runs, on=["network_id", "topology"])
    rows: list[dict[str, object]] = []
    for method, group in representative.groupby("method", sort=True):
        for structure_metric in structure_metrics:
            for outcome in outcomes:
                structure_values = group[structure_metric].to_numpy(dtype=float)
                outcome_values = group[outcome].to_numpy(dtype=float)
                structure_tolerance = max(
                    1e-12, 1e-10 * max(1.0, float(np.max(np.abs(structure_values))))
                )
                outcome_tolerance = max(
                    1e-12, 1e-10 * max(1.0, float(np.max(np.abs(outcome_values))))
                )
                structure_constant = float(np.ptp(structure_values)) <= structure_tolerance
                outcome_constant = float(np.ptp(outcome_values)) <= outcome_tolerance
                if structure_constant or outcome_constant:
                    correlation = float("nan")
                    undefined_reason = (
                        "near_constant_structure_metric"
                        if structure_constant
                        else "near_constant_qaoa_outcome"
                    )
                    correlation_defined = False
                else:
                    correlation = float(np.corrcoef(structure_values, outcome_values)[0, 1])
                    undefined_reason = ""
                    correlation_defined = True
                rows.append(
                    {
                        "method": method,
                        "structure_metric": structure_metric,
                        "qaoa_outcome": outcome,
                        "n_networks": len(group),
                        "pearson_r": correlation,
                        "correlation_defined": correlation_defined,
                        "undefined_reason": undefined_reason,
                        "interpretation": "exploratory_only_no_inference",
                    }
                )
    return pd.DataFrame(rows)


def _format_topology_rows(table: pd.DataFrame, value_columns: list[str]) -> str:
    header = "| " + " | ".join(["topology", *value_columns]) + " |"
    separator = "|" + "---|" * (len(value_columns) + 1)
    rows = [header, separator]
    for row in table.sort_values("topology").itertuples(index=False):
        values = [str(getattr(row, "topology"))]
        for column in value_columns:
            value = getattr(row, column)
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _validation_report(
    config: dict[str, object],
    registry: pd.DataFrame,
    characteristics: pd.DataFrame,
    model_overall: pd.DataFrame,
    model_by_topology: pd.DataFrame,
    coverage_overall: pd.DataFrame,
    coverage_by_topology: pd.DataFrame,
    risk_decisions: pd.DataFrame,
    qaoa_runs: pd.DataFrame,
    scenario_registry: pd.DataFrame,
) -> str:
    best_metrics = model_overall.iloc[0]
    coverage = coverage_overall.iloc[0]
    d_decisions = risk_decisions[risk_decisions["variant"] == "D"]
    qaoa_summary = (
        qaoa_runs.groupby("method", sort=True)
        .agg(
            run_count=("run_id", "size"),
            mean_valid_probability=("analytic_valid_probability", "mean"),
            mean_optimal_probability=("analytic_optimal_probability", "mean"),
            most_probable_valid_optimum_rate=("most_probable_valid_is_optimal", "mean"),
        )
        .reset_index()
        if not qaoa_runs.empty
        else pd.DataFrame()
    )
    model_topology = model_by_topology.rename(columns={"rmse": "test_rmse"})[
        ["topology", "n_samples", "mae", "test_rmse", "r2"]
    ]
    coverage_topology = coverage_by_topology[
        ["topology", "n_samples", "empirical_coverage", "mean_interval_width"]
    ]
    qaoa_lines = "QAOA was disabled."
    if not qaoa_summary.empty:
        qaoa_lines = "\n".join(
            f"- `{row.method}`: n={row.run_count}, mean valid={row.mean_valid_probability:.4f}, "
            f"mean optimal={row.mean_optimal_probability:.4f}, most-probable-valid optimum "
            f"rate={row.most_probable_valid_optimum_rate:.4f}."
            for row in qaoa_summary.itertuples()
        )
    return f"""# Validation Report — Steps 8.25 and 8.75

## Scope and validation status

- Status: **PASS** for this bounded repeated-synthetic benchmark.
- Networks: {len(registry)} total; exactly {registry.groupby('topology').size().min()}–{registry.groupby('topology').size().max()} instances per family across all five configured families.
- Candidate routing: K={int(config['candidate_k'])}; {int(characteristics['route_k_complete'].sum())}/{len(characteristics)} networks produced all K loopless routes.
- Evaluation scope: held-out observations from known synthetic networks. This is ID evaluation, not unseen-network generalization.
- Traffic: controlled simulation only; no live or real-city traffic is represented.
- Traffic variation: every network receives the same four scenario regimes, spanning incident probability {scenario_registry['incident_probability'].min():.2f}–{scenario_registry['incident_probability'].max():.2f}, congestion standard deviation {scenario_registry['congestion_std'].min():.2f}–{scenario_registry['congestion_std'].max():.2f}, and prediction-noise fraction {scenario_registry['prediction_noise_fraction'].min():.2f}–{scenario_registry['prediction_noise_fraction'].max():.2f}. Variation is within every topology, not a topology-specific confounder.
- Connectivity: all {len(characteristics)} generated graphs have one weak and one strong component; largest weak/strong component fractions are 1.0. This benchmark therefore does not study disconnected networks.
- QAOA: ideal noiseless simulator, one representative per topology, one fixed depth/seed/iteration budget per method.

## Point-model result

- Validation-selected model: `{best_metrics['model_name']}`.
- Held-out test rows: {int(best_metrics['n_samples'])}.
- Test MAE: {float(best_metrics['mae']):.4f} minutes.
- Test RMSE: {float(best_metrics['rmse']):.4f} minutes.
- Test R²: {float(best_metrics['r2']):.4f}.

{_format_topology_rows(model_topology, ['n_samples', 'mae', 'test_rmse', 'r2'])}

## Conformal result

- Method: locally scaled split conformal.
- Alpha: {float(config['conformal_alpha']):.2f}; nominal coverage: {1.0 - float(config['conformal_alpha']):.4f}.
- Empirical test coverage: {float(coverage['empirical_coverage']):.4f}.
- Mean interval width: {float(coverage['mean_interval_width']):.4f} minutes.

{_format_topology_rows(coverage_topology, ['n_samples', 'empirical_coverage', 'mean_interval_width'])}

## Risk-routing result

- Variant D changed the expected-time route in {int(d_decisions['changed_from_expected_time'].sum())}/{len(d_decisions)} network instances.
- Variant D conservative upper-bound deadline violations: {int(d_decisions['conservative_deadline_violation'].sum())}/{len(d_decisions)} selected routes.
- Mean selected-route deadline margin: {float(d_decisions['deadline_margin'].mean()):.4f} minutes.
- These are model-based conservative decisions, not observed emergency outcomes.

## Compact QAOA comparison

{qaoa_lines}

Wall-clock timing is intentionally not used to claim a quantum speedup.

## Scientific limitations

1. Three instances per family are enough for a repeated smoke benchmark, not universal topology claims.
2. Training and test rows come from the same 15 network instances; unseen-network evaluation belongs to Step 9.
3. Correlations use only five QAOA representative networks and are explicitly exploratory, with no significance claim.
   Correlations against feasible-XY valid probability are undefined because validity is structurally 1.0; the CSV stores NaN plus `near_constant_qaoa_outcome` rather than correlating floating-point noise.
4. The traffic generator is controlled synthetic data, even when graph structure is road-like.
5. QAOA uses an ideal simulator, K=4, depth {int(dict(config['qaoa'])['depth'])}, one optimization seed, and {int(dict(config['qaoa'])['iterations'])} iterations.
6. No hardware, noise, runtime advantage, medical effectiveness, or real-world deployment claim is supported.

## Reproduction

Run `python experiments/run_steps8_25_8_75_topology.py` from the repository root. The runner refuses to overwrite an existing output directory. Exact seeds, parameters, package versions, config hash, source hashes, and artifact hashes are stored in `manifest.json`.
"""


def run() -> None:
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            f"refusing to overwrite existing benchmark directory: {OUTPUT_DIRECTORY}"
        )
    config = _load_config()
    scenario_registry = _build_scenario_registry(config)
    networks, topology_labels, registry = _build_networks(config)
    candidate_k = int(config["candidate_k"])
    characteristics = _characterize_instances(networks, registry, candidate_k)

    traffic = generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=list(config["traffic_scenarios"]),
        observations_per_network_per_scenario=int(
            config["observations_per_network_per_scenario"]
        ),
        random_seed=int(config["traffic_seed"]),
    )
    splits = split_traffic_observations(
        traffic,
        split_fractions=dict(config["split_fractions"]),
        random_seed=int(config["split_seed"]),
    )
    validate_split_separation(splits, expected_observation_ids=traffic["observation_id"])
    if tuple(MODEL_FEATURE_COLUMNS) != FEATURE_COLUMNS:
        raise RuntimeError("traffic and model feature contracts disagree")
    if {"network_id", "topology"}.intersection(FEATURE_COLUMNS):
        raise RuntimeError("network identity cannot be a point-model feature")

    fitted_models = train_all_traffic_models(
        splits["train"],
        random_state=int(config["model_seed"]),
        random_forest_trees=int(config["random_forest_trees"]),
    )
    selection = select_best_by_validation_rmse(fitted_models, splits["validation"])
    model_evaluation = evaluate_traffic_model(selection.best_model, splits["test"])
    model_overall = model_evaluation.overall.copy()
    model_overall.insert(0, "model_name", selection.best_model_name)
    model_overall.insert(1, "evaluation_scope", "held_out_rows_known_networks")

    scale_model = train_residual_scale_model(
        selection.best_model,
        splits["validation"],
        random_state=int(config["residual_scale_seed"]),
        n_estimators=int(config["residual_scale_trees"]),
        minimum_scale_minutes=float(config["residual_scale_minimum_minutes"]),
    )
    conformal_model = calibrate_locally_scaled_split_conformal(
        selection.best_model,
        scale_model,
        splits["calibration"],
        alpha=float(config["conformal_alpha"]),
    )
    conformal_evaluation = evaluate_conformal(conformal_model, splits["test"])

    normalization_values = traffic["distance_normalization_km"].unique()
    if len(normalization_values) != 1:
        raise RuntimeError("traffic data must use one global distance normalization")
    inference, risk_scores, risk_decisions = _risk_routing(
        config,
        networks,
        topology_labels,
        conformal_model,
        float(normalization_values[0]),
    )
    qaoa_runs, qaoa_history = _run_qaoa(config, registry, risk_scores)
    correlations = _exploratory_correlations(characteristics, qaoa_runs)

    characteristics_by_topology = (
        characteristics.groupby("topology", sort=True)
        .agg(
            instance_count=("network_id", "size"),
            mean_node_count=("node_count", "mean"),
            mean_directed_edge_count=("directed_edge_count", "mean"),
            mean_intersection_degree=("mean_undirected_intersection_degree", "mean"),
            mean_edge_length_km=("mean_edge_length_km", "mean"),
            mean_base_travel_time_min=("mean_base_travel_time_min", "mean"),
            mean_route_jaccard_diversity=("route_mean_jaccard_diversity", "mean"),
            mean_weak_component_count=("weak_component_count", "mean"),
            mean_strong_component_count=("strong_component_count", "mean"),
            minimum_largest_weak_component_fraction=(
                "largest_weak_component_fraction",
                "min",
            ),
            minimum_largest_strong_component_fraction=(
                "largest_strong_component_fraction",
                "min",
            ),
        )
        .reset_index()
    )
    risk_summary = (
        risk_decisions.groupby(["topology", "variant"], sort=True)
        .agg(
            instance_count=("network_id", "size"),
            changed_from_expected_count=("changed_from_expected_time", "sum"),
            changed_from_expected_rate=("changed_from_expected_time", "mean"),
            conservative_violation_count=("conservative_deadline_violation", "sum"),
            conservative_violation_rate=("conservative_deadline_violation", "mean"),
            mean_deadline_margin=("deadline_margin", "mean"),
        )
        .reset_index()
    )
    qaoa_aggregate = (
        qaoa_runs.groupby("method", sort=True)
        .agg(
            run_count=("run_id", "size"),
            mean_valid_probability=("analytic_valid_probability", "mean"),
            mean_optimal_probability=("analytic_optimal_probability", "mean"),
            most_probable_valid_optimum_rate=("most_probable_valid_is_optimal", "mean"),
            mean_circuit_depth=("circuit_depth", "mean"),
            mean_gate_count=("gate_count", "mean"),
        )
        .reset_index()
        if not qaoa_runs.empty
        else pd.DataFrame()
    )

    expected_rows = len(config["instances"])
    if len(registry) != expected_rows or len(characteristics) != expected_rows:
        raise RuntimeError("instance registry and characterization row counts disagree")
    if not np.isfinite(
        characteristics.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).all():
        raise RuntimeError("network characterization contains non-finite values")
    if not (
        (characteristics["weak_component_count"] == 1).all()
        and (characteristics["strong_component_count"] == 1).all()
        and np.allclose(characteristics["largest_weak_component_fraction"], 1.0)
        and np.allclose(characteristics["largest_strong_component_fraction"], 1.0)
    ):
        raise RuntimeError("benchmark expects every generated graph to be strongly connected")
    if not qaoa_runs.empty:
        expected_qaoa_runs = len(EXPECTED_FAMILIES) * len(dict(config["qaoa"])["methods"])
        if len(qaoa_runs) != expected_qaoa_runs:
            raise RuntimeError("QAOA run count does not match the compact design")
        xy_valid = qaoa_runs.loc[
            qaoa_runs["method"] == "feasible_xy", "analytic_valid_probability"
        ]
        if not np.allclose(xy_valid, 1.0, atol=1e-8):
            raise RuntimeError("feasible XY circuit left the one-hot subspace")
        xy_valid_correlations = correlations.loc[
            (correlations["method"] == "feasible_xy")
            & (correlations["qaoa_outcome"] == "analytic_valid_probability")
        ]
        if not (
            len(xy_valid_correlations) == 5
            and (~xy_valid_correlations["correlation_defined"]).all()
            and xy_valid_correlations["pearson_r"].isna().all()
            and (
                xy_valid_correlations["undefined_reason"]
                == "near_constant_qaoa_outcome"
            ).all()
        ):
            raise RuntimeError(
                "near-constant feasible-XY validity correlations must be undefined"
            )

    split_lookup = {
        observation_id: split_name
        for split_name, frame in splits.items()
        for observation_id in frame["observation_id"]
    }
    traffic_artifact = traffic.copy()
    traffic_artifact.insert(1, "split", traffic_artifact["observation_id"].map(split_lookup))
    report = _validation_report(
        config,
        registry,
        characteristics,
        model_overall,
        model_evaluation.by_topology,
        conformal_evaluation.overall,
        conformal_evaluation.by_topology,
        risk_decisions,
        qaoa_runs,
        scenario_registry,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    tables = {
        "instance_registry.csv": registry,
        "network_characteristics.csv": characteristics,
        "network_characteristics_by_topology.csv": characteristics_by_topology,
        "traffic_scenario_registry.csv": scenario_registry,
        "traffic_observations.csv": traffic_artifact,
        "model_selection.csv": selection.ranking,
        "model_metrics_overall.csv": model_overall,
        "model_metrics_by_network.csv": model_evaluation.by_network,
        "model_metrics_by_topology.csv": model_evaluation.by_topology,
        "model_test_predictions.csv": model_evaluation.predictions,
        "conformal_coverage_overall.csv": conformal_evaluation.overall,
        "conformal_coverage_by_network.csv": conformal_evaluation.by_network,
        "conformal_coverage_by_topology.csv": conformal_evaluation.by_topology,
        "conformal_test_intervals.csv": conformal_evaluation.intervals,
        "risk_candidate_intervals.csv": inference,
        "risk_route_scores.csv": risk_scores,
        "risk_decisions.csv": risk_decisions,
        "risk_summary_by_topology.csv": risk_summary,
        "qaoa_runs.csv": qaoa_runs,
        "qaoa_training_history.csv": qaoa_history,
        "qaoa_aggregate.csv": qaoa_aggregate,
        "exploratory_structure_qaoa_correlations.csv": correlations,
    }
    for filename, table in tables.items():
        table.to_csv(OUTPUT_DIRECTORY / filename, index=False)
    joblib.dump(selection.best_model, OUTPUT_DIRECTORY / "best_model.joblib")
    joblib.dump(scale_model, OUTPUT_DIRECTORY / "residual_scale_model.joblib")
    write_json(
        OUTPUT_DIRECTORY / "conformal_calibration.json",
        {
            "method": "locally_scaled_split_conformal",
            "alpha": conformal_model.alpha,
            "nominal_coverage": 1.0 - conformal_model.alpha,
            "n_calibration": conformal_model.n_calibration,
            "finite_sample_quantile_level": conformal_model.quantile_level,
            "calibration_quantile": conformal_model.calibration_quantile,
            "quantile_units": "normalized_residual_ratio",
        },
    )
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    source_files = [
        CONFIG_PATH,
        Path(__file__),
        REPOSITORY_ROOT / "src" / "synthetic_networks.py",
        REPOSITORY_ROOT / "src" / "network_schema.py",
        REPOSITORY_ROOT / "src" / "network_characterization.py",
        REPOSITORY_ROOT / "src" / "candidate_routes.py",
        REPOSITORY_ROOT / "src" / "traffic_data.py",
        REPOSITORY_ROOT / "src" / "traffic_scenarios.py",
        REPOSITORY_ROOT / "src" / "traffic_model.py",
        REPOSITORY_ROOT / "src" / "conformal.py",
        REPOSITORY_ROOT / "src" / "route_features.py",
        REPOSITORY_ROOT / "src" / "deadline_risk.py",
        REPOSITORY_ROOT / "src" / "risk_qubo.py",
        REPOSITORY_ROOT / "src" / "qubo.py",
        REPOSITORY_ROOT / "src" / "feasibility_qaoa.py",
        REPOSITORY_ROOT / "src" / "reproducibility.py",
        REPOSITORY_ROOT / "requirements.txt",
        REPOSITORY_ROOT / "requirements-research.txt",
    ]
    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "traffic": int(config["traffic_seed"]),
            "split": int(config["split_seed"]),
            "model": int(config["model_seed"]),
            "residual_scale": int(config["residual_scale_seed"]),
            "risk_state": int(config["risk_state_seed"]),
            "qaoa_optimization": int(dict(config["qaoa"])["optimization_seed"]),
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "evaluation_scope": "held_out_rows_known_synthetic_networks",
            "network_count": len(registry),
            "instances_per_topology": registry.groupby("topology").size().to_dict(),
            "all_networks_k_complete": bool(characteristics["route_k_complete"].all()),
            "selected_model": selection.best_model_name,
            "qaoa_status": "completed" if not qaoa_runs.empty else "disabled_by_config",
            "qaoa_run_count": len(qaoa_runs),
            "claims_excluded": [
                "quantum_speedup",
                "real_traffic_effectiveness",
                "unseen_network_generalization",
                "hardware_performance",
                "medical_or_operational_effectiveness",
            ],
            "source_sha256": {
                str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): file_sha256(path)
                for path in source_files
            },
            "artifact_sha256": {
                path.name: file_sha256(path)
                for path in sorted(OUTPUT_DIRECTORY.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        }
    )
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)

    print(
        f"Validated {len(registry)} networks across {registry['topology'].nunique()} "
        f"families; all K={candidate_k} complete."
    )
    print(model_overall.to_string(index=False))
    print(conformal_evaluation.overall.to_string(index=False))
    if not qaoa_aggregate.empty:
        print(qaoa_aggregate.to_string(index=False))
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run()
