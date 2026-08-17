"""Offline-only Step 8.5 evaluation on cached real road topology."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import src.osm_adapter as osm_adapter
from src.candidate_routes import generate_candidate_routes
from src.classical_baselines import (
    dijkstra_expected_time_path,
    dijkstra_shortest_distance_path,
)
from src.conformal import (
    LocallyScaledSplitConformalRegressor,
    evaluate_conformal,
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
from src.network_loader import load_network_graphml
from src.reproducibility import build_experiment_manifest, file_sha256, write_json
from src.risk_qubo import build_risk_qubo, risk_qubo_to_ising
from src.route_features import (
    RouteTrafficState,
    build_route_feature_table,
    estimate_pretrip_speed_kph,
)
from src.traffic_data import generate_historical_trip_observations
from src.traffic_model import evaluate_traffic_model


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "real_topology_benchmark.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step8_5_real_topology"


def _activate_offline_guard() -> None:
    """Make accidental network access fail during the replay stage."""

    def forbidden(*args, **kwargs):
        raise RuntimeError("network access is forbidden in the Step 8.5 offline runner")

    socket.create_connection = forbidden  # type: ignore[assignment]
    osm_adapter.urlopen = forbidden


def _labeled(table: pd.DataFrame, label: str) -> pd.DataFrame:
    result = table.copy()
    if "study_label" in result.columns:
        result["study_label"] = label
    else:
        result.insert(0, "study_label", label)
    return result


def _load_cached_networks(
    config: dict[str, object],
) -> tuple[dict[str, nx.DiGraph], pd.DataFrame, dict[str, object]]:
    cache_directory = REPOSITORY_ROOT / str(config["cache_directory"])
    manifest_path = REPOSITORY_ROOT / str(config["cache_manifest"])
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = pd.read_csv(cache_directory / "cache_registry.csv")
    if len(registry) != len(config["regions"]):
        raise RuntimeError("cache registry and configured regions disagree")

    networks: dict[str, nx.DiGraph] = {}
    for row in registry.itertuples(index=False):
        graph_path = cache_directory / str(row.graphml_file)
        raw_path = cache_directory / str(row.raw_file)
        if file_sha256(graph_path) != str(row.graphml_sha256):
            raise RuntimeError(f"GraphML hash mismatch for {row.network_id}")
        if file_sha256(raw_path) != str(row.raw_sha256):
            raise RuntimeError(f"raw Overpass hash mismatch for {row.network_id}")
        graph = load_network_graphml(graph_path)
        if str(graph.graph.get("attribution")) != str(row.attribution):
            raise RuntimeError(f"attribution lost for {row.network_id}")
        if str(graph.graph.get("overpass_query")) != str(row.overpass_query):
            raise RuntimeError(f"query provenance lost for {row.network_id}")
        routes = generate_candidate_routes(
            graph,
            graph.graph["origin"],
            graph.graph["destination"],
            int(config["candidate_k"]),
            require_k=True,
        )
        if len(routes) != int(config["candidate_k"]):
            raise RuntimeError(f"{row.network_id} failed offline K completeness")
        networks[str(row.network_id)] = graph
    return networks, registry, cache_manifest


def _frozen_models(
    config: dict[str, object],
) -> tuple[object, LocallyScaledSplitConformalRegressor, float, dict[str, str]]:
    model_path = REPOSITORY_ROOT / str(config["frozen_step4_model"])
    train_path = REPOSITORY_ROOT / str(config["frozen_step4_train"])
    scale_path = REPOSITORY_ROOT / str(config["frozen_step5_scale_model"])
    quantile_path = REPOSITORY_ROOT / str(config["frozen_step5_quantiles"])
    base_model = joblib.load(model_path)
    scale_model = joblib.load(scale_path)
    training = pd.read_csv(train_path, usecols=["distance_normalization_km"])
    scales = training["distance_normalization_km"].unique()
    if len(scales) != 1:
        raise RuntimeError("frozen Step 4 training data must contain one distance scale")
    frozen_distance_scale = float(scales[0])

    quantiles = pd.read_csv(quantile_path)
    selected = quantiles[
        (quantiles["method"] == str(config["conformal_method"]))
        & np.isclose(quantiles["alpha"], float(config["conformal_alpha"]))
    ]
    if len(selected) != 1:
        raise RuntimeError("frozen Step 5 local conformal quantile is ambiguous")
    row = selected.iloc[0]
    conformal_model = LocallyScaledSplitConformalRegressor(
        base_model=base_model,
        scale_model=scale_model,
        alpha=float(row["alpha"]),
        calibration_quantile=float(row["calibration_residual_quantile"]),
        quantile_level=float(row["finite_sample_quantile_level"]),
        n_calibration=int(row["n_calibration"]),
    )
    hashes = {
        str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): file_sha256(path)
        for path in (model_path, train_path, scale_path, quantile_path)
    }
    return base_model, conformal_model, frozen_distance_scale, hashes


def _characteristics(
    networks: dict[str, nx.DiGraph],
    registry: pd.DataFrame,
    candidate_k: int,
) -> pd.DataFrame:
    lookup = registry.set_index("network_id")
    rows: list[dict[str, object]] = []
    for network_id in sorted(networks):
        graph = networks[network_id]
        record = characterize_network(
            graph, route_k=candidate_k, require_route_k=True
        ).as_record()
        degree_distribution = record.pop("intersection_degree_distribution")
        source = lookup.loc[network_id]
        rows.append(
            {
                "network_id": network_id,
                "topology": str(source["topology_class"]),
                "region_label": str(source["region_label"]),
                "selection_rationale": str(source["selection_rationale"]),
                **record,
                "intersection_degree_distribution_json": json.dumps(
                    degree_distribution, sort_keys=True
                ),
                "strong_component_count": nx.number_strongly_connected_components(graph),
                "parallel_edges_collapsed": int(
                    graph.graph.get("parallel_edges_collapsed", 0)
                ),
                "parallel_edge_projection_caveat": str(
                    graph.graph.get("parallel_edge_projection_caveat", "none")
                ),
            }
        )
    return pd.DataFrame(rows)


def _route_state(route: object, seed: int) -> RouteTrafficState:
    rng = np.random.default_rng(seed)
    congestion = float(rng.uniform(0.20, 0.82))
    rain = int(rng.random() < 0.30)
    incident = int(rng.random() < 0.18)
    historical_delay = float(rng.gamma(1.5, 0.65))
    free_speed = 60.0 * route.distance_km / route.base_travel_time_min
    speed = estimate_pretrip_speed_kph(
        route.distance_km,
        route.base_travel_time_min,
        congestion,
        rain,
        incident,
        sensor_adjustment_kph=float(rng.normal(0.0, 0.03 * free_speed)),
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


def _candidate_risk(
    config: dict[str, object],
    networks: dict[str, nx.DiGraph],
    conformal_model: LocallyScaledSplitConformalRegressor,
    frozen_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interval_tables: list[pd.DataFrame] = []
    score_tables: list[pd.DataFrame] = []
    decisions: list[dict[str, object]] = []
    weights = RiskWeights(**dict(config["risk_weights"]))
    for network_index, network_id in enumerate(sorted(networks)):
        graph = networks[network_id]
        routes = generate_candidate_routes(
            graph,
            graph.graph["origin"],
            graph.graph["destination"],
            int(config["candidate_k"]),
            require_k=True,
        )
        states = {
            route.route_id: _route_state(
                route,
                int(config["risk_state_seed"])
                + network_index * 100
                + route.route_index,
            )
            for route in routes
        }
        features = build_route_feature_table(
            graph,
            routes,
            states,
            distance_normalization_km=frozen_scale,
        )
        intervals = conformal_model.predict(features)
        intervals.insert(0, "network_id", network_id)
        intervals.insert(1, "topology", str(graph.graph["topology"]))
        deadline = float(
            np.quantile(
                intervals["upper_bound"],
                float(config["deadline_upper_bound_quantile"]),
                method="linear",
            )
        )
        intervals["deadline_minutes"] = deadline
        interval_tables.append(intervals)
        for variant in config["risk_variants"]:
            scored = score_route_table(
                intervals, deadline, weights=weights, variant=str(variant)
            )
            scored.insert(3, "variant", str(variant))
            score_tables.append(scored)
            chosen = scored.sort_values(["risk_score", "route_id"]).iloc[0]
            decisions.append(
                {
                    "network_id": network_id,
                    "topology": str(graph.graph["topology"]),
                    "variant": str(variant),
                    "selected_route_id": str(chosen["route_id"]),
                    "prediction": float(chosen["prediction"]),
                    "interval_half_width": float(chosen["interval_half_width"]),
                    "upper_bound": float(chosen["upper_bound"]),
                    "deadline_minutes": deadline,
                    "deadline_margin": float(chosen["deadline_margin"]),
                    "risk_score": float(chosen["risk_score"]),
                }
            )
    decision_table = pd.DataFrame(decisions)
    expected = decision_table[decision_table["variant"] == "A"].set_index(
        "network_id"
    )["selected_route_id"]
    decision_table["changed_from_expected_time"] = [
        row.selected_route_id != expected.loc[row.network_id]
        for row in decision_table.itertuples()
    ]
    return (
        pd.concat(interval_tables, ignore_index=True),
        pd.concat(score_tables, ignore_index=True),
        decision_table,
    )


def _distance_policy(networks: dict[str, nx.DiGraph]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for network_id in sorted(networks):
        graph = networks[network_id]
        collapsed = int(graph.graph.get("parallel_edges_collapsed", 0))
        expected = dijkstra_expected_time_path(
            graph, graph.graph["origin"], graph.graph["destination"]
        )
        row: dict[str, object] = {
            "network_id": network_id,
            "topology": str(graph.graph["topology"]),
            "parallel_edges_collapsed": collapsed,
            "projection_policy": str(graph.graph.get("parallel_edge_policy")),
            "projection_caveat": str(
                graph.graph.get("parallel_edge_projection_caveat")
            ),
            "graph_wide_distance_status": (
                "excluded_source_fastest_time_projection"
                if collapsed
                else "valid_no_source_parallel_edges_collapsed"
            ),
            "expected_time_path": " -> ".join(map(str, expected.path)),
            "expected_time_path_minutes": expected.objective_value,
        }
        if collapsed:
            row.update(
                distance_path=None,
                distance_km=np.nan,
                distance_path_base_time_min=np.nan,
            )
        else:
            distance = dijkstra_shortest_distance_path(
                graph, graph.graph["origin"], graph.graph["destination"]
            )
            row.update(
                distance_path=" -> ".join(map(str, distance.path)),
                distance_km=distance.distance_km,
                distance_path_base_time_min=distance.expected_travel_time_min,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _qaoa(
    config: dict[str, object], risk_scores: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qconfig = dict(config["qaoa"])
    if not bool(qconfig["enabled"]):
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, object]] = []
    histories: list[pd.DataFrame] = []
    d_scores = risk_scores[risk_scores["variant"] == "D"].copy()
    for network_index, (network_id, group) in enumerate(
        d_scores.groupby("network_id", sort=True), start=1
    ):
        group = group.assign(
            route_index=group["route_id"].str.removeprefix("x_").astype(int)
        ).sort_values("route_index")
        risk_vector = group["risk_score"].to_numpy(dtype=float)
        qubo = build_risk_qubo(risk_vector)
        hamiltonian, offset, _, _ = risk_qubo_to_ising(qubo)
        optimal_index = int(np.argmin(risk_vector))
        for method_index, method in enumerate(qconfig["methods"], start=1):
            qnode = create_expectation_qnode(
                hamiltonian,
                number_of_qubits=len(risk_vector),
                depth=int(qconfig["depth"]),
                method=str(method),
            )
            probability_qnode = create_probability_qnode(
                hamiltonian,
                number_of_qubits=len(risk_vector),
                depth=int(qconfig["depth"]),
                method=str(method),
            )
            optimized = optimize_qaoa_angles(
                qnode,
                depth=int(qconfig["depth"]),
                seed=int(qconfig["optimization_seed"]),
                iterations=int(qconfig["iterations"]),
                step_size=float(qconfig["step_size"]),
            )
            probabilities = np.asarray(
                probability_qnode(optimized.gammas, optimized.betas), dtype=float
            )
            summary = summarize_probabilities(
                probabilities, qubo, optimal_route_index=optimal_index
            )
            run_id = f"real_qaoa_{network_index:02d}_{method_index:02d}"
            rows.append(
                {
                    "run_id": run_id,
                    "network_id": network_id,
                    "topology": str(group.iloc[0]["topology"]),
                    "method": str(method),
                    "K": len(risk_vector),
                    "depth": int(qconfig["depth"]),
                    "iterations": int(qconfig["iterations"]),
                    "optimization_seed": int(qconfig["optimization_seed"]),
                    "exact_optimal_route_id": f"x_{optimal_index}",
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
                    **circuit_resource_record(
                        probability_qnode, optimized.gammas, optimized.betas
                    ),
                }
            )
            history = optimized.history.copy()
            history.insert(0, "run_id", run_id)
            history["expected_qubo_energy"] = history["hamiltonian_expectation"] + offset
            histories.append(history)
    return pd.DataFrame(rows), pd.concat(histories, ignore_index=True)


def _report(
    config: dict[str, object],
    registry: pd.DataFrame,
    characteristics: pd.DataFrame,
    model_overall: pd.DataFrame,
    conformal_overall: pd.DataFrame,
    distance_policy: pd.DataFrame,
    decisions: pd.DataFrame,
    qaoa_runs: pd.DataFrame,
    frozen_scale: float,
) -> str:
    label = str(config["study_label"])
    changed = int(decisions["changed_from_expected_time"].sum())
    projected = int((distance_policy["parallel_edges_collapsed"] > 0).sum())
    xy = qaoa_runs[qaoa_runs["method"] == "feasible_xy"]
    return f"""# Step 8.5 real-topology validation report

## Scope

**{label}.** This is not observed, historical, or live traffic. Four compact
OpenStreetMap extracts were downloaded once, cached with ODbL attribution and
source hashes, and replayed under an active no-network guard. Place labels and
bounding boxes are configuration data; no routing or learning branch uses a city name.

## Validation summary

- Cached networks: {len(registry)} across {registry['topology_class'].nunique()} topology motivations.
- K={int(config['candidate_k'])} complete networks: {int(characteristics['route_k_complete'].sum())}/{len(characteristics)}.
- Frozen Step 4 distance normalization: {frozen_scale:.12f} km, read from the original training split rather than recomputed on this real batch.
- Frozen point-model test MAE/RMSE/R2: {float(model_overall.iloc[0]['mae']):.3f} / {float(model_overall.iloc[0]['rmse']):.3f} / {float(model_overall.iloc[0]['r2']):.3f}.
- Frozen local conformal nominal/empirical coverage: {float(conformal_overall.iloc[0]['nominal_coverage']):.3f} / {float(conformal_overall.iloc[0]['empirical_coverage']):.3f}.
- Risk decisions changed from expected-time selection: {changed}/{len(decisions)} variant decisions.
- Source fastest-time projections excluded from full-multigraph distance claims: {projected}/{len(distance_policy)} networks.
- Compact simulator QAOA runs: {len(qaoa_runs)}; feasible-XY validity range {float(xy['analytic_valid_probability'].min()):.6f}-{float(xy['analytic_valid_probability'].max()):.6f}.

## Scientific interpretation

These results test empirical transfer to real road structure under a controlled
simulator. They do not establish real-traffic effectiveness, universal
cross-network generalization, quantum advantage, operational emergency benefit,
or hardware performance. Local conformal intervals are frozen from Step 5;
coverage on these shifted networks is measured rather than assumed.

## Projection policy

The OSM adapter projects parallel roads to the minimum-base-time edge required
by the canonical DiGraph. Every projection is counted and labeled. A graph-wide
distance result is omitted whenever that source projection could have removed a
shorter parallel road. The cached raw Overpass response remains the audit source.

## Reproducibility

Run the online cache stage once with
`python experiments/cache_step8_5_real_topologies.py`. Thereafter run this file.
It verifies raw/GraphML hashes and activates a guard that raises on attempted
network access. Artifact hashes and frozen-model hashes are in `manifest.json`.
No quantum hardware was used.
"""


def run() -> None:
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIRECTORY}")
    _activate_offline_guard()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    label = str(config["study_label"])
    networks, registry, cache_manifest = _load_cached_networks(config)
    base_model, conformal_model, frozen_scale, frozen_hashes = _frozen_models(config)
    topology_labels = {
        network_id: str(graph.graph["topology"])
        for network_id, graph in networks.items()
    }
    characteristics = _characteristics(
        networks, registry, int(config["candidate_k"])
    )

    traffic = generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=[str(value) for value in config["traffic_scenarios"]],
        observations_per_network_per_scenario=int(
            config["observations_per_network_per_scenario"]
        ),
        random_seed=int(config["traffic_seed"]),
    )
    traffic["distance_normalization_km"] = frozen_scale
    traffic["normalized_distance"] = traffic["route_distance_km"] / frozen_scale
    model_evaluation = evaluate_traffic_model(base_model, traffic)
    conformal_evaluation = evaluate_conformal(conformal_model, traffic)
    candidate_intervals, risk_scores, risk_decisions = _candidate_risk(
        config, networks, conformal_model, frozen_scale
    )
    distance_policy = _distance_policy(networks)
    qaoa_runs, qaoa_history = _qaoa(config, risk_scores)
    qaoa_aggregate = (
        qaoa_runs.groupby("method", sort=True)
        .agg(
            run_count=("run_id", "size"),
            mean_valid_probability=("analytic_valid_probability", "mean"),
            mean_optimal_probability=("analytic_optimal_probability", "mean"),
            most_probable_valid_optimum_rate=("most_probable_valid_is_optimal", "mean"),
        )
        .reset_index()
    )
    xy_valid = qaoa_runs.loc[
        qaoa_runs["method"] == "feasible_xy", "analytic_valid_probability"
    ]
    if not np.allclose(xy_valid, 1.0, atol=1e-8):
        raise RuntimeError("feasible XY left the one-hot subspace")
    if not characteristics["route_k_complete"].all():
        raise RuntimeError("real-topology K completeness failed")
    if not np.allclose(traffic["distance_normalization_km"], frozen_scale):
        raise RuntimeError("real batch replaced the frozen Step 4 normalization")

    report = _report(
        config,
        registry,
        characteristics,
        model_evaluation.overall,
        conformal_evaluation.overall,
        distance_policy,
        risk_decisions,
        qaoa_runs,
        frozen_scale,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    tables = {
        "network_registry.csv": registry,
        "network_characteristics.csv": characteristics,
        "controlled_traffic_observations.csv": traffic,
        "model_metrics_overall.csv": model_evaluation.overall,
        "model_metrics_by_network.csv": model_evaluation.by_network,
        "model_metrics_by_topology.csv": model_evaluation.by_topology,
        "model_predictions.csv": model_evaluation.predictions,
        "conformal_coverage_overall.csv": conformal_evaluation.overall,
        "conformal_coverage_by_network.csv": conformal_evaluation.by_network,
        "conformal_coverage_by_topology.csv": conformal_evaluation.by_topology,
        "conformal_intervals.csv": conformal_evaluation.intervals,
        "candidate_route_intervals.csv": candidate_intervals,
        "risk_route_scores.csv": risk_scores,
        "risk_decisions.csv": risk_decisions,
        "graph_distance_policy.csv": distance_policy,
        "qaoa_runs.csv": qaoa_runs,
        "qaoa_training_history.csv": qaoa_history,
        "qaoa_aggregate.csv": qaoa_aggregate,
    }
    for filename, table in tables.items():
        labeled = _labeled(table, label)
        if not (labeled["study_label"] == label).all():
            raise RuntimeError(f"missing controlled-traffic label in {filename}")
        labeled.to_csv(OUTPUT_DIRECTORY / filename, index=False)
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "controlled_traffic": int(config["traffic_seed"]),
            "risk_state": int(config["risk_state_seed"]),
            "qaoa_optimization": int(dict(config["qaoa"])["optimization_seed"]),
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "study_label": label,
            "network_access": "forbidden_by_active_runtime_guard",
            "offline_replay_proved": True,
            "cache_manifest_sha256": file_sha256(
                REPOSITORY_ROOT / str(config["cache_manifest"])
            ),
            "cache_network_hashes": {
                key: value["graphml_sha256"]
                for key, value in dict(cache_manifest["files"]).items()
            },
            "frozen_artifact_sha256": frozen_hashes,
            "frozen_step4_distance_normalization_km": frozen_scale,
            "all_networks_k_complete": True,
            "qaoa_run_count": len(qaoa_runs),
            "feasible_xy_preserved_one_hot": bool(np.allclose(xy_valid, 1.0, atol=1e-8)),
            "distance_projection_exclusion_count": int(
                (distance_policy["parallel_edges_collapsed"] > 0).sum()
            ),
            "claims_excluded": [
                "real_traffic_effectiveness",
                "live_traffic",
                "universal_generalization",
                "quantum_speedup",
                "hardware_performance",
                "medical_or_operational_effectiveness",
            ],
            "source_sha256": {
                "configs/real_topology_benchmark.json": file_sha256(CONFIG_PATH),
                "experiments/run_step8_5_real_topology_offline.py": file_sha256(
                    Path(__file__)
                ),
                "src/osm_adapter.py": file_sha256(
                    REPOSITORY_ROOT / "src" / "osm_adapter.py"
                ),
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
        f"Offline replay validated {len(networks)} cached networks; "
        f"all K={config['candidate_k']} complete.",
        flush=True,
    )
    print(model_evaluation.overall.to_string(index=False), flush=True)
    print(conformal_evaluation.overall.to_string(index=False), flush=True)
    print(qaoa_aggregate.to_string(index=False), flush=True)
    print(f"Artifacts: {OUTPUT_DIRECTORY}", flush=True)


if __name__ == "__main__":
    run()
