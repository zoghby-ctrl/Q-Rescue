"""Run the publication-oriented Step 9.2 feasibility-without-utility study.

The primary design contains 300 matched routing problems and 1,800 ideal
statevector QAOA trials.  Preparation, runtime preflight, optimization, and
finalization are separate resumable stages.  Every completed optimization is
stored as one atomically replaced JSON checkpoint before the next progress
update is reported.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.candidate_routes import (  # noqa: E402
    CandidateRoute,
    generate_candidate_routes,
    summarize_route_diversity,
)
from src.conformal import finite_sample_residual_quantile  # noqa: E402
from src.controlled_objectives import controlled_objective_table  # noqa: E402
from src.feasibility_qaoa import (  # noqa: E402
    circuit_resource_record,
    create_expectation_qnode,
    create_probability_qnode,
    optimize_qaoa_angles,
)
from src.network_characterization import characterize_network  # noqa: E402
from src.network_schema import validate_network  # noqa: E402
from src.qaoa_diagnostics import summarize_qaoa_distribution  # noqa: E402
from src.reproducibility import (  # noqa: E402
    build_experiment_manifest,
    file_sha256,
)
from src.risk_qubo import build_risk_qubo, risk_qubo_to_ising  # noqa: E402
from src.route_features import (  # noqa: E402
    RouteTrafficState,
    build_route_feature_table,
    estimate_pretrip_speed_kph,
)
from src.synthetic_networks import (  # noqa: E402
    SyntheticNetworkSpec,
    generate_synthetic_network,
)


CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "topology_uncertainty_qaoa_interaction.json"
)
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "research"
    / "step9_2_topology_uncertainty_qaoa"
)
CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"
TRIAL_DIRECTORY = CHECKPOINT_DIRECTORY / "trials"
FAILURE_DIRECTORY = CHECKPOINT_DIRECTORY / "failures"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(
        _jsonable(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _source_path(relative_path: object) -> Path:
    path = (REPOSITORY_ROOT / str(relative_path)).resolve()
    if path != REPOSITORY_ROOT.resolve() and REPOSITORY_ROOT.resolve() not in path.parents:
        raise ValueError("source path must remain inside the repository")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _verify_inherited_sources(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for role, specification in dict(config["inherited_sources"]).items():
        item = dict(specification)
        path = _source_path(item["path"])
        actual = file_sha256(path)
        expected = str(item["sha256"])
        rows.append(
            {
                "source_role": role,
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "verified": actual == expected,
            }
        )
    table = pd.DataFrame(rows).sort_values("source_role").reset_index(drop=True)
    if not table["verified"].all():
        raise RuntimeError(
            "Step 9.2 inherited source hash verification failed:\n"
            + table.loc[~table["verified"]].to_string(index=False)
        )
    return table


def _initialize_output(config: dict[str, Any]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TRIAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FAILURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    effective_path = OUTPUT_DIRECTORY / "effective_config.json"
    if effective_path.exists():
        existing = json.loads(effective_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "existing Step 9.2 checkpoints were prepared with a different config"
            )
    else:
        _atomic_write_json(effective_path, config)


def _traffic_state(
    route: CandidateRoute, *, seed: int, parameters: dict[str, Any]
) -> RouteTrafficState:
    rng = np.random.default_rng(seed)
    congestion = float(
        rng.uniform(
            float(parameters["congestion_low"]),
            float(parameters["congestion_high"]),
        )
    )
    rain = int(rng.random() < float(parameters["rain_probability"]))
    incident = int(rng.random() < float(parameters["incident_probability"]))
    historical_delay = float(
        rng.gamma(
            shape=float(parameters["historical_delay_gamma_shape"]),
            scale=float(parameters["historical_delay_gamma_scale"]),
        )
    )
    free_flow_speed = 60.0 * route.distance_km / route.base_travel_time_min
    speed = estimate_pretrip_speed_kph(
        route.distance_km,
        route.base_travel_time_min,
        congestion,
        rain,
        incident,
        sensor_adjustment_kph=float(
            rng.normal(
                0.0,
                float(parameters["sensor_noise_fraction"]) * free_flow_speed,
            )
        ),
    )
    return RouteTrafficState(
        hour_of_day=int(parameters["hour_of_day"]),
        day_of_week=int(parameters["day_of_week"]),
        congestion=congestion,
        average_speed_kph=speed,
        rain_indicator=rain,
        incident_indicator=incident,
        historical_delay_minutes=historical_delay,
    )


def _build_networks(
    primary: dict[str, Any],
) -> tuple[
    dict[str, nx.DiGraph],
    dict[str, list[CandidateRoute]],
    pd.DataFrame,
    pd.DataFrame,
]:
    networks: dict[str, nx.DiGraph] = {}
    routes_by_network: dict[str, list[CandidateRoute]] = {}
    registry_rows: list[dict[str, object]] = []
    characteristic_rows: list[dict[str, object]] = []
    maximum_k = max(map(int, primary["candidate_counts"]))
    for topology_index, (topology, raw_spec) in enumerate(
        dict(primary["topologies"]).items()
    ):
        specification = dict(raw_spec)
        for seed_id in map(int, primary["matched_seed_ids"]):
            graph_seed = (
                int(primary["graph_seed_base"])
                + int(specification["seed_offset"])
                + seed_id
            )
            network_id = f"{topology}_s{seed_id:02d}"
            graph = generate_synthetic_network(
                SyntheticNetworkSpec(
                    topology=topology,
                    seed=graph_seed,
                    parameters=dict(specification["parameters"]),
                )
            )
            multiplier = float(primary["base_time_multiplier"])
            for _, _, attributes in graph.edges(data=True):
                attributes["base_travel_time_min"] = (
                    float(attributes["base_travel_time_min"]) * multiplier
                )
                if attributes.get("speed_kph") is not None:
                    attributes["speed_kph"] = float(attributes["speed_kph"]) / multiplier
            graph.graph.update(
                network_id=network_id,
                topology_family=topology,
                replicate=seed_id,
                graph_seed=graph_seed,
                topology_index=topology_index,
            )
            validate_network(graph)
            routes = generate_candidate_routes(
                graph,
                graph.graph["origin"],
                graph.graph["destination"],
                maximum_k,
                require_k=True,
            )
            networks[network_id] = graph
            routes_by_network[network_id] = routes
            registry_rows.append(
                {
                    "network_id": network_id,
                    "topology": topology,
                    "topology_index": topology_index,
                    "matched_seed_id": seed_id,
                    "graph_seed": graph_seed,
                    "generator_parameters_json": json.dumps(
                        dict(specification["parameters"]), sort_keys=True
                    ),
                    "origin": graph.graph["origin"],
                    "destination": graph.graph["destination"],
                    "node_count": graph.number_of_nodes(),
                    "directed_edge_count": graph.number_of_edges(),
                    "K8_complete": len(routes) == maximum_k,
                }
            )

            record = characterize_network(
                graph, route_k=maximum_k, require_route_k=True
            ).as_record()
            degree_distribution = record.pop("intersection_degree_distribution")
            undirected = graph.to_undirected()
            degrees = np.asarray([degree for _, degree in undirected.degree()], dtype=float)
            record.update(
                {
                    "network_id": network_id,
                    "topology": topology,
                    "topology_index": topology_index,
                    "matched_seed_id": seed_id,
                    "graph_seed": graph_seed,
                    "undirected_average_degree": float(np.mean(degrees)),
                    "undirected_degree_variance": float(np.var(degrees, ddof=0)),
                    "undirected_average_shortest_path_length": float(
                        nx.average_shortest_path_length(undirected)
                    ),
                    "undirected_average_clustering": float(
                        nx.average_clustering(undirected)
                    ),
                    "intersection_degree_distribution_json": json.dumps(
                        degree_distribution, sort_keys=True
                    ),
                }
            )
            characteristic_rows.append(record)
    registry = pd.DataFrame(registry_rows).sort_values(
        ["topology_index", "matched_seed_id"], kind="stable"
    )
    characteristics = pd.DataFrame(characteristic_rows).sort_values(
        ["topology_index", "matched_seed_id"], kind="stable"
    )
    expected_networks = len(primary["topologies"]) * len(primary["matched_seed_ids"])
    if len(registry) != expected_networks or not registry["K8_complete"].all():
        raise RuntimeError("primary network registry or K=8 completeness is invalid")
    return networks, routes_by_network, registry.reset_index(drop=True), characteristics.reset_index(drop=True)


def _residual_pools(
    primary: dict[str, Any],
    point_model: object,
    scale_model: object,
    observations: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    scenario = str(primary["base_calibration_scenario"])
    calibration = observations[
        (observations["dataset_group"] == str(primary["calibration_dataset_group"]))
        & (observations["scenario"] == scenario)
    ].copy()
    operational = observations[
        (
            observations["dataset_group"]
            == str(primary["operational_reference_dataset_group"])
        )
        & (observations["scenario"] == scenario)
    ].copy()
    if len(calibration) < 100 or len(operational) < 100:
        raise RuntimeError("normal-scenario residual pools are unexpectedly small")
    calibration_scale = np.asarray(scale_model.predict_scale(calibration), dtype=float)
    operational_scale = np.asarray(scale_model.predict_scale(operational), dtype=float)
    calibration_z = (
        calibration["travel_time_minutes"].to_numpy(dtype=float)
        - np.asarray(point_model.predict(calibration), dtype=float)
    ) / calibration_scale
    operational_z = (
        operational["travel_time_minutes"].to_numpy(dtype=float)
        - np.asarray(point_model.predict(operational), dtype=float)
    ) / operational_scale
    center = float(np.median(calibration_z))
    calibration_centered = calibration_z - center
    operational_centered = operational_z - center
    summary = pd.DataFrame(
        [
            {
                "pool": "calibration_normal_centered",
                "dataset_group": str(primary["calibration_dataset_group"]),
                "scenario": scenario,
                "n": len(calibration_centered),
                "calibration_center_removed": center,
                "mean": float(np.mean(calibration_centered)),
                "std": float(np.std(calibration_centered, ddof=1)),
                "median_absolute": float(np.median(np.abs(calibration_centered))),
                "q90_absolute": float(np.quantile(np.abs(calibration_centered), 0.90)),
            },
            {
                "pool": "operational_test_id_normal_centered",
                "dataset_group": str(primary["operational_reference_dataset_group"]),
                "scenario": scenario,
                "n": len(operational_centered),
                "calibration_center_removed": center,
                "mean": float(np.mean(operational_centered)),
                "std": float(np.std(operational_centered, ddof=1)),
                "median_absolute": float(np.median(np.abs(operational_centered))),
                "q90_absolute": float(np.quantile(np.abs(operational_centered), 0.90)),
            },
        ]
    )
    return calibration_centered, operational_centered, summary


def _problem_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prepare_problem_tables(
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    primary = dict(config["primary_design"])
    sources = dict(config["inherited_sources"])
    point_model = joblib.load(_source_path(dict(sources["point_model"])["path"]))
    scale_model = joblib.load(
        _source_path(dict(sources["residual_scale_model"])["path"])
    )
    observations = pd.read_csv(
        _source_path(dict(sources["traffic_observations"])["path"])
    )
    normalization_values = observations["distance_normalization_km"].unique()
    if len(normalization_values) != 1:
        raise RuntimeError("Step 9 frozen distance normalization is not unique")
    normalization = float(normalization_values[0])
    base_calibration_z, operational_z, residual_summary = _residual_pools(
        primary, point_model, scale_model, observations
    )
    networks, routes_by_network, registry, characteristics = _build_networks(primary)

    uncertainty_items = list(dict(primary["uncertainty_regimes"]).items())
    baseline_fraction = float(primary["baseline_prediction_noise_fraction"])
    draw_count = int(primary["uncertainty_monte_carlo_draws"])
    calibration_draws: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    uncertainty_rows: list[dict[str, object]] = []
    for uncertainty_index, (name, raw_regime) in enumerate(uncertainty_items):
        regime = dict(raw_regime)
        target_fraction = float(regime["target_prediction_noise_fraction"])
        if target_fraction < baseline_fraction:
            raise ValueError("target uncertainty cannot be below the baseline residual noise")
        additional_fraction = math.sqrt(
            max(0.0, target_fraction**2 - baseline_fraction**2)
        )
        rng = np.random.default_rng(
            int(primary["uncertainty_calibration_seed"]) + uncertainty_index
        )
        base_draw = rng.choice(base_calibration_z, size=draw_count, replace=True)
        extra_standard_normal = rng.normal(0.0, 1.0, size=draw_count)
        calibration_draws[name] = (
            base_draw,
            extra_standard_normal,
            additional_fraction,
        )
        uncertainty_rows.append(
            {
                "uncertainty": name,
                "uncertainty_index": uncertainty_index,
                "target_prediction_noise_fraction": target_fraction,
                "baseline_prediction_noise_fraction": baseline_fraction,
                "additional_noise_fraction_by_quadrature": additional_fraction,
                "calibration_draw_count": draw_count,
                "calibration_seed": int(primary["uncertainty_calibration_seed"])
                + uncertainty_index,
                "calibration_alpha": float(primary["conformal_alpha"]),
                "mechanism": (
                    "centered empirical normal-scenario standardized residual plus "
                    "independent Gaussian observation noise from the existing "
                    "base-time-scaled residual mechanism"
                ),
            }
        )
    uncertainty_calibration = pd.DataFrame(uncertainty_rows)

    route_rows: list[dict[str, object]] = []
    problem_rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []
    traffic_parameters = dict(primary["traffic_state"])
    risk_weights = dict(primary["risk_weights"])
    regime_index = {name: index for index, (name, _) in enumerate(uncertainty_items)}
    characteristic_lookup = characteristics.set_index("network_id")

    for network in registry.itertuples(index=False):
        graph = networks[str(network.network_id)]
        routes8 = routes_by_network[str(network.network_id)]
        state_seed_root = (
            int(primary["traffic_state_seed_base"])
            + int(network.topology_index) * 10_000
            + int(network.matched_seed_id) * 100
        )
        states = {
            route.route_id: _traffic_state(
                route,
                seed=state_seed_root + route.route_index,
                parameters=traffic_parameters,
            )
            for route in routes8
        }
        features8 = build_route_feature_table(
            graph,
            routes8,
            states,
            distance_normalization_km=normalization,
        )
        predictions8 = np.asarray(point_model.predict(features8), dtype=float)
        scales8 = np.asarray(scale_model.predict_scale(features8), dtype=float)
        if (
            not np.isfinite(predictions8).all()
            or not np.isfinite(scales8).all()
            or np.any(predictions8 <= 0.0)
            or np.any(scales8 <= 0.0)
        ):
            raise RuntimeError(f"invalid prediction or residual scale for {network.network_id}")
        outcome_rng = np.random.default_rng(
            int(primary["operational_outcome_seed_base"])
            + int(network.topology_index) * 10_000
            + int(network.matched_seed_id) * 100
        )
        base_outcome_z = outcome_rng.choice(operational_z, size=len(routes8), replace=True)
        additional_outcome_z = outcome_rng.normal(0.0, 1.0, size=len(routes8))
        # Freeze deadline strictness at the complete K=8 candidate pool.  Deriving
        # a separate deadline after truncating to K=4 would confound candidate-set
        # size with the deadline treatment and change the shared routes' B/C risks.
        network_deadline = float(
            np.quantile(
                predictions8,
                float(primary["deadline_prediction_quantile"]),
                method="linear",
            )
        )

        for k in map(int, primary["candidate_counts"]):
            routes = routes8[:k]
            features = features8.iloc[:k].reset_index(drop=True)
            predictions = predictions8[:k]
            scales = scales8[:k]
            base_times = features["base_route_time_min"].to_numpy(dtype=float)
            deadline = network_deadline
            diversity = summarize_route_diversity(routes)
            expected_route_index = int(np.argmin(predictions))
            for uncertainty_name, _ in uncertainty_items:
                base_draw, extra_draw, additional_fraction = calibration_draws[
                    uncertainty_name
                ]
                half_widths = np.asarray(
                    [
                        finite_sample_residual_quantile(
                            np.abs(
                                base_draw * scales[index]
                                + extra_draw
                                * additional_fraction
                                * base_times[index]
                            ),
                            float(primary["conformal_alpha"]),
                        )
                        for index in range(k)
                    ],
                    dtype=float,
                )
                actual_times = np.maximum(
                    0.05,
                    predictions
                    + base_outcome_z[:k] * scales
                    + additional_outcome_z[:k]
                    * additional_fraction
                    * base_times,
                )
                objective = controlled_objective_table(
                    predictions,
                    half_widths,
                    deadline_minutes=deadline,
                    lambda_deadline=float(risk_weights["lambda_deadline"]),
                    lambda_uncertainty=float(risk_weights["lambda_uncertainty"]),
                )
                problem_id = (
                    f"{network.topology}_s{int(network.matched_seed_id):02d}_"
                    f"{uncertainty_name}_K{k}"
                )
                risk_vector = objective[
                    "C_expected_plus_deadline_plus_uncertainty"
                ].to_numpy(dtype=float)
                exact_risk_index = int(np.argmin(risk_vector))
                actual_fastest_index = int(np.argmin(actual_times))
                problem_payload = {
                    "problem_id": problem_id,
                    "graph_seed": int(network.graph_seed),
                    "uncertainty": uncertainty_name,
                    "K": k,
                    "route_ids": objective["route_id"].tolist(),
                    "predictions": predictions.tolist(),
                    "half_widths": half_widths.tolist(),
                    "deadline": deadline,
                    "risk_vector": risk_vector.tolist(),
                    "actual_times": actual_times.tolist(),
                }
                problem_digest = _problem_hash(problem_payload)
                risk_sorted = np.sort(risk_vector)
                characteristic = characteristic_lookup.loc[str(network.network_id)]
                problem_rows.append(
                    {
                        "problem_id": problem_id,
                        "problem_sha256": problem_digest,
                        "network_id": str(network.network_id),
                        "topology": str(network.topology),
                        "topology_index": int(network.topology_index),
                        "matched_seed_id": int(network.matched_seed_id),
                        "graph_seed": int(network.graph_seed),
                        "uncertainty": uncertainty_name,
                        "uncertainty_index": int(regime_index[uncertainty_name]),
                        "K": k,
                        "deadline_minutes": deadline,
                        "risk_vector_json": json.dumps(risk_vector.tolist()),
                        "actual_route_times_json": json.dumps(actual_times.tolist()),
                        "expected_time_route_index": expected_route_index,
                        "exact_risk_route_index": exact_risk_index,
                        "actual_fastest_route_index": actual_fastest_index,
                        "exact_risk": float(risk_vector[exact_risk_index]),
                        "candidate_expected_time_spread": float(np.ptp(predictions)),
                        "candidate_risk_spread": float(np.ptp(risk_vector)),
                        "best_second_risk_gap": float(risk_sorted[1] - risk_sorted[0]),
                        "mean_candidate_uncertainty_half_width": float(np.mean(half_widths)),
                        "candidate_uncertainty_spread": float(np.ptp(half_widths)),
                        "route_mean_jaccard_similarity": float(
                            diversity["mean_jaccard_similarity"]
                        ),
                        "route_mean_jaccard_diversity": float(
                            diversity["mean_jaccard_diversity"]
                        ),
                        "node_count": int(characteristic["node_count"]),
                        "directed_edge_count": int(characteristic["directed_edge_count"]),
                        "directed_density": float(characteristic["directed_simple_density"]),
                        "average_degree": float(characteristic["undirected_average_degree"]),
                        "degree_variance": float(characteristic["undirected_degree_variance"]),
                        "average_shortest_path_length": float(
                            characteristic["undirected_average_shortest_path_length"]
                        ),
                        "clustering_coefficient": float(
                            characteristic["undirected_average_clustering"]
                        ),
                    }
                )
                variant_columns = {
                    "A_expected_time": "A_expected_time",
                    "B_expected_plus_deadline": "B_expected_plus_deadline",
                    "C_expected_plus_deadline_plus_uncertainty": (
                        "C_expected_plus_deadline_plus_uncertainty"
                    ),
                }
                selections: dict[str, int] = {}
                for variant, column in variant_columns.items():
                    selected = int(np.argmin(objective[column].to_numpy(dtype=float)))
                    selections[variant] = selected
                    choice_rows.append(
                        {
                            "problem_id": problem_id,
                            "network_id": str(network.network_id),
                            "topology": str(network.topology),
                            "matched_seed_id": int(network.matched_seed_id),
                            "uncertainty": uncertainty_name,
                            "K": k,
                            "objective_variant": variant,
                            "selected_route_index": selected,
                            "selected_route_id": f"x_{selected}",
                            "selected_score": float(objective.loc[selected, column]),
                            "D_full_dynamic_applicable": False,
                            "D_status": (
                                "not applicable in the static factorial; inherited Step 8 "
                                "dynamic runs evaluate reroute penalties"
                            ),
                        }
                    )
                for route_index in range(k):
                    feature_row = features.iloc[route_index]
                    route_rows.append(
                        {
                            "problem_id": problem_id,
                            "problem_sha256": problem_digest,
                            "network_id": str(network.network_id),
                            "topology": str(network.topology),
                            "matched_seed_id": int(network.matched_seed_id),
                            "graph_seed": int(network.graph_seed),
                            "uncertainty": uncertainty_name,
                            "K": k,
                            "route_index": route_index,
                            "route_id": f"x_{route_index}",
                            "ordered_nodes": feature_row["ordered_nodes"],
                            "distance_km": float(feature_row["distance_km"]),
                            "base_route_time_min": float(feature_row["base_route_time_min"]),
                            "edge_count": int(feature_row["edge_count"]),
                            "route_reliability": float(feature_row["route_reliability"]),
                            "prediction_minutes": float(predictions[route_index]),
                            "local_residual_scale_minutes": float(scales[route_index]),
                            "uncertainty_half_width_minutes": float(half_widths[route_index]),
                            "lower_bound_minutes": float(
                                max(0.0, predictions[route_index] - half_widths[route_index])
                            ),
                            "upper_bound_minutes": float(
                                predictions[route_index] + half_widths[route_index]
                            ),
                            "deadline_minutes": deadline,
                            "point_deadline_exceedance_minutes": float(
                                objective.loc[
                                    route_index, "point_deadline_exceedance_minutes"
                                ]
                            ),
                            "A_expected_time": float(
                                objective.loc[route_index, "A_expected_time"]
                            ),
                            "B_expected_plus_deadline": float(
                                objective.loc[route_index, "B_expected_plus_deadline"]
                            ),
                            "C_expected_plus_deadline_plus_uncertainty": float(
                                objective.loc[
                                    route_index,
                                    "C_expected_plus_deadline_plus_uncertainty",
                                ]
                            ),
                            "realized_travel_time_minutes": float(actual_times[route_index]),
                            "realized_deadline_violation": bool(
                                actual_times[route_index] > deadline
                            ),
                            "selected_by_A": route_index == selections["A_expected_time"],
                            "selected_by_B": route_index
                            == selections["B_expected_plus_deadline"],
                            "selected_by_C": route_index
                            == selections[
                                "C_expected_plus_deadline_plus_uncertainty"
                            ],
                            "actual_fastest": route_index == actual_fastest_index,
                        }
                    )

    problems = pd.DataFrame(problem_rows).sort_values(
        ["topology_index", "matched_seed_id", "uncertainty_index", "K"],
        kind="stable",
    )
    routes = pd.DataFrame(route_rows).sort_values(
        ["problem_id", "route_index"], kind="stable"
    )
    choices = pd.DataFrame(choice_rows).sort_values(
        ["problem_id", "objective_variant"], kind="stable"
    )
    if len(problems) != int(primary["expected_problem_count"]):
        raise RuntimeError(
            f"prepared {len(problems)} problems; expected {primary['expected_problem_count']}"
        )
    expected_route_rows = len(primary["topologies"]) * len(primary["matched_seed_ids"]) * len(uncertainty_items) * sum(map(int, primary["candidate_counts"]))
    if len(routes) != expected_route_rows or routes["problem_sha256"].isna().any():
        raise RuntimeError("prepared candidate-route table is incomplete")
    return (
        registry,
        characteristics,
        residual_summary,
        uncertainty_calibration,
        problems.reset_index(drop=True),
        routes.reset_index(drop=True),
        choices.reset_index(drop=True),
    )


def prepare(config: dict[str, Any]) -> pd.DataFrame:
    _initialize_output(config)
    source_verification = _verify_inherited_sources(config)
    problem_path = OUTPUT_DIRECTORY / "problem_summary.csv"
    preparation_manifest_path = OUTPUT_DIRECTORY / "preparation_manifest.json"
    preparation_signature = {
        "design_version": str(config["design_version"]),
        "deadline_reference": "full_K8_prediction_quantile_shared_by_K4_and_K8",
    }
    if problem_path.exists() and preparation_manifest_path.exists():
        existing_signature = json.loads(
            preparation_manifest_path.read_text(encoding="utf-8")
        )
        if existing_signature != preparation_signature:
            raise RuntimeError("existing preparation signature does not match config")
        problems = pd.read_csv(problem_path)
        expected = int(config["primary_design"]["expected_problem_count"])
        if len(problems) != expected or problems["problem_id"].duplicated().any():
            raise RuntimeError("existing prepared problem table is incomplete")
        return problems
    if any(TRIAL_DIRECTORY.glob("*.json")):
        raise RuntimeError(
            "cannot regenerate problem tables after primary trial checkpoints exist"
        )
    (
        registry,
        characteristics,
        residual_summary,
        uncertainty_calibration,
        problems,
        routes,
        choices,
    ) = _prepare_problem_tables(config)
    _atomic_write_csv(OUTPUT_DIRECTORY / "source_verification.csv", source_verification)
    _atomic_write_csv(OUTPUT_DIRECTORY / "network_registry.csv", registry)
    _atomic_write_csv(
        OUTPUT_DIRECTORY / "network_characteristics.csv", characteristics
    )
    _atomic_write_csv(
        OUTPUT_DIRECTORY / "base_residual_pool_summary.csv", residual_summary
    )
    _atomic_write_csv(
        OUTPUT_DIRECTORY / "uncertainty_calibration.csv", uncertainty_calibration
    )
    _atomic_write_csv(OUTPUT_DIRECTORY / "candidate_route_problems.csv", routes)
    _atomic_write_csv(OUTPUT_DIRECTORY / "objective_ablation_choices.csv", choices)
    _atomic_write_csv(problem_path, problems)
    _atomic_write_json(preparation_manifest_path, preparation_signature)
    print(
        f"Prepared {len(problems)} matched problems and {len(routes)} route rows.",
        flush=True,
    )
    return problems


def _optimization_seed(problem: pd.Series | dict[str, Any], depth: int, primary: dict[str, Any]) -> int:
    return (
        int(primary["optimization_seed_base"])
        + int(problem["topology_index"]) * 10_000
        + int(problem["matched_seed_id"]) * 100
        + int(problem["uncertainty_index"]) * 20
        + (0 if int(problem["K"]) == 4 else 10)
        + int(depth)
    )


def _trial_specifications(
    problems: pd.DataFrame, config: dict[str, Any], *, iterations: int | None = None
) -> list[dict[str, Any]]:
    primary = dict(config["primary_design"])
    specs: list[dict[str, Any]] = []
    for problem in problems.itertuples(index=False):
        record = problem._asdict()
        for depth in map(int, primary["depths"]):
            optimization_seed = _optimization_seed(record, depth, primary)
            for method in map(str, primary["mixers"]):
                specs.append(
                    {
                        "run_id": f"{record['problem_id']}_p{depth}_{method}",
                        "problem_id": record["problem_id"],
                        "problem_sha256": record["problem_sha256"],
                        "network_id": record["network_id"],
                        "topology": record["topology"],
                        "topology_index": int(record["topology_index"]),
                        "matched_seed_id": int(record["matched_seed_id"]),
                        "graph_seed": int(record["graph_seed"]),
                        "uncertainty": record["uncertainty"],
                        "uncertainty_index": int(record["uncertainty_index"]),
                        "K": int(record["K"]),
                        "depth": depth,
                        "method": method,
                        "optimization_seed": optimization_seed,
                        "iterations": int(
                            primary["iterations"] if iterations is None else iterations
                        ),
                        "step_size": float(primary["step_size"]),
                        "risk_vector": json.loads(record["risk_vector_json"]),
                        "actual_route_times": json.loads(
                            record["actual_route_times_json"]
                        ),
                        "deadline_minutes": float(record["deadline_minutes"]),
                        "expected_time_route_index": int(
                            record["expected_time_route_index"]
                        ),
                        "exact_risk_route_index": int(record["exact_risk_route_index"]),
                        "actual_fastest_route_index": int(
                            record["actual_fastest_route_index"]
                        ),
                    }
                )
    return specs


def _run_trial_worker(specification: dict[str, Any]) -> dict[str, Any]:
    risk_vector = np.asarray(specification["risk_vector"], dtype=float)
    actual_times = np.asarray(specification["actual_route_times"], dtype=float)
    k = int(specification["K"])
    depth = int(specification["depth"])
    method = str(specification["method"])
    qubo = build_risk_qubo(risk_vector)
    cost_hamiltonian, offset, _, _ = risk_qubo_to_ising(qubo)
    expectation = create_expectation_qnode(
        cost_hamiltonian,
        number_of_qubits=k,
        depth=depth,
        method=method,
    )
    probability = create_probability_qnode(
        cost_hamiltonian,
        number_of_qubits=k,
        depth=depth,
        method=method,
    )
    started = time.perf_counter()
    optimized = optimize_qaoa_angles(
        expectation,
        depth=depth,
        seed=int(specification["optimization_seed"]),
        iterations=int(specification["iterations"]),
        step_size=float(specification["step_size"]),
    )
    probabilities = np.asarray(probability(optimized.gammas, optimized.betas), dtype=float)
    diagnostics = summarize_qaoa_distribution(probabilities, qubo)
    resources = circuit_resource_record(probability, optimized.gammas, optimized.betas)
    elapsed = time.perf_counter() - started
    history = optimized.history.copy()
    history["expected_qubo_energy"] = history["hamiltonian_expectation"] + offset
    selected = int(diagnostics.metrics["selected_route_index"])
    expected_index = int(specification["expected_time_route_index"])
    risk_index = int(specification["exact_risk_route_index"])
    fastest_index = int(specification["actual_fastest_route_index"])
    selected_actual = float(actual_times[selected])
    deadline = float(specification["deadline_minutes"])
    gate_sizes = json.loads(str(resources["gate_sizes_json"]))
    metric = {
        key: specification[key]
        for key in (
            "run_id",
            "problem_id",
            "problem_sha256",
            "network_id",
            "topology",
            "topology_index",
            "matched_seed_id",
            "graph_seed",
            "uncertainty",
            "uncertainty_index",
            "K",
            "depth",
            "method",
            "optimization_seed",
            "iterations",
            "step_size",
            "deadline_minutes",
            "expected_time_route_index",
            "exact_risk_route_index",
            "actual_fastest_route_index",
        )
    }
    metric.update(diagnostics.metrics)
    metric.update(
        {
            "qubo_penalty": float(qubo.penalty),
            "hamiltonian_identity_offset": float(offset),
            "initial_expected_qubo_energy": float(
                history.iloc[0]["expected_qubo_energy"]
            ),
            "final_recorded_expected_qubo_energy": float(
                history.iloc[-1]["expected_qubo_energy"]
            ),
            "best_recorded_expected_qubo_energy": float(
                history["expected_qubo_energy"].min()
            ),
            "best_recorded_iteration": int(
                history["expected_qubo_energy"].idxmin()
            ),
            "last_five_energy_range": float(
                history["expected_qubo_energy"].tail(5).max()
                - history["expected_qubo_energy"].tail(5).min()
            ),
            "optimizer_steps": int(specification["iterations"]),
            "recorded_energy_evaluations": len(history),
            "gammas_json": json.dumps([float(value) for value in optimized.gammas]),
            "betas_json": json.dumps([float(value) for value in optimized.betas]),
            "wall_time_seconds": elapsed,
            "two_qubit_gate_count_logical": int(gate_sizes.get("2", 0)),
            "selected_realized_travel_time_minutes": selected_actual,
            "selected_realized_deadline_violation": selected_actual > deadline,
            "selected_realized_deadline_margin_minutes": deadline - selected_actual,
            "selected_realized_regret_vs_expected_time_minutes": selected_actual
            - float(actual_times[expected_index]),
            "selected_realized_regret_vs_exact_risk_minutes": selected_actual
            - float(actual_times[risk_index]),
            "selected_realized_regret_vs_actual_fastest_minutes": selected_actual
            - float(actual_times[fastest_index]),
            "selected_route_changed_from_expected_time": selected != expected_index,
            "selected_route_changed_from_exact_risk": selected != risk_index,
            **resources,
        }
    )
    distribution = diagnostics.feasible_distribution.copy()
    distribution.insert(0, "run_id", str(specification["run_id"]))
    distribution.insert(1, "problem_id", str(specification["problem_id"]))
    distribution.insert(2, "method", method)
    distribution.insert(3, "depth", depth)
    history.insert(0, "run_id", str(specification["run_id"]))
    history.insert(1, "problem_id", str(specification["problem_id"]))
    history.insert(2, "method", method)
    history.insert(3, "depth", depth)
    return {
        "completed_at_utc": _utc_now(),
        "metric": _jsonable(metric),
        "feasible_distribution": _jsonable(distribution.to_dict(orient="records")),
        "training_history": _jsonable(history.to_dict(orient="records")),
    }


def benchmark_runtime(config: dict[str, Any], problems: pd.DataFrame) -> pd.DataFrame:
    benchmark_path = OUTPUT_DIRECTORY / "runtime_benchmark.csv"
    if benchmark_path.exists():
        return pd.read_csv(benchmark_path)
    preflight = dict(config["runtime_preflight"])
    primary = dict(config["primary_design"])
    representative = problems[
        (problems["matched_seed_id"] == int(preflight["representative_seed_id"]))
        & (problems["uncertainty"] == str(preflight["representative_uncertainty"]))
        & (problems["topology"] == "grid")
    ]
    rows: list[dict[str, object]] = []
    for cell in preflight["cells"]:
        cell = dict(cell)
        problem = representative[representative["K"] == int(cell["K"])].iloc[0]
        all_specs = _trial_specifications(
            pd.DataFrame([problem]),
            config,
            iterations=int(preflight["benchmark_iterations"]),
        )
        spec = next(
            item
            for item in all_specs
            if item["depth"] == int(cell["depth"])
            and item["method"] == str(cell["mixer"])
        )
        result = _run_trial_worker(spec)
        seconds = float(result["metric"]["wall_time_seconds"])
        rows.append(
            {
                "K": int(cell["K"]),
                "depth": int(cell["depth"]),
                "method": str(cell["mixer"]),
                "benchmark_iterations": int(preflight["benchmark_iterations"]),
                "wall_time_seconds": seconds,
                "seconds_per_iteration": seconds
                / int(preflight["benchmark_iterations"]),
                "representative_problem_id": str(problem["problem_id"]),
            }
        )
        print(
            f"Runtime preflight K={cell['K']} p={cell['depth']} {cell['mixer']}: "
            f"{seconds:.2f}s",
            flush=True,
        )
    table = pd.DataFrame(rows)
    _atomic_write_csv(benchmark_path, table)
    replications_per_method_depth_k = (
        len(primary["topologies"])
        * len(primary["matched_seed_ids"])
        * len(primary["uncertainty_regimes"])
    )
    sequential_seconds = float(
        (
            table["seconds_per_iteration"]
            * int(primary["iterations"])
            * replications_per_method_depth_k
        ).sum()
    )
    _atomic_write_json(
        OUTPUT_DIRECTORY / "runtime_estimate.json",
        {
            "benchmark_created_at_utc": _utc_now(),
            "primary_trial_count": int(primary["expected_qaoa_trial_count"]),
            "primary_iterations": int(primary["iterations"]),
            "replications_per_K_depth_method_cell": replications_per_method_depth_k,
            "estimated_sequential_seconds": sequential_seconds,
            "estimated_sequential_hours": sequential_seconds / 3600.0,
            "default_workers": int(primary["default_workers"]),
            "parallel_estimate_is_not_reported_as_speedup": True,
            "timing_scope": "local simulator capacity planning only",
        },
    )
    return table


def _trial_path(run_id: str) -> Path:
    return TRIAL_DIRECTORY / f"{run_id}.json"


def run_primary(
    config: dict[str, Any], problems: pd.DataFrame, *, workers: int
) -> None:
    primary = dict(config["primary_design"])
    specifications = _trial_specifications(problems, config)
    if len(specifications) != int(primary["expected_qaoa_trial_count"]):
        raise RuntimeError("primary trial grid does not contain 1,800 cells")
    completed_ids = {path.stem for path in TRIAL_DIRECTORY.glob("*.json")}
    missing = [spec for spec in specifications if spec["run_id"] not in completed_ids]
    total = len(specifications)
    print(
        f"Step 9.2 primary: {len(completed_ids)}/{total} checkpointed; "
        f"{len(missing)} remaining; workers={workers}.",
        flush=True,
    )
    if not missing:
        return
    started = time.perf_counter()
    failures: list[str] = []
    completed_this_run = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_trial_worker, spec): spec for spec in missing}
        try:
            for future in as_completed(futures):
                specification = futures[future]
                run_id = str(specification["run_id"])
                metric: dict[str, Any] = {}
                try:
                    result = future.result()
                    metric = dict(result["metric"])
                    _atomic_write_json(_trial_path(run_id), result)
                    failure_path = FAILURE_DIRECTORY / f"{run_id}.json"
                    if failure_path.exists():
                        failure_path.unlink()
                    completed_this_run += 1
                except Exception as error:  # preserve every failed cell
                    failures.append(run_id)
                    _atomic_write_json(
                        FAILURE_DIRECTORY / f"{run_id}.json",
                        {
                            "run_id": run_id,
                            "failed_at_utc": _utc_now(),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                            "specification": specification,
                        },
                    )
                checkpoint_count = len(completed_ids) + completed_this_run
                elapsed = time.perf_counter() - started
                rate = completed_this_run / elapsed if elapsed > 0 else 0.0
                remaining_count = total - checkpoint_count
                _atomic_write_json(
                    CHECKPOINT_DIRECTORY / "progress.json",
                    {
                        "updated_at_utc": _utc_now(),
                        "total_trials": total,
                        "completed_trials": checkpoint_count,
                        "failed_trials_this_run": len(failures),
                        "workers": workers,
                        "elapsed_seconds_this_run": elapsed,
                        "trials_per_second_this_run": rate,
                        "estimated_remaining_seconds": (
                            remaining_count / rate if rate > 0 else None
                        ),
                    },
                )
                if completed_this_run == 1 or completed_this_run % 10 == 0:
                    print(
                        f"[{checkpoint_count}/{total}] {run_id} "
                        f"Pfeas={metric.get('feasibility_probability', float('nan')):.4f} "
                        f"Popt={metric.get('optimal_state_probability', float('nan')):.4f} "
                        f"elapsed={elapsed/60.0:.1f}m",
                        flush=True,
                    )
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    if failures:
        raise RuntimeError(
            f"{len(failures)} primary trials failed; failure records were preserved"
        )


def _load_trial_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(TRIAL_DIRECTORY.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _inherited_dynamic_link(config: dict[str, Any]) -> pd.DataFrame:
    sources = dict(config["inherited_sources"])
    runs = pd.read_csv(_source_path(dict(sources["dynamic_runs"])["path"]))
    decisions = pd.read_csv(
        _source_path(dict(sources["dynamic_qaoa_decisions"])["path"])
    )
    qaoa_runs = runs[runs["policy"] == "uncertainty_aware_qaoa"].copy()
    per_run = (
        decisions.groupby("run_id", sort=True)
        .agg(
            qaoa_decision_count=("qaoa_call_index", "size"),
            mean_dynamic_P_opt=("analytic_optimal_probability", "mean"),
            minimum_dynamic_P_opt=("analytic_optimal_probability", "min"),
            dynamic_exact_choice_rate=("matched_exact_risk", "mean"),
            mean_dynamic_P_feasible=("analytic_valid_probability", "mean"),
        )
        .reset_index()
    )
    return qaoa_runs.merge(per_run, on="run_id", how="left", validate="one_to_one")


def _validation_report(
    runs: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    dynamic_link: pd.DataFrame,
    total_wall_time: float,
) -> str:
    xy = runs[runs["method"] == "feasible_xy"]
    standard = runs[runs["method"] == "standard_x"]
    return f"""# Validation Report — Step 9.2 Feasibility Without Utility

## Status

- Primary factorial status: **PASS** for completeness and matched-input checks.
- New ideal-statevector QAOA trials: {len(runs)}.
- Matched routing problems: {runs['problem_id'].nunique()}.
- Topologies: {runs['topology'].nunique()}; uncertainty regimes: {runs['uncertainty'].nunique()}.
- K values: {sorted(runs['K'].unique().tolist())}; depths: {sorted(runs['depth'].unique().tolist())}.
- Mixers: standard Pauli-X and one-hot feasibility-preserving XY.
- Independent matched seed IDs per topology × uncertainty × K × depth × mixer cell: 10.
- Measured cumulative optimization wall time (sum across workers): {total_wall_time:.2f} seconds. This is simulator accounting, not a speed benchmark.

## Matched design checks

- Every problem/depth pair contains exactly one standard-X and one feasible-XY trial.
- The two mixers share the same graph, traffic state, candidate order, uncertainty state, deadline, risk vector, exact optimum, problem hash, and optimization seed.
- Deadlines are computed from point predictions once and held fixed across low/medium/high uncertainty.
- Objective A is expected time; B adds point-deadline exceedance; C adds uncertainty while preserving B's deadline term. D is excluded from the static factorial because a reroute state does not exist there; inherited Step 8 dynamic evidence retains the reroute term.

## Descriptive primary results

- Feasible-XY mean P_feasible: {xy['feasibility_probability'].mean():.6f}.
- Standard-X mean P_feasible: {standard['feasibility_probability'].mean():.6f}.
- Feasible-XY mean P_opt: {xy['optimal_state_probability'].mean():.6f}.
- Standard-X mean P_opt: {standard['optimal_state_probability'].mean():.6f}.
- Feasible-XY mean conditional feasible entropy: {xy['normalized_feasible_entropy'].mean():.6f}.
- Feasible-XY mean selected-route risk regret: {xy['risk_regret'].mean():.6f} minute-equivalent units.
- Inherited dynamic QAOA runs linked to decision-level P_opt: {len(dynamic_link)} (separate population; not pooled with the factorial).

The complete topology × uncertainty × K × depth × mixer aggregate is stored in
`qaoa_factorial_aggregate.csv`; paired X/XY differences are in
`qaoa_mixer_pairs.csv`. Statistical confidence intervals and association tests
are produced in Step 9.6 and must be consulted before inferential claims.

## Scientific limitations

1. All 1,800 primary trials are ideal statevector simulations with a fixed 15-step Adam budget. Convergence diagnostics are saved; deeper circuits may remain under-optimized.
2. The low/medium/high regimes use the existing base-time-scaled stochastic residual mechanism. They isolate predictive residual dispersion while leaving point predictions and deadlines fixed, but they remain controlled synthetic uncertainty.
3. The operational outcome attached to each primary problem is a matched one-decision route realization, not a full edge-by-edge rerouting trajectory. Full dynamic metrics come from the separately identified Step 8 population.
4. Exact optimality refers to the encoded candidate-risk objective, not the realized fastest route or a patient outcome.
5. K=4/8 remain classically trivial. Exact enumeration is the correctness reference and no quantum advantage, speedup, or operational superiority is claimed.

## Reproduction

Run `python experiments/run_step9_2_topology_uncertainty_qaoa.py all`. Each
completed trial is atomically stored under `checkpoints/trials/`; rerunning the
command resumes missing cells. Source/config/environment hashes and a hash of
every raw trial checkpoint are recorded in `manifest.json` and
`checkpoint_hashes.json`.
"""


def finalize(config: dict[str, Any], problems: pd.DataFrame) -> None:
    primary = dict(config["primary_design"])
    records = _load_trial_records()
    expected = int(primary["expected_qaoa_trial_count"])
    if len(records) != expected:
        raise RuntimeError(f"cannot finalize: {len(records)}/{expected} trials exist")
    runs = pd.DataFrame([record["metric"] for record in records])
    distributions = pd.DataFrame(
        [row for record in records for row in record["feasible_distribution"]]
    )
    histories = pd.DataFrame(
        [row for record in records for row in record["training_history"]]
    )
    if runs["run_id"].duplicated().any() or runs["problem_id"].nunique() != len(problems):
        raise RuntimeError("trial checkpoints contain duplicate or missing identifiers")
    expected_grid = (
        runs.groupby(
            ["topology", "uncertainty", "K", "depth", "method"], sort=True
        )
        .size()
        .rename("run_count")
        .reset_index()
    )
    if len(expected_grid) != 5 * 3 * 2 * 3 * 2 or not (
        expected_grid["run_count"] == 10
    ).all():
        raise RuntimeError("primary factorial cells are not complete at n=10")
    pair_checks = (
        runs.groupby(["problem_id", "depth"], sort=True)
        .agg(
            method_count=("method", "nunique"),
            row_count=("run_id", "size"),
            problem_hash_count=("problem_sha256", "nunique"),
            optimization_seed_count=("optimization_seed", "nunique"),
            exact_index_count=("exact_risk_route_index", "nunique"),
        )
        .reset_index()
    )
    if not (
        (pair_checks["method_count"] == 2)
        & (pair_checks["row_count"] == 2)
        & (pair_checks["problem_hash_count"] == 1)
        & (pair_checks["optimization_seed_count"] == 1)
        & (pair_checks["exact_index_count"] == 1)
    ).all():
        raise RuntimeError("matched X/XY input checks failed")
    xy = runs[runs["method"] == "feasible_xy"]
    if not np.allclose(xy["feasibility_probability"], 1.0, atol=1e-8):
        raise RuntimeError("ideal feasible-XY probability departed from one")

    metrics = [
        "feasibility_probability",
        "optimal_state_probability",
        "optimal_probability_given_feasible",
        "expected_hamiltonian_energy",
        "energy_regret",
        "risk_regret",
        "normalized_feasible_entropy",
        "top1_feasible_probability_conditional",
        "top2_feasible_probability_conditional",
        "top1_top2_probability_gap_conditional",
        "selected_realized_travel_time_minutes",
        "selected_realized_deadline_violation",
        "selected_realized_regret_vs_expected_time_minutes",
        "selected_realized_regret_vs_exact_risk_minutes",
        "wall_time_seconds",
    ]
    aggregate = (
        runs.groupby(
            ["topology", "uncertainty", "K", "depth", "method"], sort=True
        )[metrics]
        .agg(["count", "mean", "median", "std"])
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()

    pair_metric_names = [
        "feasibility_probability",
        "optimal_state_probability",
        "optimal_probability_given_feasible",
        "normalized_feasible_entropy",
        "risk_regret",
        "energy_regret",
        "selected_realized_deadline_violation",
        "selected_realized_regret_vs_expected_time_minutes",
    ]
    pair_rows: list[dict[str, object]] = []
    for (problem_id, depth), group in runs.groupby(["problem_id", "depth"], sort=True):
        indexed = group.set_index("method")
        row: dict[str, object] = {
            "problem_id": problem_id,
            "depth": int(depth),
            "topology": indexed.iloc[0]["topology"],
            "uncertainty": indexed.iloc[0]["uncertainty"],
            "K": int(indexed.iloc[0]["K"]),
            "matched_seed_id": int(indexed.iloc[0]["matched_seed_id"]),
            "problem_sha256": indexed.iloc[0]["problem_sha256"],
            "optimization_seed": int(indexed.iloc[0]["optimization_seed"]),
        }
        for metric in pair_metric_names:
            x_value = float(indexed.loc["standard_x", metric])
            xy_value = float(indexed.loc["feasible_xy", metric])
            row[f"X_{metric}"] = x_value
            row[f"XY_{metric}"] = xy_value
            row[f"XY_minus_X_{metric}"] = xy_value - x_value
        pair_rows.append(row)
    paired = pd.DataFrame(pair_rows)
    dynamic_link = _inherited_dynamic_link(config)

    _atomic_write_csv(OUTPUT_DIRECTORY / "qaoa_runs.csv", runs.sort_values("run_id"))
    _atomic_write_csv(
        OUTPUT_DIRECTORY / "feasible_route_distributions.csv",
        distributions.sort_values(["run_id", "route_index"]),
    )
    _atomic_write_csv(
        OUTPUT_DIRECTORY / "qaoa_training_history.csv",
        histories.sort_values(["run_id", "iteration"]),
    )
    _atomic_write_csv(
        OUTPUT_DIRECTORY / "qaoa_factorial_aggregate.csv", aggregate
    )
    _atomic_write_csv(OUTPUT_DIRECTORY / "qaoa_mixer_pairs.csv", paired)
    _atomic_write_csv(
        OUTPUT_DIRECTORY / "inherited_dynamic_qaoa_link.csv", dynamic_link
    )
    _atomic_write_json(
        OUTPUT_DIRECTORY / "validation_checks.json",
        {
            "all_checks_pass": True,
            "trial_count": len(runs),
            "problem_count": runs["problem_id"].nunique(),
            "factorial_cell_count": len(expected_grid),
            "all_factorial_cells_n10": bool((expected_grid["run_count"] == 10).all()),
            "matched_pair_count": len(pair_checks),
            "all_mixer_pairs_match_problem_and_seed": True,
            "all_xy_feasibility_one": True,
            "failed_checkpoint_count": len(list(FAILURE_DIRECTORY.glob("*.json"))),
        },
    )

    total_wall_time = float(runs["wall_time_seconds"].sum())
    report = _validation_report(runs, aggregate, paired, dynamic_link, total_wall_time)
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    checkpoint_hashes = {
        path.name: file_sha256(path) for path in sorted(TRIAL_DIRECTORY.glob("*.json"))
    }
    _atomic_write_json(
        OUTPUT_DIRECTORY / "checkpoint_hashes.json",
        {
            "trial_count": len(checkpoint_hashes),
            "trial_sha256": checkpoint_hashes,
        },
    )

    source_files = [
        CONFIG_PATH,
        Path(__file__),
        REPOSITORY_ROOT / "src" / "qaoa_diagnostics.py",
        REPOSITORY_ROOT / "src" / "controlled_objectives.py",
        REPOSITORY_ROOT / "src" / "feasibility_qaoa.py",
        REPOSITORY_ROOT / "src" / "risk_qubo.py",
        REPOSITORY_ROOT / "src" / "route_features.py",
        REPOSITORY_ROOT / "src" / "synthetic_networks.py",
        REPOSITORY_ROOT / "src" / "conformal.py",
        REPOSITORY_ROOT / "src" / "reproducibility.py",
        REPOSITORY_ROOT / "requirements.txt",
        REPOSITORY_ROOT / "requirements-research.txt",
    ]
    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "graph_seed_base": int(primary["graph_seed_base"]),
            "uncertainty_calibration_seed": int(
                primary["uncertainty_calibration_seed"]
            ),
            "traffic_state_seed_base": int(primary["traffic_state_seed_base"]),
            "operational_outcome_seed_base": int(
                primary["operational_outcome_seed_base"]
            ),
            "optimization_seed_base": int(primary["optimization_seed_base"]),
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "git_state_available": False,
            "primary_problem_count": runs["problem_id"].nunique(),
            "primary_qaoa_trial_count": len(runs),
            "matched_seed_ids": list(map(int, primary["matched_seed_ids"])),
            "cumulative_trial_wall_time_seconds": total_wall_time,
            "checkpoint_trial_count": len(checkpoint_hashes),
            "source_sha256": {
                path.relative_to(REPOSITORY_ROOT).as_posix(): file_sha256(path)
                for path in source_files
            },
            "inherited_source_sha256": {
                row.path: row.actual_sha256
                for row in pd.read_csv(
                    OUTPUT_DIRECTORY / "source_verification.csv"
                ).itertuples(index=False)
            },
            "artifact_sha256": {
                path.name: file_sha256(path)
                for path in sorted(OUTPUT_DIRECTORY.iterdir())
                if path.is_file()
                and path.name not in {"manifest.json", "INCOMPLETE.md"}
            },
            "claim_exclusions": [
                "quantum_advantage",
                "quantum_speedup",
                "qaoa_superiority_over_classical_routing",
                "hardware_behavior",
                "real_traffic_validation",
                "operational_deployment_readiness",
            ],
        }
    )
    _atomic_write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    print(
        f"Step 9.2 finalized: {len(runs)} trials, {len(problems)} problems, "
        f"{len(distributions)} feasible-state rows.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=("prepare", "benchmark", "run", "finalize", "all"),
    )
    parser.add_argument("--workers", type=int, default=None)
    arguments = parser.parse_args()
    config = _load_config()
    primary = dict(config["primary_design"])
    workers = (
        int(primary["default_workers"])
        if arguments.workers is None
        else int(arguments.workers)
    )
    if workers <= 0:
        raise ValueError("workers must be positive")
    problems = prepare(config)
    if arguments.mode == "prepare":
        return
    if arguments.mode in {"benchmark", "all"}:
        benchmark_runtime(config, problems)
        if arguments.mode == "benchmark":
            return
    if arguments.mode in {"run", "all"}:
        run_primary(config, problems, workers=workers)
        if arguments.mode == "run":
            return
    if arguments.mode in {"finalize", "all"}:
        finalize(config, problems)


if __name__ == "__main__":
    main()
