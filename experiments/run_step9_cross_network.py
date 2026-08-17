"""Run the primary Step 9 network-separated ID/OOD benchmark.

This runner is deliberately offline and synthetic.  Network partitions are
fixed and audited before any shifted traffic is generated.  Model selection,
scale fitting, and conformal calibration use mutually disjoint networks.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import sys

import joblib
import networkx as nx
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.candidate_routes import CandidateRoute, generate_candidate_routes
from src.conformal import (
    calibrate_locally_scaled_split_conformal,
    calibrate_relative_split_conformal,
    calibrate_split_conformal,
    evaluate_conformal,
    train_residual_scale_model,
)
from src.deadline_risk import RiskWeights, score_route_table
from src.feasibility_qaoa import (
    circuit_resource_record,
    create_expectation_qnode,
    create_probability_qnode,
    finite_shot_resample,
    one_hot_basis_indices,
    optimize_qaoa_angles,
    summarize_probabilities,
)
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
from src.synthetic_networks import (
    SUPPORTED_TOPOLOGIES,
    SyntheticNetworkSpec,
    generate_synthetic_network,
)
from src.traffic_data import (
    MODEL_FEATURE_COLUMNS,
    generate_historical_trip_observations,
)
from src.traffic_model import (
    FEATURE_COLUMNS,
    evaluate_traffic_model,
    select_best_by_validation_rmse,
    train_all_traffic_models,
)
from src.traffic_scenarios import get_scenario


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "cross_network_benchmark.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step9_cross_network"
BASE_PARTITIONS = ("train", "validation", "calibration", "test_id", "ood_topology")
EVALUATION_GROUPS = ("TEST-ID", "OOD-topology", "OOD-traffic", "OOD-combined")


def _load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _stable_edge_key(edge: tuple[object, object]) -> tuple[str, str]:
    return repr(edge[0]), repr(edge[1])


def _build_networks(
    config: dict[str, object],
) -> tuple[dict[str, nx.DiGraph], pd.DataFrame]:
    networks: dict[str, nx.DiGraph] = {}
    rows: list[dict[str, object]] = []
    candidate_k = int(config["candidate_k"])
    for entry in config["networks"]:
        values = dict(entry)
        network_id = str(values["network_id"])
        if network_id in networks:
            raise ValueError(f"duplicate network_id {network_id!r}")
        topology = str(values["topology"])
        partition = str(values["partition"])
        seed = int(values["seed"])
        parameters = dict(values["parameters"])
        multiplier = float(values["base_time_multiplier"])
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("base_time_multiplier must be positive and finite")
        graph = generate_synthetic_network(
            SyntheticNetworkSpec(topology=topology, seed=seed, parameters=parameters)
        )
        for _, _, attributes in graph.edges(data=True):
            attributes["base_travel_time_min"] = (
                float(attributes["base_travel_time_min"]) * multiplier
            )
        graph.graph.update(
            {
                "network_id": network_id,
                "benchmark_partition": partition,
                "base_time_multiplier": multiplier,
            }
        )
        validate_network(graph)
        routes = generate_candidate_routes(
            graph,
            graph.graph["origin"],
            graph.graph["destination"],
            candidate_k,
            require_k=True,
        )
        weak_components = list(nx.weakly_connected_components(graph))
        strong_components = list(nx.strongly_connected_components(graph))
        networks[network_id] = graph
        rows.append(
            {
                "network_id": network_id,
                "topology": topology,
                "base_partition": partition,
                "seed": seed,
                "base_time_multiplier": multiplier,
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "node_count": graph.number_of_nodes(),
                "directed_edge_count": graph.number_of_edges(),
                "origin": str(graph.graph["origin"]),
                "destination": str(graph.graph["destination"]),
                "candidate_k": candidate_k,
                "route_k_complete": len(routes) == candidate_k,
                "weak_component_count": len(weak_components),
                "strong_component_count": len(strong_components),
                "largest_weak_component_fraction": max(map(len, weak_components))
                / graph.number_of_nodes(),
                "largest_strong_component_fraction": max(map(len, strong_components))
                / graph.number_of_nodes(),
            }
        )
    return networks, pd.DataFrame(rows).sort_values(
        ["base_partition", "topology"], kind="stable"
    ).reset_index(drop=True)


def _validate_partition_design(
    config: dict[str, object], registry: pd.DataFrame
) -> pd.DataFrame:
    if set(registry["base_partition"]) != set(BASE_PARTITIONS):
        raise ValueError("network registry does not define the five required partitions")
    held_out = str(config["held_out_topology_family"])
    known_families = set(SUPPORTED_TOPOLOGIES) - {held_out}
    if len(known_families) != 4:
        raise ValueError("Step 9 requires exactly four known topology families")
    for partition in BASE_PARTITIONS[:-1]:
        group = registry[registry["base_partition"] == partition]
        if len(group) != 4 or set(group["topology"]) != known_families:
            raise ValueError(
                f"{partition} must contain one network from each known family"
            )
    ood = registry[registry["base_partition"] == "ood_topology"]
    if len(ood) != 4 or set(ood["topology"]) != {held_out}:
        raise ValueError("held-out topology must be entirely isolated in ood_topology")
    if registry["network_id"].duplicated().any():
        raise ValueError("network IDs must be globally unique")
    if not registry["route_k_complete"].all():
        raise ValueError("all benchmark networks must provide exactly K routes")

    rows: list[dict[str, object]] = []
    partition_sets = {
        partition: set(
            registry.loc[
                registry["base_partition"] == partition, "network_id"
            ].astype(str)
        )
        for partition in BASE_PARTITIONS
    }
    for first, second in combinations(BASE_PARTITIONS, 2):
        overlap = sorted(partition_sets[first] & partition_sets[second])
        rows.append(
            {
                "first_partition": first,
                "second_partition": second,
                "overlap_count": len(overlap),
                "overlapping_network_ids_json": json.dumps(overlap),
                "passes_zero_overlap": len(overlap) == 0,
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["passes_zero_overlap"].all():
        raise RuntimeError("network-level partition overlap detected")
    return audit


def _partition_networks(
    networks: dict[str, nx.DiGraph], registry: pd.DataFrame, partition: str
) -> tuple[dict[str, nx.DiGraph], dict[str, str]]:
    ids = registry.loc[
        registry["base_partition"] == partition, "network_id"
    ].astype(str)
    selected = {network_id: networks[network_id] for network_id in ids}
    labels = {
        network_id: str(networks[network_id].graph["topology"])
        for network_id in selected
    }
    return selected, labels


def _prefix_observation_ids(frame: pd.DataFrame, dataset_group: str) -> pd.DataFrame:
    result = frame.copy()
    result["observation_id"] = (
        dataset_group.replace("-", "_").upper()
        + "::"
        + result["observation_id"].astype(str)
    )
    result.insert(1, "dataset_group", dataset_group)
    return result


def _apply_frozen_normalization(
    frame: pd.DataFrame, frozen_maximum_km: float
) -> pd.DataFrame:
    result = frame.copy()
    result["distance_normalization_km"] = frozen_maximum_km
    result["normalized_distance"] = result["route_distance_km"] / frozen_maximum_km
    return result


def _generate_observation_partitions(
    config: dict[str, object],
    networks: dict[str, nx.DiGraph],
    registry: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    seeds = dict(config["seeds"])
    base_frames: dict[str, pd.DataFrame] = {}
    for partition in BASE_PARTITIONS:
        selected, labels = _partition_networks(networks, registry, partition)
        seed_key = f"traffic_{partition}"
        frame = generate_historical_trip_observations(
            selected,
            labels,
            scenarios=list(config["historical_scenarios"]),
            observations_per_network_per_scenario=int(
                config["observations_per_network_per_scenario"]
            ),
            random_seed=int(seeds[seed_key]),
        )
        base_frames[partition] = _prefix_observation_ids(frame, partition)

    training_maximum = float(base_frames["train"]["route_distance_km"].max())
    if not np.isfinite(training_maximum) or training_maximum <= 0.0:
        raise RuntimeError("training-set distance maximum is invalid")
    frames = {
        partition: _apply_frozen_normalization(frame, training_maximum)
        for partition, frame in base_frames.items()
    }

    # Distribution-shift data are deliberately created only after the network
    # partitions above have been fixed and the training normalization frozen.
    shift_design = {
        "ood_traffic": ("test_id", "OOD-traffic", "traffic_ood_shift"),
        "ood_combined": (
            "ood_topology",
            "OOD-combined",
            "traffic_ood_combined",
        ),
    }
    for key, (source_partition, label, seed_key) in shift_design.items():
        selected, labels = _partition_networks(networks, registry, source_partition)
        frame = generate_historical_trip_observations(
            selected,
            labels,
            scenarios=["shift"],
            observations_per_network_per_scenario=int(
                config["shift_observations_per_network"]
            ),
            random_seed=int(seeds[seed_key]),
        )
        frames[key] = _apply_frozen_normalization(
            _prefix_observation_ids(frame, label), training_maximum
        )

    all_observation_ids = pd.concat(
        [frame["observation_id"] for frame in frames.values()], ignore_index=True
    )
    if all_observation_ids.duplicated().any():
        raise RuntimeError("prefixed observation IDs are not globally unique")
    if not np.isclose(frames["train"]["normalized_distance"].max(), 1.0):
        raise RuntimeError("training-set normalization does not attain one")
    if any(
        not np.allclose(frame["distance_normalization_km"], training_maximum)
        for frame in frames.values()
    ):
        raise RuntimeError("a partition recomputed or changed the frozen normalization")

    summary_rows: list[dict[str, object]] = []
    for key, frame in frames.items():
        summary_rows.append(
            {
                "dataset_key": key,
                "dataset_group": str(frame["dataset_group"].iloc[0]),
                "row_count": len(frame),
                "network_count": frame["network_id"].nunique(),
                "network_ids_json": json.dumps(sorted(frame["network_id"].unique())),
                "topologies_json": json.dumps(sorted(frame["topology"].unique())),
                "scenario_count": frame["scenario"].nunique(),
                "scenarios_json": json.dumps(sorted(frame["scenario"].unique())),
                "maximum_raw_route_distance_km": float(
                    frame["route_distance_km"].max()
                ),
                "maximum_normalized_distance": float(
                    frame["normalized_distance"].max()
                ),
                "normalized_distance_above_one_count": int(
                    (frame["normalized_distance"] > 1.0).sum()
                ),
                "frozen_training_distance_maximum_km": training_maximum,
            }
        )
    normalization = pd.DataFrame(
        [
            {
                "normalization_feature": "normalized_distance",
                "fit_partition": "train",
                "raw_training_route_distance_maximum_km": training_maximum,
                "formula": "route_distance_km / raw_training_route_distance_maximum_km",
                "clipping_applied": False,
                "ood_values_above_one_allowed": True,
            }
        ]
    )
    return frames, pd.DataFrame(summary_rows), normalization


def _evaluation_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "TEST-ID": frames["test_id"],
        "OOD-topology": frames["ood_topology"],
        "OOD-traffic": frames["ood_traffic"],
        "OOD-combined": frames["ood_combined"],
    }


def _evaluate_point_model(
    model: object, evaluation_frames: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_tables: list[pd.DataFrame] = []
    network_tables: list[pd.DataFrame] = []
    topology_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    for group_name, frame in evaluation_frames.items():
        evaluation = evaluate_traffic_model(model, frame)
        for table, destination in (
            (evaluation.overall, overall_tables),
            (evaluation.by_network, network_tables),
            (evaluation.by_topology, topology_tables),
            (evaluation.predictions, prediction_tables),
        ):
            result = table.copy()
            result.insert(0, "evaluation_group", group_name)
            destination.append(result)
    return tuple(
        pd.concat(tables, ignore_index=True)
        for tables in (
            overall_tables,
            network_tables,
            topology_tables,
            prediction_tables,
        )
    )


def _fit_conformal_models(
    config: dict[str, object], best_model: object, scale_model: object, calibration: pd.DataFrame
) -> tuple[dict[tuple[str, float], object], pd.DataFrame]:
    models: dict[tuple[str, float], object] = {}
    rows: list[dict[str, object]] = []
    for alpha_value in config["conformal_alphas"]:
        alpha = float(alpha_value)
        calibrated = {
            "absolute": calibrate_split_conformal(
                best_model, calibration, alpha=alpha
            ),
            "relative": calibrate_relative_split_conformal(
                best_model,
                calibration,
                alpha=alpha,
                minimum_scale_minutes=float(
                    config["relative_minimum_scale_minutes"]
                ),
            ),
            "local": calibrate_locally_scaled_split_conformal(
                best_model, scale_model, calibration, alpha=alpha
            ),
        }
        for method, model in calibrated.items():
            models[(method, alpha)] = model
            rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "nominal_coverage": 1.0 - alpha,
                    "calibration_partition": "calibration",
                    "n_calibration": model.n_calibration,
                    "finite_sample_quantile_level": model.quantile_level,
                    "calibration_quantile": model.calibration_quantile,
                    "quantile_units": {
                        "absolute": "minutes",
                        "relative": "absolute_residual_over_prediction_scale",
                        "local": "absolute_residual_over_fitted_local_scale",
                    }[method],
                }
            )
    return models, pd.DataFrame(rows)


def _evaluate_conformal_models(
    models: dict[tuple[str, float], object],
    evaluation_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_tables: list[pd.DataFrame] = []
    network_tables: list[pd.DataFrame] = []
    topology_tables: list[pd.DataFrame] = []
    interval_tables: list[pd.DataFrame] = []
    for (method, alpha), model in sorted(models.items()):
        for group_name, frame in evaluation_frames.items():
            evaluation = evaluate_conformal(model, frame)
            for table, destination in (
                (evaluation.overall, overall_tables),
                (evaluation.by_network, network_tables),
                (evaluation.by_topology, topology_tables),
                (evaluation.intervals, interval_tables),
            ):
                result = table.copy()
                result.insert(0, "evaluation_group", group_name)
                result.insert(1, "method", method)
                result.insert(2, "configured_alpha", alpha)
                destination.append(result)
    return tuple(
        pd.concat(tables, ignore_index=True)
        for tables in (
            overall_tables,
            network_tables,
            topology_tables,
            interval_tables,
        )
    )


def _scenario_registry(config: dict[str, object]) -> pd.DataFrame:
    names = set(map(str, config["historical_scenarios"]))
    names.update(map(str, config["route_decision_scenarios"]))
    names.add("shift")
    rows: list[dict[str, object]] = []
    for name in sorted(names):
        row = asdict(get_scenario(name))
        row["hours_json"] = json.dumps(row.pop("hours"))
        rows.append(row)
    return pd.DataFrame(rows)


def _simulate_route_event(
    graph: nx.DiGraph,
    routes: list[CandidateRoute],
    scenario_name: str,
    rng: np.random.Generator,
) -> tuple[dict[str, RouteTrafficState], pd.DataFrame, dict[str, int]]:
    scenario = get_scenario(scenario_name)
    hour = int(rng.choice(scenario.hours))
    day = int(rng.integers(0, 7))
    rain = int(rng.random() < scenario.rain_probability)
    union_edges = sorted(
        {edge for route in routes for edge in route.edges}, key=_stable_edge_key
    )
    edge_congestion: dict[tuple[object, object], float] = {}
    edge_incident: dict[tuple[object, object], int] = {}
    edge_actual_time: dict[tuple[object, object], float] = {}
    for edge in union_edges:
        attributes = graph.edges[edge]
        base_time = float(attributes["base_travel_time_min"])
        reliability = float(attributes.get("reliability", 0.90))
        congestion = float(
            np.clip(rng.normal(scenario.congestion_mean, scenario.congestion_std), 0.0, 1.0)
        )
        incident = int(
            rng.random()
            < min(1.0, scenario.incident_probability + 0.10 * (1.0 - reliability))
        )
        multiplier = 1.0 + (
            scenario.congestion_delay_weight * congestion
            + scenario.rain_delay_fraction * rain
            + scenario.incident_delay_fraction * incident
            + scenario.shift_delay_fraction
        )
        noise = float(rng.normal(0.0, base_time * scenario.prediction_noise_fraction))
        edge_congestion[edge] = congestion
        edge_incident[edge] = incident
        edge_actual_time[edge] = max(0.01, base_time * multiplier + noise)

    states: dict[str, RouteTrafficState] = {}
    outcome_rows: list[dict[str, object]] = []
    for route in routes:
        base_times = np.asarray(
            [float(graph.edges[edge]["base_travel_time_min"]) for edge in route.edges]
        )
        congestion = float(
            np.average([edge_congestion[edge] for edge in route.edges], weights=base_times)
        )
        incident = int(any(edge_incident[edge] for edge in route.edges))
        historical_delay = float(rng.gamma(shape=1.5, scale=0.65))
        free_flow_speed = 60.0 * route.distance_km / route.base_travel_time_min
        speed = estimate_pretrip_speed_kph(
            route.distance_km,
            route.base_travel_time_min,
            congestion,
            rain,
            incident,
            sensor_adjustment_kph=float(rng.normal(0.0, 0.04 * free_flow_speed)),
        )
        states[route.route_id] = RouteTrafficState(
            hour_of_day=hour,
            day_of_week=day,
            congestion=congestion,
            average_speed_kph=speed,
            rain_indicator=rain,
            incident_indicator=incident,
            historical_delay_minutes=historical_delay,
        )
        actual_time = sum(edge_actual_time[edge] for edge in route.edges)
        actual_time += 0.15 * historical_delay
        outcome_rows.append(
            {
                "route_id": route.route_id,
                "actual_travel_time_minutes": float(actual_time),
            }
        )
    return states, pd.DataFrame(outcome_rows), {"hour_of_day": hour, "day_of_week": day, "rain_indicator": rain}


def _route_decision_evaluation(
    config: dict[str, object],
    networks: dict[str, nx.DiGraph],
    registry: pd.DataFrame,
    risk_conformal_model: object,
    frozen_normalization_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_design = {
        "TEST-ID": ("test_id", list(map(str, config["route_decision_scenarios"]))),
        "OOD-topology": (
            "ood_topology",
            list(map(str, config["route_decision_scenarios"])),
        ),
        "OOD-traffic": ("test_id", ["shift"]),
        "OOD-combined": ("ood_topology", ["shift"]),
    }
    events_per_network = int(config["route_decision_events_per_network"])
    if events_per_network <= 0:
        raise ValueError("route_decision_events_per_network must be positive")
    weights = RiskWeights(**dict(config["risk_weights"]))
    candidate_tables: list[pd.DataFrame] = []
    decision_rows: list[dict[str, object]] = []
    event_index = 0
    for evaluation_group in EVALUATION_GROUPS:
        partition, scenarios = group_design[evaluation_group]
        network_ids = sorted(
            registry.loc[
                registry["base_partition"] == partition, "network_id"
            ].astype(str)
        )
        for network_id in network_ids:
            graph = networks[network_id]
            routes = generate_candidate_routes(
                graph,
                graph.graph["origin"],
                graph.graph["destination"],
                int(config["candidate_k"]),
                require_k=True,
            )
            for local_event_index in range(events_per_network):
                scenario_name = scenarios[local_event_index % len(scenarios)]
                scenario_replicate = local_event_index // len(scenarios)
                event_index += 1
                event_id = f"route_event_{event_index:04d}"
                event_seed = int(dict(config["seeds"])["route_decisions"]) + event_index
                rng = np.random.default_rng(event_seed)
                states, outcomes, event_context = _simulate_route_event(
                    graph, routes, scenario_name, rng
                )
                features = build_route_feature_table(
                    graph,
                    routes,
                    states,
                    distance_normalization_km=frozen_normalization_km,
                )
                intervals = risk_conformal_model.predict(features)
                if "actual_travel_time_minutes" in intervals.columns:
                    raise RuntimeError("actual outcomes leaked into route selection inputs")
                deadline = float(config["deadline_free_flow_multiplier"]) * min(
                    route.base_travel_time_min for route in routes
                )
                scored = score_route_table(
                    intervals,
                    deadline,
                    weights=weights,
                    variant="D",
                )
                expected_choice = str(
                    intervals.sort_values(["prediction", "route_id"], kind="stable").iloc[0][
                        "route_id"
                    ]
                )
                risk_choice = str(
                    scored.sort_values(["risk_score", "route_id"], kind="stable").iloc[0][
                        "route_id"
                    ]
                )
                enriched = scored.merge(outcomes, on="route_id", validate="one_to_one")
                enriched.insert(0, "event_id", event_id)
                enriched.insert(1, "evaluation_group", evaluation_group)
                enriched.insert(2, "network_id", network_id)
                enriched.insert(3, "topology", str(graph.graph["topology"]))
                enriched.insert(4, "scenario", scenario_name)
                enriched.insert(5, "scenario_replicate", scenario_replicate)
                enriched.insert(6, "event_seed", event_seed)
                enriched["deadline_rule"] = "multiplier_times_fastest_free_flow_candidate"
                enriched["deadline_free_flow_multiplier"] = float(
                    config["deadline_free_flow_multiplier"]
                )
                enriched["selected_by_expected"] = enriched["route_id"] == expected_choice
                enriched["selected_by_calibrated_risk"] = enriched["route_id"] == risk_choice
                for key, value in event_context.items():
                    enriched[f"event_{key}"] = value
                candidate_tables.append(enriched)

                actual_best = float(outcomes["actual_travel_time_minutes"].min())
                for strategy, route_id in (
                    ("expected_time", expected_choice),
                    ("calibrated_risk", risk_choice),
                ):
                    chosen = enriched[enriched["route_id"] == route_id].iloc[0]
                    actual = float(chosen["actual_travel_time_minutes"])
                    decision_rows.append(
                        {
                            "event_id": event_id,
                            "evaluation_group": evaluation_group,
                            "network_id": network_id,
                            "topology": str(graph.graph["topology"]),
                            "scenario": scenario_name,
                            "scenario_replicate": scenario_replicate,
                            "event_seed": event_seed,
                            "selection_strategy": strategy,
                            "selected_route_id": route_id,
                            "expected_time_route_id": expected_choice,
                            "calibrated_risk_route_id": risk_choice,
                            "selection_changed_from_expected": risk_choice != expected_choice,
                            "prediction_minutes": float(chosen["prediction"]),
                            "upper_bound_minutes": float(chosen["upper_bound"]),
                            "modeled_risk_score": float(chosen["risk_score"]),
                            "deadline_minutes": deadline,
                            "deadline_rule": "multiplier_times_fastest_free_flow_candidate",
                            "actual_travel_time_minutes": actual,
                            "actual_deadline_violation": actual > deadline,
                            "actual_deadline_margin_minutes": deadline - actual,
                            "actual_best_candidate_time_minutes": actual_best,
                            "actual_regret_minutes": actual - actual_best,
                            "selected_actual_fastest": np.isclose(actual, actual_best),
                        }
                    )
    candidates = pd.concat(candidate_tables, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)
    summary = (
        decisions.groupby(["evaluation_group", "selection_strategy"], sort=True)
        .agg(
            event_count=("event_id", "size"),
            network_count=("network_id", "nunique"),
            actual_deadline_violation_rate=("actual_deadline_violation", "mean"),
            mean_actual_deadline_margin_minutes=(
                "actual_deadline_margin_minutes",
                "mean",
            ),
            mean_actual_regret_minutes=("actual_regret_minutes", "mean"),
            median_actual_regret_minutes=("actual_regret_minutes", "median"),
            actual_fastest_selection_rate=("selected_actual_fastest", "mean"),
            changed_from_expected_rate=("selection_changed_from_expected", "mean"),
        )
        .reset_index()
    )
    return candidates, decisions, summary


def _run_qaoa_subset(
    config: dict[str, object], route_candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qaoa_config = dict(config["qaoa"])
    representative_count = int(qaoa_config["representatives_per_group_topology"])
    event_registry = (
        route_candidates[
            ["event_id", "evaluation_group", "network_id", "topology", "scenario"]
        ]
        .drop_duplicates()
        .sort_values(
            ["evaluation_group", "topology", "network_id", "event_id"], kind="stable"
        )
        .groupby(["evaluation_group", "topology"], sort=True, as_index=False)
        .head(representative_count)
        .reset_index(drop=True)
    )
    metric_rows: list[dict[str, object]] = []
    history_tables: list[pd.DataFrame] = []
    run_index = 0
    for representative_index, event in enumerate(event_registry.itertuples(index=False)):
        group = route_candidates[route_candidates["event_id"] == event.event_id].copy()
        group["route_index"] = group["route_id"].str.removeprefix("x_").astype(int)
        group = group.sort_values("route_index", kind="stable")
        k = int(config["candidate_k"])
        if len(group) != k:
            raise RuntimeError(f"QAOA event {event.event_id} does not have exactly K routes")
        risk_vector = group["risk_score"].to_numpy(dtype=float)
        qubo = build_risk_qubo(risk_vector)
        cost_hamiltonian, offset, _, _ = risk_qubo_to_ising(qubo)
        optimal_index = int(np.argmin(risk_vector))
        event_optimization_seed = int(dict(config["seeds"])["qaoa"]) + representative_index
        for method in map(str, qaoa_config["methods"]):
            run_index += 1
            expectation_qnode = create_expectation_qnode(
                cost_hamiltonian,
                number_of_qubits=k,
                depth=int(qaoa_config["depth"]),
                method=method,
            )
            probability_qnode = create_probability_qnode(
                cost_hamiltonian,
                number_of_qubits=k,
                depth=int(qaoa_config["depth"]),
                method=method,
            )
            optimized = optimize_qaoa_angles(
                expectation_qnode,
                depth=int(qaoa_config["depth"]),
                seed=event_optimization_seed,
                iterations=int(qaoa_config["iterations"]),
                step_size=float(qaoa_config["step_size"]),
            )
            probabilities = np.asarray(
                probability_qnode(optimized.gammas, optimized.betas), dtype=float
            )
            analytic = summarize_probabilities(
                probabilities, qubo, optimal_route_index=optimal_index
            )
            shot_seed = event_optimization_seed + 100_000 + run_index
            shot_counts = finite_shot_resample(
                probabilities,
                number_of_qubits=k,
                shots=int(qaoa_config["shots"]),
                seed=shot_seed,
            )
            valid_bitstrings = {
                format(index, f"0{k}b") for index in one_hot_basis_indices(k)
            }
            optimal_bitstring = format(1 << (k - 1 - optimal_index), f"0{k}b")
            valid_shots = sum(
                count for bitstring, count in shot_counts.items() if bitstring in valid_bitstrings
            )
            optimal_shots = int(shot_counts.get(optimal_bitstring, 0))
            resources = circuit_resource_record(
                probability_qnode, optimized.gammas, optimized.betas
            )
            run_id = f"step9_qaoa_{run_index:03d}"
            metric_rows.append(
                {
                    "run_id": run_id,
                    "event_id": event.event_id,
                    "evaluation_group": event.evaluation_group,
                    "network_id": event.network_id,
                    "topology": event.topology,
                    "scenario": event.scenario,
                    "method": method,
                    "K": k,
                    "depth": int(qaoa_config["depth"]),
                    "iterations": int(qaoa_config["iterations"]),
                    "step_size": float(qaoa_config["step_size"]),
                    "optimization_seed": event_optimization_seed,
                    "shots": int(qaoa_config["shots"]),
                    "shot_seed": shot_seed,
                    "exact_optimal_route_index": optimal_index,
                    "exact_optimal_route_id": f"x_{optimal_index}",
                    "exact_optimal_risk": float(risk_vector[optimal_index]),
                    "initial_expected_qubo_energy": float(
                        optimized.history.iloc[0]["hamiltonian_expectation"] + offset
                    ),
                    "final_expected_qubo_energy": float(analytic["expected_qubo_energy"]),
                    "analytic_valid_probability": float(analytic["valid_probability"]),
                    "analytic_optimal_probability": float(analytic["optimal_probability"]),
                    "shot_valid_probability": valid_shots / int(qaoa_config["shots"]),
                    "shot_optimal_probability": optimal_shots / int(qaoa_config["shots"]),
                    "most_probable_valid_route_index": int(
                        analytic["most_probable_valid_route_index"]
                    ),
                    "most_probable_valid_is_optimal": int(
                        analytic["most_probable_valid_route_index"]
                    )
                    == optimal_index,
                    "gammas_json": json.dumps([float(value) for value in optimized.gammas]),
                    "betas_json": json.dumps([float(value) for value in optimized.betas]),
                    "shot_counts_json": json.dumps(shot_counts, sort_keys=True),
                    **resources,
                }
            )
            history = optimized.history.copy()
            history.insert(0, "run_id", run_id)
            history.insert(1, "event_id", event.event_id)
            history.insert(2, "evaluation_group", event.evaluation_group)
            history.insert(3, "topology", event.topology)
            history.insert(4, "method", method)
            history["expected_qubo_energy"] = history["hamiltonian_expectation"] + offset
            history_tables.append(history)
    return (
        event_registry,
        pd.DataFrame(metric_rows),
        pd.concat(history_tables, ignore_index=True),
    )


def _qaoa_summary(qaoa_runs: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        qaoa_runs.groupby([*group_columns, "method"], sort=True)
        .agg(
            run_count=("run_id", "size"),
            mean_analytic_valid_probability=("analytic_valid_probability", "mean"),
            mean_analytic_optimal_probability=("analytic_optimal_probability", "mean"),
            mean_shot_valid_probability=("shot_valid_probability", "mean"),
            mean_shot_optimal_probability=("shot_optimal_probability", "mean"),
            most_probable_valid_optimum_rate=("most_probable_valid_is_optimal", "mean"),
        )
        .reset_index()
    )


def _markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    rows = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for record in table.loc[:, columns].itertuples(index=False, name=None):
        values = []
        for value in record:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _validation_report(
    config: dict[str, object],
    registry: pd.DataFrame,
    observation_summary: pd.DataFrame,
    normalization: pd.DataFrame,
    selected_model_name: str,
    point_overall: pd.DataFrame,
    conformal_overall: pd.DataFrame,
    decision_summary: pd.DataFrame,
    qaoa_group_summary: pd.DataFrame,
    qaoa_runs: pd.DataFrame,
) -> str:
    normalization_value = float(
        normalization.iloc[0]["raw_training_route_distance_maximum_km"]
    )
    above_one = int(observation_summary["normalized_distance_above_one_count"].sum())
    return f"""# Validation Report — Step 9 Cross-Network/OOD Benchmark

## Status and separation

- Status: **PASS** for this bounded synthetic cross-network experiment.
- Base network partitions: 4 train, 4 validation, 4 calibration, 4 TEST-ID, and 4 held-out-topology networks.
- Known families: grid, radial, sparse_mixed, and dense_mixed, with one distinct network per family in each known-family partition.
- Held-out family: `{config['held_out_topology_family']}`; all four instances occur only in OOD-topology/OOD-combined evaluation.
- Every pair of train/validation/calibration/TEST-ID/OOD-topology base partitions has zero network-ID overlap. OOD-traffic intentionally shifts the TEST-ID networks, while OOD-combined intentionally shifts the held-out-topology networks.
- Observation IDs are partition-prefixed and globally unique.
- Shift scenario generation occurs only after the base partition audit and training-scale freeze.

## Frozen preprocessing

- The sole distance normalization is the raw training-set route-distance maximum: **{normalization_value:.6f} km**.
- The same divisor is applied without clipping to every validation, calibration, ID, topology-OOD, traffic-OOD, and combined-OOD row.
- {above_one} rows have normalized distance above 1.0. This is expected OOD behavior, not an error or a batch-derived rescaling.
- Network identity, topology, scenario, and observation IDs are metadata and are excluded from model features.

## Point prediction

- All three point models were trained only on train networks and ranked only on validation networks.
- Selected model: `{selected_model_name}`.

{_markdown_table(point_overall, ['evaluation_group', 'n_samples', 'mae', 'rmse', 'r2'])}

## Conformal prediction

- Residual scale fitting uses validation networks only.
- Absolute, relative, and locally scaled conformal scores use calibration networks only.
- The table below reports every requested method, alpha, and held-out group; no result was filtered for favorability.

{_markdown_table(conformal_overall.sort_values(['evaluation_group', 'method', 'configured_alpha']), ['evaluation_group', 'method', 'configured_alpha', 'nominal_coverage', 'empirical_coverage', 'mean_interval_width'])}

## Route decisions with simulated outcomes

- Every group has 4 networks × 4 pre-outcome events and exactly K={int(config['candidate_k'])} routes per event.
- Deadline rule fixed before prediction and outcome: `{float(config['deadline_free_flow_multiplier']):.2f} ×` the fastest candidate's free-flow time.
- Expected-time and calibrated-risk choices use no actual-outcome column. Regret and violations are computed only after selection against the shared event simulation.
- Risk selection uses `{config['risk_conformal_method']}` conformal intervals at alpha={float(config['risk_conformal_alpha']):.2f} and risk variant D.

{_markdown_table(decision_summary, ['evaluation_group', 'selection_strategy', 'event_count', 'actual_deadline_violation_rate', 'mean_actual_deadline_margin_minutes', 'mean_actual_regret_minutes', 'actual_fastest_selection_rate'])}

## Compact real QAOA subset

- Both standard-X and feasible-XY circuits were optimized and evaluated on an ideal statevector simulator, then resampled with {int(dict(config['qaoa'])['shots'])} deterministic shots.
- The reproducible subset contains one event per evaluation-group/topology cell: {len(qaoa_runs) // 2} QUBOs and {len(qaoa_runs)} circuit runs.
- The exact optimum is the modeled calibrated-risk QUBO optimum, not necessarily the actual fastest route.

{_markdown_table(qaoa_group_summary, ['evaluation_group', 'method', 'run_count', 'mean_analytic_valid_probability', 'mean_analytic_optimal_probability', 'mean_shot_valid_probability', 'mean_shot_optimal_probability', 'most_probable_valid_optimum_rate'])}

## Scientific limitations

1. All networks and traffic are synthetic; this is not evidence of real-world operational or medical effectiveness.
2. TEST-ID means unseen network instances from known synthetic families, not observations from a previously seen network.
3. OOD-topology isolates one held-out synthetic family; it does not establish universal topology generalization.
4. OOD-traffic uses one configured shift preset, and OOD-combined composes that preset with one held-out family.
5. Route outcomes are controlled simulations. The deadline rule is illustrative and is not a clinical or dispatch policy.
6. QAOA uses K=4, depth {int(dict(config['qaoa'])['depth'])}, a small reproducible subset, one seeded optimization per QUBO/method, and an ideal simulator. No hardware or speedup claim is supported.
7. Group/topology QAOA cells contain only the explicit run counts in the CSV; do not infer population-level advantages.

## Reproduction

Run `python experiments/run_step9_cross_network.py` from the repository root. The runner refuses to overwrite an existing result directory. `manifest.json` records the effective configuration, package versions, seeds, source hashes, and artifact hashes.
"""


def run() -> None:
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            f"refusing to overwrite existing Step 9 output: {OUTPUT_DIRECTORY}"
        )
    config = _load_config()
    networks, registry = _build_networks(config)
    overlap_audit = _validate_partition_design(config, registry)
    frames, observation_summary, normalization = _generate_observation_partitions(
        config, networks, registry
    )
    evaluation_frames = _evaluation_frames(frames)
    if tuple(MODEL_FEATURE_COLUMNS) != FEATURE_COLUMNS:
        raise RuntimeError("traffic generation and point-model feature contracts disagree")
    if {"network_id", "topology", "scenario", "observation_id"}.intersection(
        FEATURE_COLUMNS
    ):
        raise RuntimeError("identity or scenario metadata leaked into model features")

    seeds = dict(config["seeds"])
    fitted_models = train_all_traffic_models(
        frames["train"],
        random_state=int(seeds["model"]),
        random_forest_trees=int(config["random_forest_trees"]),
    )
    selection = select_best_by_validation_rmse(
        fitted_models, frames["validation"]
    )
    scale_model = train_residual_scale_model(
        selection.best_model,
        frames["validation"],
        random_state=int(seeds["residual_scale"]),
        n_estimators=int(config["residual_scale_trees"]),
        minimum_scale_minutes=float(config["residual_scale_minimum_minutes"]),
    )
    conformal_models, calibration_table = _fit_conformal_models(
        config, selection.best_model, scale_model, frames["calibration"]
    )
    point_overall, point_by_network, point_by_topology, point_predictions = (
        _evaluate_point_model(selection.best_model, evaluation_frames)
    )
    (
        conformal_overall,
        conformal_by_network,
        conformal_by_topology,
        conformal_intervals,
    ) = _evaluate_conformal_models(conformal_models, evaluation_frames)

    risk_key = (
        str(config["risk_conformal_method"]),
        float(config["risk_conformal_alpha"]),
    )
    if risk_key not in conformal_models:
        raise ValueError("risk conformal method/alpha is absent from calibrated models")
    frozen_normalization = float(
        normalization.iloc[0]["raw_training_route_distance_maximum_km"]
    )
    route_candidates, route_decisions, route_summary = _route_decision_evaluation(
        config,
        networks,
        registry,
        conformal_models[risk_key],
        frozen_normalization,
    )
    qaoa_subset, qaoa_runs, qaoa_history = _run_qaoa_subset(
        config, route_candidates
    )
    qaoa_group_summary = _qaoa_summary(qaoa_runs, ["evaluation_group"])
    qaoa_group_topology_summary = _qaoa_summary(
        qaoa_runs, ["evaluation_group", "topology"]
    )

    expected_rows = (
        4
        * len(config["historical_scenarios"])
        * int(config["observations_per_network_per_scenario"])
    )
    for key in BASE_PARTITIONS:
        if len(frames[key]) != expected_rows:
            raise RuntimeError(f"unexpected observation count in {key}")
    expected_shift_rows = 4 * int(config["shift_observations_per_network"])
    for key in ("ood_traffic", "ood_combined"):
        if len(frames[key]) != expected_shift_rows:
            raise RuntimeError(f"unexpected shifted observation count in {key}")
    if len(point_overall) != len(EVALUATION_GROUPS):
        raise RuntimeError("point-model evaluation omitted a held-out group")
    expected_conformal_rows = (
        len(EVALUATION_GROUPS) * 3 * len(config["conformal_alphas"])
    )
    if len(conformal_overall) != expected_conformal_rows:
        raise RuntimeError("conformal evaluation grid is incomplete")
    expected_events = len(EVALUATION_GROUPS) * 4 * int(
        config["route_decision_events_per_network"]
    )
    if route_decisions["event_id"].nunique() != expected_events:
        raise RuntimeError("route-decision event grid is incomplete")
    if len(route_candidates) != expected_events * int(config["candidate_k"]):
        raise RuntimeError("route candidate table is not exactly K-complete")
    expected_qaoa_qubos = qaoa_subset.groupby(
        ["evaluation_group", "topology"]
    ).size().sum()
    if len(qaoa_runs) != expected_qaoa_qubos * len(dict(config["qaoa"])["methods"]):
        raise RuntimeError("QAOA method grid is incomplete")
    feasible_valid = qaoa_runs.loc[
        qaoa_runs["method"] == "feasible_xy", "analytic_valid_probability"
    ]
    if not np.allclose(feasible_valid, 1.0, atol=1e-8):
        raise RuntimeError("feasible-XY QAOA left the one-hot subspace")
    numeric_tables = [
        point_overall,
        conformal_overall,
        route_summary,
        qaoa_runs.select_dtypes(include=[np.number]),
    ]
    if any(
        not np.isfinite(table.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
        for table in numeric_tables
    ):
        raise RuntimeError("a primary benchmark result contains non-finite values")

    scenario_registry = _scenario_registry(config)
    all_traffic = pd.concat(
        [frames[key] for key in (*BASE_PARTITIONS, "ood_traffic", "ood_combined")],
        ignore_index=True,
    )
    report = _validation_report(
        config,
        registry,
        observation_summary,
        normalization,
        selection.best_model_name,
        point_overall,
        conformal_overall,
        route_summary,
        qaoa_group_summary,
        qaoa_runs,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    tables = {
        "partition_registry.csv": registry,
        "network_overlap_audit.csv": overlap_audit,
        "observation_partition_summary.csv": observation_summary,
        "frozen_normalization.csv": normalization,
        "traffic_scenario_registry.csv": scenario_registry,
        "traffic_observations.csv": all_traffic,
        "model_selection.csv": selection.ranking,
        "point_metrics_overall.csv": point_overall,
        "point_metrics_by_network.csv": point_by_network,
        "point_metrics_by_topology.csv": point_by_topology,
        "point_predictions.csv": point_predictions,
        "conformal_calibration.csv": calibration_table,
        "conformal_metrics_overall.csv": conformal_overall,
        "conformal_metrics_by_network.csv": conformal_by_network,
        "conformal_metrics_by_topology.csv": conformal_by_topology,
        "conformal_intervals.csv": conformal_intervals,
        "route_decision_candidates.csv": route_candidates,
        "route_decisions.csv": route_decisions,
        "route_decision_summary.csv": route_summary,
        "qaoa_subset_registry.csv": qaoa_subset,
        "qaoa_runs.csv": qaoa_runs,
        "qaoa_training_history.csv": qaoa_history,
        "qaoa_summary_by_group.csv": qaoa_group_summary,
        "qaoa_summary_by_group_topology.csv": qaoa_group_topology_summary,
    }
    for filename, table in tables.items():
        table.to_csv(OUTPUT_DIRECTORY / filename, index=False)
    joblib.dump(selection.best_model, OUTPUT_DIRECTORY / "best_model.joblib")
    joblib.dump(scale_model, OUTPUT_DIRECTORY / "residual_scale_model.joblib")
    joblib.dump(conformal_models, OUTPUT_DIRECTORY / "conformal_models.joblib")
    write_json(OUTPUT_DIRECTORY / "effective_config.json", config)
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    source_files = [
        CONFIG_PATH,
        Path(__file__),
        REPOSITORY_ROOT / "src" / "candidate_routes.py",
        REPOSITORY_ROOT / "src" / "conformal.py",
        REPOSITORY_ROOT / "src" / "deadline_risk.py",
        REPOSITORY_ROOT / "src" / "feasibility_qaoa.py",
        REPOSITORY_ROOT / "src" / "network_schema.py",
        REPOSITORY_ROOT / "src" / "qubo.py",
        REPOSITORY_ROOT / "src" / "reproducibility.py",
        REPOSITORY_ROOT / "src" / "risk_qubo.py",
        REPOSITORY_ROOT / "src" / "route_features.py",
        REPOSITORY_ROOT / "src" / "synthetic_networks.py",
        REPOSITORY_ROOT / "src" / "traffic_data.py",
        REPOSITORY_ROOT / "src" / "traffic_model.py",
        REPOSITORY_ROOT / "src" / "traffic_scenarios.py",
        REPOSITORY_ROOT / "requirements.txt",
        REPOSITORY_ROOT / "requirements-research.txt",
    ]
    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={key: int(value) for key, value in seeds.items()},
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "evaluation_groups": list(EVALUATION_GROUPS),
            "network_partition_overlap_verified_zero": True,
            "observation_ids_globally_unique": True,
            "selected_model": selection.best_model_name,
            "frozen_training_distance_maximum_km": frozen_normalization,
            "normalization_clipped": False,
            "qaoa_run_count": len(qaoa_runs),
            "claims_excluded": [
                "universal_network_generalization",
                "real_traffic_effectiveness",
                "quantum_speedup",
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
        f"Step 9 complete: {len(registry)} disjoint base networks, "
        f"{len(all_traffic)} observations, {len(qaoa_runs)} QAOA runs."
    )
    print(selection.ranking.to_string(index=False))
    print(point_overall.to_string(index=False))
    print(conformal_overall.to_string(index=False))
    print(route_summary.to_string(index=False))
    print(qaoa_group_summary.to_string(index=False))
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run()
