"""Run strict leave-one-network-out evaluation on the Step 9 network corpus."""

from __future__ import annotations

import hashlib
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

from src.conformal import (
    calibrate_locally_scaled_split_conformal,
    calibrate_relative_split_conformal,
    calibrate_split_conformal,
    compute_coverage_metrics,
    evaluate_conformal,
    train_residual_scale_model,
)
from src.network_schema import validate_network
from src.reproducibility import (
    build_experiment_manifest,
    file_sha256,
    write_json,
)
from src.synthetic_networks import (
    SUPPORTED_TOPOLOGIES,
    SyntheticNetworkSpec,
    generate_synthetic_network,
)
from src.traffic_data import MODEL_FEATURE_COLUMNS, generate_historical_trip_observations
from src.traffic_model import (
    FEATURE_COLUMNS,
    compute_regression_metrics,
    evaluate_traffic_model,
    select_best_by_validation_rmse,
    train_all_traffic_models,
)


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "leave_one_network_out_benchmark.json"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "research" / "step9_1_leave_one_network_out"
)
FOLD_ROLES = ("train", "validation", "calibration", "heldout")


def _load_configs() -> tuple[dict[str, object], dict[str, object], Path]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    corpus_path = (REPOSITORY_ROOT / str(config["corpus_config_path"])).resolve()
    if REPOSITORY_ROOT.resolve() not in corpus_path.parents:
        raise ValueError("corpus_config_path must remain inside the repository")
    corpus_config = json.loads(corpus_path.read_text(encoding="utf-8"))
    return config, corpus_config, corpus_path


def _build_corpus_networks(
    corpus_config: dict[str, object],
) -> tuple[dict[str, nx.DiGraph], pd.DataFrame]:
    networks: dict[str, nx.DiGraph] = {}
    rows: list[dict[str, object]] = []
    for entry in corpus_config["networks"]:
        values = dict(entry)
        network_id = str(values["network_id"])
        topology = str(values["topology"])
        seed = int(values["seed"])
        multiplier = float(values["base_time_multiplier"])
        parameters = dict(values["parameters"])
        if network_id in networks:
            raise ValueError(f"duplicate corpus network {network_id!r}")
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
                "original_step9_partition": str(values["partition"]),
                "base_time_multiplier": multiplier,
            }
        )
        validate_network(graph)
        networks[network_id] = graph
        rows.append(
            {
                "network_id": network_id,
                "topology": topology,
                "seed": seed,
                "original_step9_partition": str(values["partition"]),
                "base_time_multiplier": multiplier,
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "node_count": graph.number_of_nodes(),
                "directed_edge_count": graph.number_of_edges(),
                "origin": str(graph.graph["origin"]),
                "destination": str(graph.graph["destination"]),
                "weak_component_count": nx.number_weakly_connected_components(graph),
                "strong_component_count": nx.number_strongly_connected_components(graph),
            }
        )
    registry = pd.DataFrame(rows).sort_values(
        ["topology", "seed", "network_id"], kind="stable"
    ).reset_index(drop=True)
    if len(registry) != 20:
        raise ValueError("Step 9.1 requires the complete 20-network corpus")
    family_counts = registry.groupby("topology").size()
    if set(registry["topology"]) != set(SUPPORTED_TOPOLOGIES) or not (
        family_counts == 4
    ).all():
        raise ValueError("corpus must contain four networks from each of five families")
    return networks, registry


def _generate_raw_corpus(
    config: dict[str, object],
    networks: dict[str, nx.DiGraph],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for network_index, row in enumerate(registry.itertuples(index=False)):
        network_id = str(row.network_id)
        frame = generate_historical_trip_observations(
            {network_id: networks[network_id]},
            {network_id: str(row.topology)},
            scenarios=list(config["historical_scenarios"]),
            observations_per_network_per_scenario=int(
                config["observations_per_network_per_scenario"]
            ),
            random_seed=int(config["traffic_seed_base"]) + network_index,
        )
        frame["observation_id"] = network_id + "::" + frame["observation_id"].astype(str)
        frame = frame.rename(
            columns={
                "distance_normalization_km": (
                    "unused_single_network_generator_normalization_km"
                ),
                "normalized_distance": (
                    "unused_single_network_generator_normalized_distance"
                ),
            }
        )
        tables.append(frame)
    corpus = pd.concat(tables, ignore_index=True)
    if corpus["observation_id"].duplicated().any():
        raise RuntimeError("raw corpus observation IDs are not globally unique")
    expected_per_network = len(config["historical_scenarios"]) * int(
        config["observations_per_network_per_scenario"]
    )
    if not (corpus.groupby("network_id").size() == expected_per_network).all():
        raise RuntimeError("raw corpus row counts differ across networks")
    return corpus


def _rotate(values: list[str], offset: int) -> list[str]:
    position = offset % len(values)
    return values[position:] + values[:position]


def _string_set_sha256(values: set[str]) -> str:
    payload = json.dumps(sorted(values), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_fold_splits(
    config: dict[str, object], registry: pd.DataFrame
) -> pd.DataFrame:
    family_order = list(SUPPORTED_TOPOLOGIES)
    family_members = {
        family: registry.loc[registry["topology"] == family]
        .sort_values(["seed", "network_id"], kind="stable")["network_id"]
        .astype(str)
        .tolist()
        for family in family_order
    }
    heldout_order = [
        network_id
        for family in family_order
        for network_id in family_members[family]
    ]
    registry_lookup = registry.set_index("network_id")
    rows: list[dict[str, object]] = []
    rotation_offset = int(config["split_rotation_offset"])
    for fold_index, heldout_id in enumerate(heldout_order):
        fold_id = f"lono_{fold_index:02d}_{heldout_id}"
        assignments: dict[str, str] = {heldout_id: "heldout"}
        for family_index, family in enumerate(family_order):
            ordered = _rotate(
                family_members[family],
                rotation_offset + fold_index + family_index,
            )
            eligible = [network_id for network_id in ordered if network_id != heldout_id]
            if len(eligible) < 2:
                raise RuntimeError("not enough family peers for validation and calibration")
            assignments[eligible[0]] = "validation"
            assignments[eligible[1]] = "calibration"
            for network_id in eligible[2:]:
                assignments[network_id] = "train"
        if set(assignments) != set(registry["network_id"].astype(str)):
            raise RuntimeError(f"fold {fold_id} does not assign every corpus network")
        for network_id, role in assignments.items():
            metadata = registry_lookup.loc[network_id]
            rows.append(
                {
                    "fold_index": fold_index,
                    "fold_id": fold_id,
                    "network_id": network_id,
                    "topology": str(metadata["topology"]),
                    "fold_role": role,
                    "heldout_network_id": heldout_id,
                    "heldout_topology": str(registry_lookup.loc[heldout_id, "topology"]),
                    "original_step9_partition": str(
                        metadata["original_step9_partition"]
                    ),
                    "original_partition_used_for_assignment": False,
                }
            )
    splits = pd.DataFrame(rows).sort_values(
        ["fold_index", "fold_role", "topology", "network_id"], kind="stable"
    ).reset_index(drop=True)
    return splits


def _audit_fold_splits(
    splits: pd.DataFrame, registry: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, object]] = []
    for fold_id, group in splits.groupby("fold_id", sort=True):
        role_sets = {
            role: set(group.loc[group["fold_role"] == role, "network_id"].astype(str))
            for role in FOLD_ROLES
        }
        pairwise_overlap = sum(
            len(role_sets[first] & role_sets[second])
            for first, second in (
                ("train", "validation"),
                ("train", "calibration"),
                ("train", "heldout"),
                ("validation", "calibration"),
                ("validation", "heldout"),
                ("calibration", "heldout"),
            )
        )
        validation_family_counts = group.loc[
            group["fold_role"] == "validation"
        ].groupby("topology").size()
        calibration_family_counts = group.loc[
            group["fold_role"] == "calibration"
        ].groupby("topology").size()
        heldout_row = group[group["fold_role"] == "heldout"].iloc[0]
        fit_networks = (
            role_sets["train"] | role_sets["validation"] | role_sets["calibration"]
        )
        exclusion_payload = {
            "fit_network_ids": sorted(fit_networks),
            "heldout_network_ids": sorted(role_sets["heldout"]),
            "overlap": sorted(fit_networks & role_sets["heldout"]),
        }
        audit_rows.append(
            {
                "fold_id": fold_id,
                "heldout_network_id": heldout_row["network_id"],
                "heldout_topology": heldout_row["topology"],
                "train_network_count": len(role_sets["train"]),
                "validation_network_count": len(role_sets["validation"]),
                "calibration_network_count": len(role_sets["calibration"]),
                "heldout_network_count": len(role_sets["heldout"]),
                "train_network_ids_json": json.dumps(sorted(role_sets["train"])),
                "validation_network_ids_json": json.dumps(
                    sorted(role_sets["validation"])
                ),
                "calibration_network_ids_json": json.dumps(
                    sorted(role_sets["calibration"])
                ),
                "heldout_network_ids_json": json.dumps(sorted(role_sets["heldout"])),
                "train_network_ids_sha256": _string_set_sha256(role_sets["train"]),
                "validation_network_ids_sha256": _string_set_sha256(
                    role_sets["validation"]
                ),
                "calibration_network_ids_sha256": _string_set_sha256(
                    role_sets["calibration"]
                ),
                "heldout_network_ids_sha256": _string_set_sha256(
                    role_sets["heldout"]
                ),
                "all_fit_network_ids_sha256": _string_set_sha256(fit_networks),
                "heldout_exclusion_evidence_sha256": hashlib.sha256(
                    json.dumps(
                        exclusion_payload, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "pairwise_network_overlap_count": pairwise_overlap,
                "validation_has_one_per_family": bool(
                    len(validation_family_counts) == 5
                    and (validation_family_counts == 1).all()
                ),
                "calibration_has_one_per_family": bool(
                    len(calibration_family_counts) == 5
                    and (calibration_family_counts == 1).all()
                ),
                "all_20_networks_assigned_once": len(group) == 20
                and group["network_id"].nunique() == 20,
                "normalization_fit_role": "train",
                "point_model_fit_role": "train",
                "point_model_selection_role": "validation",
                "residual_scale_fit_role": "validation",
                "conformal_quantile_fit_role": "calibration",
                "evaluation_role": "heldout",
                "alpha_source": "fixed_config_before_folds",
                "clipping_applied": False,
            }
        )
    audit = pd.DataFrame(audit_rows)
    required = (
        (audit["train_network_count"] == 9)
        & (audit["validation_network_count"] == 5)
        & (audit["calibration_network_count"] == 5)
        & (audit["heldout_network_count"] == 1)
        & (audit["pairwise_network_overlap_count"] == 0)
        & audit["validation_has_one_per_family"]
        & audit["calibration_has_one_per_family"]
        & audit["all_20_networks_assigned_once"]
    )
    if not required.all():
        raise RuntimeError("one or more LONO folds failed the network split audit")

    role_counts = (
        splits.groupby(["network_id", "topology", "fold_role"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for role in FOLD_ROLES:
        if role not in role_counts:
            role_counts[role] = 0
    role_counts = role_counts[
        ["network_id", "topology", "train", "validation", "calibration", "heldout"]
    ]
    role_counts["all_roles_observed"] = (
        role_counts[["train", "validation", "calibration", "heldout"]] > 0
    ).all(axis=1)
    if not (role_counts["heldout"] == 1).all() or not role_counts[
        "all_roles_observed"
    ].all():
        raise RuntimeError("deterministic rotation did not expose every network to every role")

    original_role = (
        splits.groupby(["original_step9_partition", "fold_role"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for role in FOLD_ROLES:
        if role not in original_role:
            original_role[role] = 0
    original_role["distinct_fold_roles"] = (
        original_role[["train", "validation", "calibration", "heldout"]] > 0
    ).sum(axis=1)
    original_role["original_label_reused_as_fixed_role"] = (
        original_role["distinct_fold_roles"] == 1
    )
    if original_role["original_label_reused_as_fixed_role"].any():
        raise RuntimeError("original Step 9 partition labels remained fixed in LONO")
    if set(registry["network_id"]) != set(role_counts["network_id"]):
        raise RuntimeError("role rotation audit omitted a corpus network")
    return audit, role_counts, original_role


def _normalized_frame(
    raw_corpus: pd.DataFrame, network_ids: set[str], training_maximum_km: float
) -> pd.DataFrame:
    frame = raw_corpus[raw_corpus["network_id"].isin(network_ids)].copy()
    frame["distance_normalization_km"] = training_maximum_km
    frame["normalized_distance"] = (
        frame["route_distance_km"] / training_maximum_km
    )
    return frame


def _run_folds(
    config: dict[str, object],
    raw_corpus: pd.DataFrame,
    splits: pd.DataFrame,
) -> dict[str, object]:
    alpha = float(config["conformal_alpha"])
    selection_tables: list[pd.DataFrame] = []
    point_rows: list[dict[str, object]] = []
    point_prediction_tables: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    conformal_rows: list[dict[str, object]] = []
    interval_tables: list[pd.DataFrame] = []
    normalization_rows: list[dict[str, object]] = []
    fitted_artifacts: dict[str, object] = {}
    audit_runtime_rows: list[dict[str, object]] = []

    for fold_index, (fold_id, fold_split) in enumerate(
        splits.groupby("fold_id", sort=True)
    ):
        role_networks = {
            role: set(
                fold_split.loc[fold_split["fold_role"] == role, "network_id"].astype(str)
            )
            for role in FOLD_ROLES
        }
        training_raw = raw_corpus[
            raw_corpus["network_id"].isin(role_networks["train"])
        ]
        training_maximum = float(training_raw["route_distance_km"].max())
        if not np.isfinite(training_maximum) or training_maximum <= 0.0:
            raise RuntimeError(f"invalid fold training maximum for {fold_id}")
        frames = {
            role: _normalized_frame(raw_corpus, network_ids, training_maximum)
            for role, network_ids in role_networks.items()
        }
        observation_sets = {
            role: set(frame["observation_id"].astype(str))
            for role, frame in frames.items()
        }
        observation_overlap = sum(
            len(observation_sets[first] & observation_sets[second])
            for first, second in (
                ("train", "validation"),
                ("train", "calibration"),
                ("train", "heldout"),
                ("validation", "calibration"),
                ("validation", "heldout"),
                ("calibration", "heldout"),
            )
        )
        if observation_overlap:
            raise RuntimeError(f"observation leakage detected in {fold_id}")
        fit_observations = (
            observation_sets["train"]
            | observation_sets["validation"]
            | observation_sets["calibration"]
        )
        observation_exclusion_payload = {
            "fit_observation_ids_sha256": _string_set_sha256(fit_observations),
            "heldout_observation_ids_sha256": _string_set_sha256(
                observation_sets["heldout"]
            ),
            "overlap_count": len(fit_observations & observation_sets["heldout"]),
        }
        if not np.isclose(frames["train"]["normalized_distance"].max(), 1.0):
            raise RuntimeError(f"training normalization does not attain one in {fold_id}")
        if any(
            not np.allclose(frame["distance_normalization_km"], training_maximum)
            for frame in frames.values()
        ):
            raise RuntimeError(f"fold-local normalization changed within {fold_id}")

        model_seed = int(config["model_seed_base"]) + fold_index
        scale_seed = int(config["residual_scale_seed_base"]) + fold_index
        models = train_all_traffic_models(
            frames["train"],
            random_state=model_seed,
            random_forest_trees=int(config["random_forest_trees"]),
        )
        selection = select_best_by_validation_rmse(models, frames["validation"])
        ranking = selection.ranking.copy()
        ranking.insert(0, "fold_id", fold_id)
        ranking.insert(1, "heldout_network_id", next(iter(role_networks["heldout"])))
        ranking.insert(2, "model_seed", model_seed)
        selection_tables.append(ranking)

        scale_model = train_residual_scale_model(
            selection.best_model,
            frames["validation"],
            random_state=scale_seed,
            n_estimators=int(config["residual_scale_trees"]),
            minimum_scale_minutes=float(config["residual_scale_minimum_minutes"]),
        )
        conformal_models = {
            "absolute": calibrate_split_conformal(
                selection.best_model, frames["calibration"], alpha=alpha
            ),
            "relative": calibrate_relative_split_conformal(
                selection.best_model,
                frames["calibration"],
                alpha=alpha,
                minimum_scale_minutes=float(config["relative_minimum_scale_minutes"]),
            ),
            "local": calibrate_locally_scaled_split_conformal(
                selection.best_model,
                scale_model,
                frames["calibration"],
                alpha=alpha,
            ),
        }
        heldout_id = next(iter(role_networks["heldout"]))
        heldout_topology = str(frames["heldout"]["topology"].iloc[0])
        point_evaluation = evaluate_traffic_model(
            selection.best_model, frames["heldout"]
        )
        point_metric = point_evaluation.overall.iloc[0]
        point_rows.append(
            {
                "fold_id": fold_id,
                "heldout_network_id": heldout_id,
                "heldout_topology": heldout_topology,
                "selected_model": selection.best_model_name,
                "n_samples": int(point_metric["n_samples"]),
                "mae": float(point_metric["mae"]),
                "rmse": float(point_metric["rmse"]),
                "r2": float(point_metric["r2"]),
            }
        )
        predictions = point_evaluation.predictions.copy()
        predictions.insert(0, "fold_id", fold_id)
        predictions.insert(1, "heldout_network_id", heldout_id)
        predictions.insert(2, "heldout_topology", heldout_topology)
        predictions.insert(3, "selected_model", selection.best_model_name)
        point_prediction_tables.append(predictions)

        for method, conformal_model in conformal_models.items():
            calibration_rows.append(
                {
                    "fold_id": fold_id,
                    "heldout_network_id": heldout_id,
                    "heldout_topology": heldout_topology,
                    "method": method,
                    "alpha": alpha,
                    "n_calibration": conformal_model.n_calibration,
                    "finite_sample_quantile_level": conformal_model.quantile_level,
                    "calibration_quantile": conformal_model.calibration_quantile,
                    "calibration_network_count": len(role_networks["calibration"]),
                    "calibration_role_only": True,
                }
            )
            evaluation = evaluate_conformal(conformal_model, frames["heldout"])
            metric = evaluation.overall.iloc[0]
            conformal_rows.append(
                {
                    "fold_id": fold_id,
                    "heldout_network_id": heldout_id,
                    "heldout_topology": heldout_topology,
                    "method": method,
                    "n_samples": int(metric["n_samples"]),
                    "alpha": alpha,
                    "nominal_coverage": 1.0 - alpha,
                    "empirical_coverage": float(metric["empirical_coverage"]),
                    "mean_interval_width": float(metric["mean_interval_width"]),
                    "median_interval_width": float(metric["median_interval_width"]),
                }
            )
            intervals = evaluation.intervals.copy()
            intervals.insert(0, "fold_id", fold_id)
            intervals.insert(1, "heldout_network_id", heldout_id)
            intervals.insert(2, "heldout_topology", heldout_topology)
            intervals.insert(3, "method", method)
            interval_tables.append(intervals)

        normalization_rows.append(
            {
                "fold_id": fold_id,
                "heldout_network_id": heldout_id,
                "heldout_topology": heldout_topology,
                "fit_role": "train",
                "training_network_count": len(role_networks["train"]),
                "training_raw_route_distance_maximum_km": training_maximum,
                "heldout_raw_route_distance_maximum_km": float(
                    frames["heldout"]["route_distance_km"].max()
                ),
                "heldout_maximum_normalized_distance": float(
                    frames["heldout"]["normalized_distance"].max()
                ),
                "heldout_above_one_count": int(
                    (frames["heldout"]["normalized_distance"] > 1.0).sum()
                ),
                "clipping_applied": False,
            }
        )
        audit_runtime_rows.append(
            {
                "fold_id": fold_id,
                "train_observation_count": len(frames["train"]),
                "validation_observation_count": len(frames["validation"]),
                "calibration_observation_count": len(frames["calibration"]),
                "heldout_observation_count": len(frames["heldout"]),
                "pairwise_observation_overlap_count": observation_overlap,
                "train_observation_ids_sha256": _string_set_sha256(
                    observation_sets["train"]
                ),
                "validation_observation_ids_sha256": _string_set_sha256(
                    observation_sets["validation"]
                ),
                "calibration_observation_ids_sha256": _string_set_sha256(
                    observation_sets["calibration"]
                ),
                "heldout_observation_ids_sha256": _string_set_sha256(
                    observation_sets["heldout"]
                ),
                "heldout_observation_exclusion_evidence_sha256": hashlib.sha256(
                    json.dumps(
                        observation_exclusion_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "heldout_network_absent_from_all_fit_roles": not any(
                    heldout_id in role_networks[role]
                    for role in ("train", "validation", "calibration")
                ),
                "normalization_uses_training_rows_only": True,
                "model_fit_uses_training_rows_only": True,
                "model_selection_uses_validation_rows_only": True,
                "scale_fit_uses_validation_rows_only": True,
                "conformal_fit_uses_calibration_rows_only": True,
                "heldout_used_for_evaluation_only": True,
            }
        )
        fitted_artifacts[fold_id] = {
            "heldout_network_id": heldout_id,
            "training_distance_maximum_km": training_maximum,
            "selected_model": selection.best_model,
            "residual_scale_model": scale_model,
            "conformal_models": conformal_models,
        }

    return {
        "model_selection": pd.concat(selection_tables, ignore_index=True),
        "point_by_fold": pd.DataFrame(point_rows),
        "point_predictions": pd.concat(point_prediction_tables, ignore_index=True),
        "conformal_calibration": pd.DataFrame(calibration_rows),
        "conformal_by_fold": pd.DataFrame(conformal_rows),
        "conformal_intervals": pd.concat(interval_tables, ignore_index=True),
        "normalization": pd.DataFrame(normalization_rows),
        "runtime_audit": pd.DataFrame(audit_runtime_rows),
        "models": fitted_artifacts,
    }


def _aggregate_point_metrics(
    point_by_fold: pd.DataFrame, point_predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = compute_regression_metrics(
        point_predictions["actual_travel_time_minutes"],
        point_predictions["predicted_travel_time_minutes"],
    )
    overall = pd.DataFrame(
        [
            {
                "fold_count": point_by_fold["fold_id"].nunique(),
                "heldout_network_count": point_by_fold["heldout_network_id"].nunique(),
                "n_pooled_samples": pooled.n_samples,
                "pooled_mae": pooled.mae,
                "pooled_rmse": pooled.rmse,
                "pooled_r2": pooled.r2,
                "macro_fold_mean_mae": float(point_by_fold["mae"].mean()),
                "macro_fold_std_mae": float(point_by_fold["mae"].std(ddof=1)),
                "macro_fold_mean_rmse": float(point_by_fold["rmse"].mean()),
                "macro_fold_std_rmse": float(point_by_fold["rmse"].std(ddof=1)),
                "macro_fold_mean_r2": float(point_by_fold["r2"].mean()),
                "macro_fold_std_r2": float(point_by_fold["r2"].std(ddof=1)),
            }
        ]
    )
    topology_rows: list[dict[str, object]] = []
    for topology, predictions in point_predictions.groupby(
        "heldout_topology", sort=True
    ):
        metrics = compute_regression_metrics(
            predictions["actual_travel_time_minutes"],
            predictions["predicted_travel_time_minutes"],
        )
        folds = point_by_fold[point_by_fold["heldout_topology"] == topology]
        topology_rows.append(
            {
                "topology": topology,
                "fold_count": folds["fold_id"].nunique(),
                "n_pooled_samples": metrics.n_samples,
                "pooled_mae": metrics.mae,
                "pooled_rmse": metrics.rmse,
                "pooled_r2": metrics.r2,
                "macro_fold_mean_mae": float(folds["mae"].mean()),
                "macro_fold_mean_rmse": float(folds["rmse"].mean()),
                "macro_fold_mean_r2": float(folds["r2"].mean()),
            }
        )
    return overall, pd.DataFrame(topology_rows)


def _aggregate_conformal_metrics(
    conformal_by_fold: pd.DataFrame, intervals: pd.DataFrame, alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    for method, group in intervals.groupby("method", sort=True):
        metrics = compute_coverage_metrics(
            group["travel_time_minutes"],
            group["lower_bound"],
            group["upper_bound"],
            alpha=alpha,
        )
        folds = conformal_by_fold[conformal_by_fold["method"] == method]
        overall_rows.append(
            {
                "method": method,
                **metrics.as_dict(),
                "fold_count": folds["fold_id"].nunique(),
                "macro_fold_mean_coverage": float(
                    folds["empirical_coverage"].mean()
                ),
                "macro_fold_std_coverage": float(
                    folds["empirical_coverage"].std(ddof=1)
                ),
                "macro_fold_mean_width": float(folds["mean_interval_width"].mean()),
                "macro_fold_std_width": float(
                    folds["mean_interval_width"].std(ddof=1)
                ),
            }
        )
        for topology, topology_group in group.groupby("heldout_topology", sort=True):
            topology_metrics = compute_coverage_metrics(
                topology_group["travel_time_minutes"],
                topology_group["lower_bound"],
                topology_group["upper_bound"],
                alpha=alpha,
            )
            topology_folds = folds[folds["heldout_topology"] == topology]
            topology_rows.append(
                {
                    "topology": topology,
                    "method": method,
                    **topology_metrics.as_dict(),
                    "fold_count": topology_folds["fold_id"].nunique(),
                    "macro_fold_mean_coverage": float(
                        topology_folds["empirical_coverage"].mean()
                    ),
                    "macro_fold_mean_width": float(
                        topology_folds["mean_interval_width"].mean()
                    ),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(topology_rows)


def _unfavorable_fold_table(
    point_by_fold: pd.DataFrame, conformal_by_fold: pd.DataFrame
) -> pd.DataFrame:
    coverage = conformal_by_fold.pivot(
        index=["fold_id", "heldout_network_id", "heldout_topology"],
        columns="method",
        values="empirical_coverage",
    ).reset_index()
    coverage = coverage.rename(
        columns={method: f"{method}_coverage" for method in ("absolute", "relative", "local")}
    )
    table = point_by_fold.merge(
        coverage,
        on=["fold_id", "heldout_network_id", "heldout_topology"],
        validate="one_to_one",
    )
    coverage_columns = [
        "absolute_coverage",
        "relative_coverage",
        "local_coverage",
    ]
    table["minimum_method_coverage"] = table[coverage_columns].min(axis=1)
    table["maximum_coverage_shortfall"] = (
        table["nominal_coverage"] - table["minimum_method_coverage"]
        if "nominal_coverage" in table
        else 0.90 - table["minimum_method_coverage"]
    )
    table["rmse_worst_rank"] = table["rmse"].rank(
        method="min", ascending=False
    ).astype(int)
    table["coverage_worst_rank"] = table["minimum_method_coverage"].rank(
        method="min", ascending=True
    ).astype(int)
    table["unfavorable_flag"] = (
        (table["rmse_worst_rank"] <= 5)
        | (table["coverage_worst_rank"] <= 5)
        | (table["minimum_method_coverage"] < 0.80)
    )
    return table.sort_values(
        ["unfavorable_flag", "minimum_method_coverage", "rmse"],
        ascending=[False, True, False],
        kind="stable",
    ).reset_index(drop=True)


def _markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    rows = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for record in table.loc[:, columns].itertuples(index=False, name=None):
        values: list[str] = []
        for value in record:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _validation_report(
    config: dict[str, object],
    split_audit: pd.DataFrame,
    role_by_network: pd.DataFrame,
    original_role_audit: pd.DataFrame,
    normalization: pd.DataFrame,
    point_by_fold: pd.DataFrame,
    point_aggregate: pd.DataFrame,
    point_by_topology: pd.DataFrame,
    conformal_by_fold: pd.DataFrame,
    conformal_aggregate: pd.DataFrame,
    conformal_by_topology: pd.DataFrame,
    unfavorable: pd.DataFrame,
) -> str:
    point_overall = point_aggregate.iloc[0]
    fold_summary = point_by_fold.merge(
        conformal_by_fold.pivot(
            index="fold_id", columns="method", values="empirical_coverage"
        ).reset_index(),
        on="fold_id",
        validate="one_to_one",
    )
    unfavorable_top = unfavorable.head(5)
    selected_counts = (
        point_by_fold.groupby("selected_model", sort=True).size().rename("fold_count").reset_index()
    )
    return f"""# Validation Report — Step 9.1 Leave-One-Network-Out

## Audit status

- Status: **PASS** for the mechanics of this bounded synthetic LONO experiment.
- Corpus: 20 networks, four from each of five topology families; every network is held out exactly once.
- Each fold contains 9 whole training networks, 5 whole validation networks, 5 whole calibration networks, and 1 whole held-out network.
- Validation and calibration each contain exactly one remaining network per family in every fold.
- All {len(split_audit)} folds have zero pairwise network overlap and zero pairwise observation overlap.
- Original Step 9 partition labels are retained only as metadata. All {len(original_role_audit)} original-label groups appear across train, validation, calibration, and held-out roles; no original label is reused as a fixed role.
- Every individual network appears in all four roles across the rotation and exactly once as held-out.
- Normalization, point fitting, point selection, residual-scale fitting, conformal calibration, and evaluation use train, train, validation, validation, calibration, and held-out roles respectively.
- Alpha={float(config['conformal_alpha']):.2f} is fixed in configuration before folds. There is no fold-wise alpha selection.
- Fold-local distance divisors range from {normalization['training_raw_route_distance_maximum_km'].min():.4f} to {normalization['training_raw_route_distance_maximum_km'].max():.4f} km. No clipping is used; {int(normalization['heldout_above_one_count'].sum())} held-out rows exceed normalized distance 1.0.

## Point prediction aggregate

- Pooled held-out rows: {int(point_overall['n_pooled_samples'])}.
- Pooled MAE: {float(point_overall['pooled_mae']):.4f} minutes.
- Pooled RMSE: {float(point_overall['pooled_rmse']):.4f} minutes.
- Pooled R²: {float(point_overall['pooled_r2']):.4f}.
- Macro fold mean ± SD RMSE: {float(point_overall['macro_fold_mean_rmse']):.4f} ± {float(point_overall['macro_fold_std_rmse']):.4f} minutes.

{_markdown_table(selected_counts, ['selected_model', 'fold_count'])}

### By held-out topology

{_markdown_table(point_by_topology, ['topology', 'fold_count', 'n_pooled_samples', 'pooled_mae', 'pooled_rmse', 'pooled_r2'])}

## Conformal aggregate at alpha 0.10

{_markdown_table(conformal_aggregate, ['method', 'n_samples', 'nominal_coverage', 'empirical_coverage', 'mean_interval_width', 'macro_fold_std_coverage'])}

### By held-out topology

{_markdown_table(conformal_by_topology, ['topology', 'method', 'n_samples', 'empirical_coverage', 'mean_interval_width'])}

## Per-fold results

{_markdown_table(fold_summary.sort_values(['heldout_topology', 'heldout_network_id']), ['fold_id', 'heldout_network_id', 'heldout_topology', 'selected_model', 'mae', 'rmse', 'r2', 'absolute', 'relative', 'local'])}

## Explicit unfavorable folds

The following are the five folds with the worst minimum conformal-method coverage, breaking ties by higher RMSE. They are retained without exclusion or post-hoc retuning.

{_markdown_table(unfavorable_top, ['fold_id', 'heldout_network_id', 'heldout_topology', 'selected_model', 'rmse', 'r2', 'absolute_coverage', 'relative_coverage', 'local_coverage'])}

## Scientific limitations

1. Twenty synthetic networks, only four per family, are a small corpus. These folds are strongly dependent because the same networks recur in fitting roles.
2. Leave-one-network-out performance on these generators is not universal network, city, traffic, medical, or operational generalization.
3. Original Step 9 networks vary in size, density, and base-time multiplier. Topology summaries therefore mix topology with finite configured instance variation.
4. Conformal guarantees are exchangeability-based; systematic network or topology shifts can cause under-coverage, which is reported rather than repaired using held-out data.
5. No held-out fold is used to choose a model, residual scale, conformal quantile, normalization, or alpha. Consequently, unfavorable held-out behavior is not adaptively corrected.
6. This experiment contains no quantum circuit or hardware execution and supports no speedup claim.

## Reproduction

Run `python experiments/run_step9_1_leave_one_network_out.py` from the repository root. The runner refuses to overwrite an existing result directory. The manifest records both configuration hashes, package versions, seeds, source hashes, and artifact hashes.
"""


def run() -> None:
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(
            f"refusing to overwrite existing Step 9.1 output: {OUTPUT_DIRECTORY}"
        )
    config, corpus_config, corpus_config_path = _load_configs()
    if tuple(MODEL_FEATURE_COLUMNS) != FEATURE_COLUMNS:
        raise RuntimeError("traffic and point-model feature contracts disagree")
    if {"network_id", "topology", "scenario", "observation_id"}.intersection(
        FEATURE_COLUMNS
    ):
        raise RuntimeError("identity metadata leaked into model features")
    normalization_config = dict(config["normalization"])
    if normalization_config != {
        "fit_role": "train",
        "raw_column": "route_distance_km",
        "method": "divide_by_training_maximum",
        "clip": False,
    }:
        raise ValueError("unsupported or leakage-prone normalization configuration")

    networks, registry = _build_corpus_networks(corpus_config)
    raw_corpus = _generate_raw_corpus(config, networks, registry)
    splits = _build_fold_splits(config, registry)
    split_audit, role_by_network, original_role_audit = _audit_fold_splits(
        splits, registry
    )
    fold_results = _run_folds(config, raw_corpus, splits)
    runtime_audit = fold_results["runtime_audit"]
    if not (
        (runtime_audit["pairwise_observation_overlap_count"] == 0).all()
        and runtime_audit[
            [
                "heldout_network_absent_from_all_fit_roles",
                "normalization_uses_training_rows_only",
                "model_fit_uses_training_rows_only",
                "model_selection_uses_validation_rows_only",
                "scale_fit_uses_validation_rows_only",
                "conformal_fit_uses_calibration_rows_only",
                "heldout_used_for_evaluation_only",
            ]
        ].all(axis=None)
    ):
        raise RuntimeError("runtime leakage audit failed")
    split_audit = split_audit.merge(runtime_audit, on="fold_id", validate="one_to_one")

    point_aggregate, point_by_topology = _aggregate_point_metrics(
        fold_results["point_by_fold"], fold_results["point_predictions"]
    )
    conformal_aggregate, conformal_by_topology = _aggregate_conformal_metrics(
        fold_results["conformal_by_fold"],
        fold_results["conformal_intervals"],
        float(config["conformal_alpha"]),
    )
    unfavorable = _unfavorable_fold_table(
        fold_results["point_by_fold"], fold_results["conformal_by_fold"]
    )

    expected_rows_per_network = len(config["historical_scenarios"]) * int(
        config["observations_per_network_per_scenario"]
    )
    if len(raw_corpus) != 20 * expected_rows_per_network:
        raise RuntimeError("raw corpus size is incomplete")
    if len(fold_results["point_by_fold"]) != 20:
        raise RuntimeError("not every network was evaluated exactly once")
    if len(fold_results["point_predictions"]) != len(raw_corpus):
        raise RuntimeError("pooled held-out predictions do not cover the corpus once")
    if len(fold_results["conformal_by_fold"]) != 20 * 3:
        raise RuntimeError("conformal fold-method grid is incomplete")
    if len(fold_results["conformal_intervals"]) != len(raw_corpus) * 3:
        raise RuntimeError("held-out conformal intervals do not cover each method")
    primary_numeric_tables = [
        fold_results["point_by_fold"],
        point_aggregate,
        point_by_topology,
        fold_results["conformal_by_fold"],
        conformal_aggregate,
        conformal_by_topology,
    ]
    if any(
        not np.isfinite(table.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
        for table in primary_numeric_tables
    ):
        raise RuntimeError("primary LONO metrics contain non-finite values")

    report = _validation_report(
        config,
        split_audit,
        role_by_network,
        original_role_audit,
        fold_results["normalization"],
        fold_results["point_by_fold"],
        point_aggregate,
        point_by_topology,
        fold_results["conformal_by_fold"],
        conformal_aggregate,
        conformal_by_topology,
        unfavorable,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    tables = {
        "corpus_network_registry.csv": registry,
        "corpus_observations_raw.csv": raw_corpus,
        "fold_split_registry.csv": splits,
        "fold_split_audit.csv": split_audit,
        "role_rotation_by_network.csv": role_by_network,
        "role_rotation_by_original_partition.csv": original_role_audit,
        "fold_normalization.csv": fold_results["normalization"],
        "model_selection_by_fold.csv": fold_results["model_selection"],
        "point_metrics_by_fold.csv": fold_results["point_by_fold"],
        "point_metrics_aggregate.csv": point_aggregate,
        "point_metrics_by_topology.csv": point_by_topology,
        "heldout_point_predictions.csv": fold_results["point_predictions"],
        "conformal_calibration_by_fold.csv": fold_results["conformal_calibration"],
        "conformal_metrics_by_fold.csv": fold_results["conformal_by_fold"],
        "conformal_metrics_aggregate.csv": conformal_aggregate,
        "conformal_metrics_by_topology.csv": conformal_by_topology,
        "heldout_conformal_intervals.csv": fold_results["conformal_intervals"],
        "unfavorable_folds.csv": unfavorable,
    }
    for filename, table in tables.items():
        table.to_csv(OUTPUT_DIRECTORY / filename, index=False)
    joblib.dump(fold_results["models"], OUTPUT_DIRECTORY / "fold_models.joblib")
    write_json(
        OUTPUT_DIRECTORY / "effective_config.json",
        {
            "step9_1_config": config,
            "referenced_step9_corpus_config": corpus_config,
        },
    )
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    source_files = [
        CONFIG_PATH,
        corpus_config_path,
        Path(__file__),
        REPOSITORY_ROOT / "src" / "conformal.py",
        REPOSITORY_ROOT / "src" / "network_schema.py",
        REPOSITORY_ROOT / "src" / "reproducibility.py",
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
        random_seeds={
            "traffic_seed_base": int(config["traffic_seed_base"]),
            "model_seed_base": int(config["model_seed_base"]),
            "residual_scale_seed_base": int(config["residual_scale_seed_base"]),
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "referenced_corpus_config_path": str(corpus_config_path),
            "referenced_corpus_config_sha256": file_sha256(corpus_config_path),
            "network_count": len(registry),
            "fold_count": len(fold_results["point_by_fold"]),
            "all_networks_held_out_exactly_once": True,
            "network_and_observation_overlap_verified_zero": True,
            "preprocessing_fit_role": "fold_train_only",
            "model_selection_role": "fold_validation_only",
            "residual_scale_fit_role": "fold_validation_only",
            "conformal_fit_role": "fold_calibration_only",
            "evaluation_role": "fold_heldout_only",
            "normalization_clipped": False,
            "quantum_or_hardware_executed": False,
            "claims_excluded": [
                "universal_network_generalization",
                "independent_fold_replication",
                "real_traffic_effectiveness",
                "medical_or_operational_effectiveness",
                "quantum_speedup",
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
        f"Step 9.1 complete: {len(registry)} networks, "
        f"{len(fold_results['point_by_fold'])} LONO folds, "
        f"{len(raw_corpus)} pooled held-out rows."
    )
    print(point_aggregate.to_string(index=False))
    print(point_by_topology.to_string(index=False))
    print(conformal_aggregate.to_string(index=False))
    print(conformal_by_topology.to_string(index=False))
    print(unfavorable.head(5).to_string(index=False))
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run()
