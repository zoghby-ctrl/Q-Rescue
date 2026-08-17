"""Run the reproducible, location-neutral Step 4 traffic benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.reproducibility import build_experiment_manifest, write_json
from src.synthetic_networks import SyntheticNetworkSpec, generate_synthetic_network
from src.traffic_data import (
    IDENTITY_METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
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


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "traffic_ai_benchmark.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step4_traffic_ai"


def _load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _build_networks(config: dict[str, object]) -> tuple[dict[str, object], dict[str, str]]:
    networks: dict[str, object] = {}
    topologies: dict[str, str] = {}
    for entry in config["networks"]:  # type: ignore[index]
        network_entry = dict(entry)
        network_id = str(network_entry["network_id"])
        topology = str(network_entry["topology"])
        graph = generate_synthetic_network(
            SyntheticNetworkSpec(
                topology=topology,
                seed=int(network_entry["seed"]),
                parameters=dict(network_entry["parameters"]),
            )
        )
        graph.graph["network_id"] = network_id
        networks[network_id] = graph
        topologies[network_id] = topology
    return networks, topologies


def _quality_report(
    data: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> dict[str, object]:
    numeric = data.loc[:, [*MODEL_FEATURE_COLUMNS, TARGET_COLUMN]].to_numpy(dtype=float)
    split_ids = {name: set(frame["observation_id"]) for name, frame in splits.items()}
    overlap_pairs = {
        f"{left}__{right}": len(split_ids[left].intersection(split_ids[right]))
        for index, left in enumerate(splits)
        for right in list(splits)[index + 1 :]
    }
    return {
        "grain": "one simulated complete origin-destination trip",
        "row_count": len(data),
        "column_count": len(data.columns),
        "unique_observation_ids": int(data["observation_id"].nunique()),
        "duplicate_observation_ids": int(data["observation_id"].duplicated().sum()),
        "total_missing_values": int(data.isna().sum().sum()),
        "all_model_values_finite": bool(np.isfinite(numeric).all()),
        "minimum_travel_time_minutes": float(data[TARGET_COLUMN].min()),
        "maximum_travel_time_minutes": float(data[TARGET_COLUMN].max()),
        "network_count": int(data["network_id"].nunique()),
        "topology_count": int(data["topology"].nunique()),
        "scenario_count": int(data["scenario"].nunique()),
        "route_edge_count_min": int(data["route_edge_count"].min()),
        "route_edge_count_max": int(data["route_edge_count"].max()),
        "multi_edge_trip_share": float((data["route_edge_count"] > 1).mean()),
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "identity_columns_excluded_from_features": sorted(
            set(IDENTITY_METADATA_COLUMNS).difference(MODEL_FEATURE_COLUMNS)
        ),
        "feature_contract_matches_model": tuple(MODEL_FEATURE_COLUMNS) == FEATURE_COLUMNS,
        "split_row_counts": {name: len(frame) for name, frame in splits.items()},
        "split_overlap_counts": overlap_pairs,
        "split_ids_cover_source": set().union(*split_ids.values())
        == set(data["observation_id"]),
        "networks_per_split": {
            name: int(frame["network_id"].nunique()) for name, frame in splits.items()
        },
    }


def _evaluation_rows(model_name: str, split_name: str, evaluation: object) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    overall = evaluation.overall.iloc[0].to_dict()
    overall_rows = [{"model_name": model_name, "split": split_name, **overall}]
    network_rows = [
        {"model_name": model_name, "split": split_name, **row}
        for row in evaluation.by_network.to_dict(orient="records")
    ]
    topology_rows = [
        {"model_name": model_name, "split": split_name, **row}
        for row in evaluation.by_topology.to_dict(orient="records")
    ]
    return overall_rows, network_rows, topology_rows


def run() -> None:
    config = _load_config()
    networks, topology_labels = _build_networks(config)
    data = generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=list(config["scenarios"]),
        observations_per_network_per_scenario=int(
            config["observations_per_network_per_scenario"]
        ),
        random_seed=int(config["traffic_seed"]),
    )
    splits = split_traffic_observations(
        data,
        split_fractions=dict(config["split_fractions"]),
        random_seed=int(config["split_seed"]),
    )
    validate_split_separation(splits, expected_observation_ids=data["observation_id"])

    fitted_models = train_all_traffic_models(
        splits["train"],
        random_state=int(config["model_seed"]),
        random_forest_trees=int(config["random_forest_trees"]),
    )
    selection = select_best_by_validation_rmse(fitted_models, splits["validation"])

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    split_lookup = {
        observation_id: split_name
        for split_name, frame in splits.items()
        for observation_id in frame["observation_id"]
    }
    artifact_data = data.copy()
    artifact_data.insert(
        1, "split", artifact_data["observation_id"].map(split_lookup)
    )
    artifact_data.to_csv(OUTPUT_DIRECTORY / "traffic_observations.csv", index=False)
    for split_name, frame in splits.items():
        frame.to_csv(OUTPUT_DIRECTORY / f"{split_name}.csv", index=False)

    overall_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []
    for model_name, model in fitted_models.items():
        for split_name in ("validation", "test"):
            evaluation = evaluate_traffic_model(model, splits[split_name])
            overall, by_network, by_topology = _evaluation_rows(
                model_name, split_name, evaluation
            )
            overall_rows.extend(overall)
            network_rows.extend(by_network)
            topology_rows.extend(by_topology)
            prediction_table = evaluation.predictions.copy()
            prediction_table.insert(0, "model_name", model_name)
            prediction_table.insert(1, "split", split_name)
            prediction_tables.append(prediction_table)

    selection.ranking.to_csv(OUTPUT_DIRECTORY / "model_selection.csv", index=False)
    pd.DataFrame(overall_rows).to_csv(
        OUTPUT_DIRECTORY / "model_metrics_overall.csv", index=False
    )
    pd.DataFrame(network_rows).to_csv(
        OUTPUT_DIRECTORY / "model_metrics_by_network.csv", index=False
    )
    pd.DataFrame(topology_rows).to_csv(
        OUTPUT_DIRECTORY / "model_metrics_by_topology.csv", index=False
    )
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "predictions.csv", index=False
    )
    joblib.dump(selection.best_model, OUTPUT_DIRECTORY / "best_model.joblib")

    quality = _quality_report(data, splits)
    write_json(OUTPUT_DIRECTORY / "data_quality.json", quality)
    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "traffic": int(config["traffic_seed"]),
            "split": int(config["split_seed"]),
            "model": int(config["model_seed"]),
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest["best_validation_model"] = selection.best_model_name
    manifest["artifact_directory"] = str(OUTPUT_DIRECTORY)
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)

    test_metrics = pd.DataFrame(overall_rows)
    test_metrics = test_metrics[test_metrics["split"] == "test"]
    print(f"Generated {len(data)} trip observations across {len(networks)} networks.")
    print(f"Selected by validation RMSE: {selection.best_model_name}")
    print(test_metrics.to_string(index=False))
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run()
