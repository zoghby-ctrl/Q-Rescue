"""Calibrate manual split-conformal intervals and test ID/traffic shift."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.conformal import (
    calibrate_relative_supported_alphas,
    calibrate_locally_scaled_supported_alphas,
    calibrate_supported_alphas,
    evaluate_conformal,
    train_residual_scale_model,
)
from src.reproducibility import build_experiment_manifest, write_json
from src.synthetic_networks import SyntheticNetworkSpec, generate_synthetic_network
from src.traffic_data import generate_historical_trip_observations


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "conformal_benchmark.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step5_conformal"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_networks(step4_config: dict[str, object]) -> tuple[dict[str, object], dict[str, str]]:
    networks: dict[str, object] = {}
    topologies: dict[str, str] = {}
    for entry in step4_config["networks"]:  # type: ignore[index]
        values = dict(entry)
        network_id = str(values["network_id"])
        topology = str(values["topology"])
        graph = generate_synthetic_network(
            SyntheticNetworkSpec(
                topology=topology,
                seed=int(values["seed"]),
                parameters=dict(values["parameters"]),
            )
        )
        graph.graph["network_id"] = network_id
        networks[network_id] = graph
        topologies[network_id] = topology
    return networks, topologies


def _tag_metrics(
    alpha: float,
    evaluation_set: str,
    evaluation: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    overall = [
        {"evaluation_set": evaluation_set, **evaluation.overall.iloc[0].to_dict()}
    ]
    by_network = [
        {"evaluation_set": evaluation_set, **row}
        for row in evaluation.by_network.to_dict(orient="records")
    ]
    by_topology = [
        {"evaluation_set": evaluation_set, **row}
        for row in evaluation.by_topology.to_dict(orient="records")
    ]
    return overall, by_network, by_topology


def run() -> None:
    config = _read_json(CONFIG_PATH)
    step4_config_path = REPOSITORY_ROOT / str(config["base_step4_config"])
    step4_artifacts = REPOSITORY_ROOT / str(config["base_step4_artifacts"])
    step4_config = _read_json(step4_config_path)

    calibration_data = pd.read_csv(step4_artifacts / "calibration.csv")
    validation_data = pd.read_csv(step4_artifacts / "validation.csv")
    test_data = pd.read_csv(step4_artifacts / "test.csv")
    base_model = joblib.load(step4_artifacts / "best_model.joblib")

    networks, topology_labels = _build_networks(step4_config)
    shifted_data = generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=list(config["shift_scenarios"]),
        observations_per_network_per_scenario=int(
            config["shift_observations_per_network_per_scenario"]
        ),
        random_seed=int(config["shift_traffic_seed"]),
    )

    scale_model = train_residual_scale_model(
        base_model,
        validation_data,
        random_state=int(config["local_scale_seed"]),
        n_estimators=int(config["local_scale_random_forest_trees"]),
        minimum_scale_minutes=float(config["local_scale_minimum_minutes"]),
    )
    conformal_models_by_method = {
        "absolute": calibrate_supported_alphas(
            base_model,
            calibration_data,
            alphas=list(config["alphas"]),
        ),
        "relative_scaled": calibrate_relative_supported_alphas(
            base_model,
            calibration_data,
            alphas=list(config["alphas"]),
            minimum_scale_minutes=float(config["relative_minimum_scale_minutes"]),
        ),
        "local_scaled": calibrate_locally_scaled_supported_alphas(
            base_model,
            scale_model,
            calibration_data,
            alphas=list(config["alphas"]),
        ),
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shifted_data.to_csv(OUTPUT_DIRECTORY / "traffic_shift_observations.csv", index=False)

    calibration_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    interval_tables: list[pd.DataFrame] = []
    for method, conformal_models in conformal_models_by_method.items():
        for alpha, conformal_model in conformal_models.items():
            calibration_rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "nominal_coverage": 1.0 - alpha,
                    "n_calibration": conformal_model.n_calibration,
                    "finite_sample_quantile_level": conformal_model.quantile_level,
                    "calibration_residual_quantile": conformal_model.calibration_quantile,
                    "quantile_units": "minutes" if method == "absolute" else "normalized_ratio",
                }
            )
            for evaluation_set, frame in (("test_id", test_data), ("test_ood_traffic", shifted_data)):
                evaluation = evaluate_conformal(conformal_model, frame)
                overall, by_network, by_topology = _tag_metrics(
                    alpha, evaluation_set, evaluation
                )
                for row_collection in (overall, by_network, by_topology):
                    for row in row_collection:
                        row["method"] = method
                overall_rows.extend(overall)
                network_rows.extend(by_network)
                topology_rows.extend(by_topology)
                intervals = evaluation.intervals.copy()
                intervals.insert(0, "method", method)
                intervals.insert(1, "evaluation_set", evaluation_set)
                intervals.insert(2, "alpha", alpha)
                interval_tables.append(intervals)

    pd.DataFrame(calibration_rows).to_csv(
        OUTPUT_DIRECTORY / "calibration_quantiles.csv", index=False
    )
    pd.DataFrame(overall_rows).to_csv(
        OUTPUT_DIRECTORY / "coverage_overall.csv", index=False
    )
    pd.DataFrame(network_rows).to_csv(
        OUTPUT_DIRECTORY / "coverage_by_network.csv", index=False
    )
    pd.DataFrame(topology_rows).to_csv(
        OUTPUT_DIRECTORY / "coverage_by_topology.csv", index=False
    )
    pd.concat(interval_tables, ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "intervals.csv", index=False
    )
    joblib.dump(scale_model, OUTPUT_DIRECTORY / "residual_scale_model.joblib")

    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "shift_traffic": int(config["shift_traffic_seed"]),
            "local_scale": int(config["local_scale_seed"]),
        },
        parameters={**config, "selected_base_model": base_model.name},
        config_path=CONFIG_PATH,
    )
    manifest["base_step4_config_sha256"] = build_experiment_manifest(
        "step4-reference",
        REPOSITORY_ROOT,
        {},
        {},
        step4_config_path,
    )["config_sha256"]
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)

    print(pd.DataFrame(calibration_rows).to_string(index=False))
    print(pd.DataFrame(overall_rows).to_string(index=False))
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run()
