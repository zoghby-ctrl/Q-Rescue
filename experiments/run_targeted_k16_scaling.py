"""Run a resumable, matched-prefix K=16 indicative scaling study."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import joblib
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.run_step9_2_topology_uncertainty_qaoa import (  # noqa: E402
    _build_networks,
    _problem_hash,
    _residual_pools,
    _run_trial_worker,
    _traffic_state,
)
from src.candidate_routes import summarize_route_diversity  # noqa: E402
from src.conformal import finite_sample_residual_quantile  # noqa: E402
from src.controlled_objectives import controlled_objective_table  # noqa: E402
from src.reproducibility import (  # noqa: E402
    build_experiment_manifest,
    file_sha256,
    write_json,
)
from src.route_features import build_route_feature_table  # noqa: E402
from src.statistical_inference import percentile_bootstrap_interval  # noqa: E402


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "targeted_k16_scaling.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "targeted_k16_scaling"
CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"
FAILURE_DIRECTORY = OUTPUT_DIRECTORY / "failures"


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _source_directory(config: dict[str, Any]) -> Path:
    path = (REPOSITORY_ROOT / str(config["source_directory"])).resolve()
    if REPOSITORY_ROOT.resolve() not in path.parents or not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _verify_sources(source: Path) -> dict[str, str]:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = [
        "effective_config.json",
        "network_registry.csv",
        "problem_summary.csv",
        "candidate_route_problems.csv",
        "qaoa_runs.csv",
        "validation_checks.json",
    ]
    verified = {"manifest.json": file_sha256(manifest_path)}
    for name in required:
        actual = file_sha256(source / name)
        if manifest["artifact_sha256"].get(name) != actual:
            raise RuntimeError(f"Step 9.2 artifact hash mismatch: {name}")
        verified[name] = actual
    checks = json.loads((source / "validation_checks.json").read_text())
    if not checks.get("all_checks_pass") or checks.get("trial_count") != 1800:
        raise RuntimeError("Step 9.2 is not publication-complete")
    return verified


def _relative_path(path_text: str) -> Path:
    path = (REPOSITORY_ROOT / path_text).resolve()
    if REPOSITORY_ROOT.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _preparation_tables(
    config: dict[str, Any], source: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    problem_path = OUTPUT_DIRECTORY / "problem_summary.csv"
    route_path = OUTPUT_DIRECTORY / "candidate_routes.csv"
    prefix_path = OUTPUT_DIRECTORY / "k8_prefix_reproduction.csv"
    registry_path = OUTPUT_DIRECTORY / "network_registry.csv"
    if all(path.is_file() for path in [problem_path, route_path, prefix_path, registry_path]):
        problems = pd.read_csv(problem_path)
        if len(problems) != 18 or problems["problem_id"].duplicated().any():
            raise RuntimeError("existing K=16 preparation is incomplete")
        return (
            problems,
            pd.read_csv(route_path),
            pd.read_csv(prefix_path),
            pd.read_csv(registry_path),
        )
    if any(CHECKPOINT_DIRECTORY.glob("*.json")):
        raise RuntimeError("cannot change K=16 problems after checkpoints exist")

    primary_config = json.loads((source / "effective_config.json").read_text())
    primary = dict(primary_config["primary_design"])
    accepted_registry = pd.read_csv(source / "network_registry.csv")
    topology_mapping = accepted_registry[
        ["topology", "topology_index"]
    ].drop_duplicates()
    if topology_mapping["topology"].duplicated().any():
        raise RuntimeError("accepted topology-index mapping is not unique")
    original_topology_index = {
        str(row.topology): int(row.topology_index)
        for row in topology_mapping.itertuples(index=False)
    }
    original_topologies = list(original_topology_index)
    selected_topologies = list(map(str, config["topologies"]))
    if any(topology not in original_topologies for topology in selected_topologies):
        raise ValueError("targeted topology is absent from the primary design")
    network_primary = dict(primary)
    network_primary["topologies"] = {
        topology: dict(primary["topologies"])[topology]
        for topology in selected_topologies
    }
    network_primary["matched_seed_ids"] = list(map(int, config["matched_seed_ids"]))
    network_primary["candidate_counts"] = [int(config["K"])]
    networks, routes_by_network, registry, _ = _build_networks(network_primary)
    registry["topology_index"] = registry["topology"].map(original_topology_index)
    registry = registry.rename(columns={"K8_complete": "K16_complete"})
    if not registry["K16_complete"].all():
        raise RuntimeError("a targeted network lacks 16 candidate routes")

    sources = dict(primary_config["inherited_sources"])
    for key in ["point_model", "residual_scale_model", "traffic_observations"]:
        source_record = dict(sources[key])
        path = _relative_path(str(source_record["path"]))
        if file_sha256(path) != str(source_record["sha256"]):
            raise RuntimeError(f"inherited source hash mismatch: {key}")
    point_model = joblib.load(_relative_path(str(dict(sources["point_model"])["path"])))
    scale_model = joblib.load(
        _relative_path(str(dict(sources["residual_scale_model"])["path"]))
    )
    observations = pd.read_csv(
        _relative_path(str(dict(sources["traffic_observations"])["path"]))
    )
    normalization_values = observations["distance_normalization_km"].unique()
    if len(normalization_values) != 1:
        raise RuntimeError("frozen distance normalization is not unique")
    normalization = float(normalization_values[0])
    calibration_z, operational_z, _ = _residual_pools(
        primary, point_model, scale_model, observations
    )

    source_problem_table = pd.read_csv(source / "problem_summary.csv")
    uncertainty_mapping = source_problem_table[
        ["uncertainty", "uncertainty_index"]
    ].drop_duplicates()
    if uncertainty_mapping["uncertainty"].duplicated().any():
        raise RuntimeError("accepted uncertainty-index mapping is not unique")
    accepted_uncertainty_index = {
        str(row.uncertainty): int(row.uncertainty_index)
        for row in uncertainty_mapping.itertuples(index=False)
    }
    uncertainty_items = sorted(
        dict(primary["uncertainty_regimes"]).items(),
        key=lambda item: accepted_uncertainty_index[str(item[0])],
    )
    selected_uncertainty = set(map(str, config["uncertainty_regimes"]))
    baseline_fraction = float(primary["baseline_prediction_noise_fraction"])
    calibration_draws: dict[str, tuple[np.ndarray, np.ndarray, float, int]] = {}
    for uncertainty_index, (name, raw_regime) in enumerate(uncertainty_items):
        target = float(dict(raw_regime)["target_prediction_noise_fraction"])
        additional = float(np.sqrt(max(0.0, target**2 - baseline_fraction**2)))
        rng = np.random.default_rng(
            int(primary["uncertainty_calibration_seed"]) + uncertainty_index
        )
        count = int(primary["uncertainty_monte_carlo_draws"])
        calibration_draws[name] = (
            rng.choice(calibration_z, size=count, replace=True),
            rng.normal(0.0, 1.0, size=count),
            additional,
            uncertainty_index,
        )
    if not selected_uncertainty.issubset(calibration_draws):
        raise ValueError("targeted uncertainty regime is absent from primary design")

    source_problems = source_problem_table.set_index("problem_id")
    source_routes = pd.read_csv(source / "candidate_route_problems.csv")
    route_rows: list[dict[str, object]] = []
    problem_rows: list[dict[str, object]] = []
    prefix_rows: list[dict[str, object]] = []
    traffic_parameters = dict(primary["traffic_state"])
    risk_weights = dict(primary["risk_weights"])

    for network in registry.itertuples(index=False):
        network_id = str(network.network_id)
        topology = str(network.topology)
        topology_index = int(original_topology_index[topology])
        seed_id = int(network.matched_seed_id)
        graph = networks[network_id]
        routes16 = routes_by_network[network_id]
        state_seed_root = (
            int(primary["traffic_state_seed_base"])
            + topology_index * 10_000
            + seed_id * 100
        )
        states = {
            route.route_id: _traffic_state(
                route,
                seed=state_seed_root + route.route_index,
                parameters=traffic_parameters,
            )
            for route in routes16
        }
        features = build_route_feature_table(
            graph,
            routes16,
            states,
            distance_normalization_km=normalization,
        )
        predictions = np.asarray(point_model.predict(features), dtype=float)
        scales = np.asarray(scale_model.predict_scale(features), dtype=float)
        base_times = features["base_route_time_min"].to_numpy(dtype=float)
        if not np.isfinite(predictions).all() or not np.isfinite(scales).all():
            raise RuntimeError(f"nonfinite K=16 predictions for {network_id}")

        source_route_prefix = source_routes[
            (source_routes["network_id"] == network_id)
            & (source_routes["uncertainty"] == "low")
            & (source_routes["K"] == 8)
        ].sort_values("route_index")
        if len(source_route_prefix) != 8:
            raise RuntimeError(f"missing accepted K=8 route prefix for {network_id}")
        generated_nodes = [" -> ".join(map(str, route.nodes)) for route in routes16[:8]]
        nodes_match = generated_nodes == source_route_prefix["ordered_nodes"].tolist()
        prediction_prefix_error = float(
            np.max(
                np.abs(
                    predictions[:8]
                    - source_route_prefix["prediction_minutes"].to_numpy(float)
                )
            )
        )
        scale_prefix_error = float(
            np.max(
                np.abs(
                    scales[:8]
                    - source_route_prefix["local_residual_scale_minutes"].to_numpy(float)
                )
            )
        )
        if not nodes_match or max(prediction_prefix_error, scale_prefix_error) > 1e-9:
            raise RuntimeError(
                f"K=16 route prefix does not reproduce K=8: {network_id}; "
                f"nodes_match={nodes_match}, prediction_error={prediction_prefix_error}, "
                f"scale_error={scale_prefix_error}, topology_index={topology_index}, "
                f"state_seed_root={state_seed_root}"
            )

        deadline_values = source_problems[
            (source_problems["network_id"] == network_id)
            & (source_problems["K"] == 8)
        ]["deadline_minutes"].unique()
        if len(deadline_values) != 1:
            raise RuntimeError(f"K=8 deadline is not unique for {network_id}")
        deadline = float(deadline_values[0])

        outcome_seed = (
            int(primary["operational_outcome_seed_base"])
            + topology_index * 10_000
            + seed_id * 100
        )
        prefix_rng = np.random.default_rng(outcome_seed)
        base_outcome = prefix_rng.choice(operational_z, size=8, replace=True)
        extra_outcome = prefix_rng.normal(0.0, 1.0, size=8)
        extension_rng = np.random.default_rng(
            outcome_seed + int(config["extension_outcome_seed_offset"])
        )
        base_outcome = np.concatenate(
            [base_outcome, extension_rng.choice(operational_z, size=8, replace=True)]
        )
        extra_outcome = np.concatenate(
            [extra_outcome, extension_rng.normal(0.0, 1.0, size=8)]
        )
        diversity = summarize_route_diversity(routes16)

        for uncertainty_name, _ in uncertainty_items:
            if uncertainty_name not in selected_uncertainty:
                continue
            base_draw, extra_draw, additional, uncertainty_index = calibration_draws[
                uncertainty_name
            ]
            half_widths = np.asarray(
                [
                    finite_sample_residual_quantile(
                        np.abs(
                            base_draw * scales[index]
                            + extra_draw * additional * base_times[index]
                        ),
                        float(primary["conformal_alpha"]),
                    )
                    for index in range(int(config["K"]))
                ]
            )
            actual_times = np.maximum(
                0.05,
                predictions
                + base_outcome * scales
                + extra_outcome * additional * base_times,
            )
            objective = controlled_objective_table(
                predictions,
                half_widths,
                deadline_minutes=deadline,
                lambda_deadline=float(risk_weights["lambda_deadline"]),
                lambda_uncertainty=float(risk_weights["lambda_uncertainty"]),
            )
            risks = objective[
                "C_expected_plus_deadline_plus_uncertainty"
            ].to_numpy(float)
            source_problem_id = f"{topology}_s{seed_id:02d}_{uncertainty_name}_K8"
            source_problem = source_problems.loc[source_problem_id]
            source_risks = np.asarray(json.loads(source_problem["risk_vector_json"]), float)
            source_actual = np.asarray(
                json.loads(source_problem["actual_route_times_json"]), float
            )
            risk_prefix_error = float(np.max(np.abs(risks[:8] - source_risks)))
            actual_prefix_error = float(
                np.max(np.abs(actual_times[:8] - source_actual))
            )
            half_width_prefix = source_routes[
                source_routes["problem_id"] == source_problem_id
            ].sort_values("route_index")["uncertainty_half_width_minutes"].to_numpy(float)
            half_width_prefix_error = float(
                np.max(np.abs(half_widths[:8] - half_width_prefix))
            )
            prefix_rows.append(
                {
                    "network_id": network_id,
                    "uncertainty": uncertainty_name,
                    "route_nodes_exact_match": nodes_match,
                    "prediction_max_absolute_error": prediction_prefix_error,
                    "residual_scale_max_absolute_error": scale_prefix_error,
                    "half_width_max_absolute_error": half_width_prefix_error,
                    "risk_max_absolute_error": risk_prefix_error,
                    "actual_time_max_absolute_error": actual_prefix_error,
                    "deadline_absolute_error": abs(
                        deadline - float(source_problem["deadline_minutes"])
                    ),
                }
            )
            if max(risk_prefix_error, actual_prefix_error, half_width_prefix_error) > 1e-9:
                raise RuntimeError(
                    f"K=16 problem does not preserve K=8 prefix: {source_problem_id}; "
                    f"risk_error={risk_prefix_error}, actual_error={actual_prefix_error}, "
                    f"half_width_error={half_width_prefix_error}"
                )

            exact_index = int(np.argmin(risks))
            expected_index = int(np.argmin(predictions))
            fastest_index = int(np.argmin(actual_times))
            problem_id = f"{topology}_s{seed_id:02d}_{uncertainty_name}_K16"
            payload = {
                "problem_id": problem_id,
                "graph_seed": int(network.graph_seed),
                "uncertainty": uncertainty_name,
                "K": 16,
                "route_ids": objective["route_id"].tolist(),
                "predictions": predictions.tolist(),
                "half_widths": half_widths.tolist(),
                "deadline": deadline,
                "risk_vector": risks.tolist(),
                "actual_times": actual_times.tolist(),
            }
            risk_sorted = np.sort(risks)
            problem_rows.append(
                {
                    "problem_id": problem_id,
                    "problem_sha256": _problem_hash(payload),
                    "network_id": network_id,
                    "topology": topology,
                    "topology_index": topology_index,
                    "matched_seed_id": seed_id,
                    "graph_seed": int(network.graph_seed),
                    "uncertainty": uncertainty_name,
                    "uncertainty_index": int(uncertainty_index),
                    "K": 16,
                    "deadline_minutes": deadline,
                    "risk_vector_json": json.dumps(risks.tolist()),
                    "actual_route_times_json": json.dumps(actual_times.tolist()),
                    "expected_time_route_index": expected_index,
                    "exact_risk_route_index": exact_index,
                    "actual_fastest_route_index": fastest_index,
                    "exact_risk": float(risks[exact_index]),
                    "candidate_expected_time_spread": float(np.ptp(predictions)),
                    "candidate_risk_spread": float(np.ptp(risks)),
                    "best_second_risk_gap": float(risk_sorted[1] - risk_sorted[0]),
                    "mean_candidate_uncertainty_half_width": float(np.mean(half_widths)),
                    "candidate_uncertainty_spread": float(np.ptp(half_widths)),
                    "route_mean_jaccard_similarity": float(
                        diversity["mean_jaccard_similarity"]
                    ),
                }
            )
            for index, route in enumerate(routes16):
                route_rows.append(
                    {
                        "problem_id": problem_id,
                        "network_id": network_id,
                        "topology": topology,
                        "matched_seed_id": seed_id,
                        "uncertainty": uncertainty_name,
                        "K": 16,
                        "route_index": index,
                        "route_id": route.route_id,
                        "ordered_nodes": " -> ".join(map(str, route.nodes)),
                        "prediction_minutes": float(predictions[index]),
                        "residual_scale_minutes": float(scales[index]),
                        "uncertainty_half_width_minutes": float(half_widths[index]),
                        "risk_score": float(risks[index]),
                        "actual_travel_time_minutes": float(actual_times[index]),
                        "deadline_minutes": deadline,
                        "is_exact_risk_optimum": index == exact_index,
                        "is_actual_fastest": index == fastest_index,
                    }
                )

    problems = pd.DataFrame(problem_rows).sort_values(
        ["topology_index", "matched_seed_id", "uncertainty_index"], kind="stable"
    )
    routes = pd.DataFrame(route_rows).sort_values(
        ["problem_id", "route_index"], kind="stable"
    )
    prefix = pd.DataFrame(prefix_rows).sort_values(
        ["network_id", "uncertainty"], kind="stable"
    )
    if len(problems) != 18 or len(routes) != 18 * 16 or len(prefix) != 18:
        raise RuntimeError("K=16 preparation has an unexpected sample size")
    _atomic_csv(problem_path, problems)
    _atomic_csv(route_path, routes)
    _atomic_csv(prefix_path, prefix)
    _atomic_csv(registry_path, registry)
    return problems, routes, prefix, registry


def _trial_specs(problems: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for problem in problems.itertuples(index=False):
        record = problem._asdict()
        for depth in map(int, config["depths"]):
            optimization_seed = (
                int(config["optimization_seed_base"])
                + int(record["topology_index"]) * 10_000
                + int(record["matched_seed_id"]) * 100
                + int(record["uncertainty_index"]) * 20
                + depth
            )
            for method in map(str, config["mixers"]):
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
                        "K": 16,
                        "depth": depth,
                        "method": method,
                        "optimization_seed": optimization_seed,
                        "iterations": int(config["iterations"]),
                        "step_size": float(config["step_size"]),
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


def benchmark(problems: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    benchmark_path = OUTPUT_DIRECTORY / "runtime_benchmark.csv"
    estimate_path = OUTPUT_DIRECTORY / "runtime_estimate.json"
    if benchmark_path.is_file() and estimate_path.is_file():
        return pd.read_csv(benchmark_path)
    representative = problems[
        (problems["topology"] == "grid")
        & (problems["matched_seed_id"] == 0)
        & (problems["uncertainty"] == "low")
    ]
    if len(representative) != 1:
        raise RuntimeError("K=16 benchmark problem is not unique")
    base_specs = _trial_specs(representative, config)
    rows: list[dict[str, object]] = []
    for iterations in map(int, config["benchmark_iterations"]):
        for spec in base_specs:
            trial = dict(spec)
            trial["iterations"] = iterations
            trial["run_id"] = f"benchmark_i{iterations}_p{spec['depth']}_{spec['method']}"
            result = _run_trial_worker(trial)
            rows.append(
                {
                    "benchmark_iterations": iterations,
                    "depth": int(spec["depth"]),
                    "method": str(spec["method"]),
                    "wall_time_seconds": float(result["metric"]["wall_time_seconds"]),
                    "statevector_complex_bytes": 2**16 * 16,
                    "statevector_mib": 2**16 * 16 / 2**20,
                    "qubo_nonidentity_terms": 16 + 16 * 15 // 2,
                }
            )
            print(
                f"K16 benchmark i={iterations} p={spec['depth']} {spec['method']}: "
                f"{rows[-1]['wall_time_seconds']:.2f}s",
                flush=True,
            )
    table = pd.DataFrame(rows)
    per_cell: list[dict[str, object]] = []
    for (depth, method), group in table.groupby(["depth", "method"], sort=True):
        times = group.set_index("benchmark_iterations")["wall_time_seconds"]
        first = float(times.loc[1])
        third = float(times.loc[3])
        slope = max(0.0, (third - first) / 2.0)
        intercept = max(0.0, first - slope)
        estimated = 1.25 * (intercept + slope * int(config["iterations"]))
        per_cell.append(
            {
                "depth": int(depth),
                "method": method,
                "estimated_seconds_per_15_iteration_trial_with_25pct_margin": estimated,
            }
        )
    cell_table = pd.DataFrame(per_cell)
    replications = 18
    sequential_seconds = float(
        cell_table["estimated_seconds_per_15_iteration_trial_with_25pct_margin"].sum()
        * replications
    )
    workers = int(config["default_workers"])
    estimate = {
        "K": 16,
        "complex_statevector_mib": 1.0,
        "probability_vector_mib": 0.5,
        "logical_qubo_nonidentity_terms": 136,
        "trial_count": 72,
        "estimated_sequential_seconds_with_margin": sequential_seconds,
        "estimated_parallel_seconds_at_default_workers": sequential_seconds / workers,
        "estimated_parallel_hours_at_default_workers": sequential_seconds / workers / 3600.0,
        "default_workers": workers,
        "estimation_method": "linear fit to 1- and 3-iteration real K16 cells plus 25% margin",
    }
    _atomic_csv(benchmark_path, table)
    _atomic_csv(OUTPUT_DIRECTORY / "runtime_estimate_by_cell.csv", cell_table)
    _atomic_json(estimate_path, estimate)
    if estimate["estimated_parallel_hours_at_default_workers"] > float(
        config["maximum_estimated_parallel_hours"]
    ):
        raise RuntimeError(
            "K=16 estimate exceeds the predeclared compute stop threshold; "
            "benchmark artifacts were preserved"
        )
    return table


def run_trials(
    problems: pd.DataFrame, config: dict[str, Any], *, workers: int
) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FAILURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    specs = _trial_specs(problems, config)
    missing = [
        spec
        for spec in specs
        if not (CHECKPOINT_DIRECTORY / f"{spec['run_id']}.json").is_file()
    ]
    print(
        f"K16 checkpoints: {len(specs) - len(missing)}/{len(specs)} complete; "
        f"launching {len(missing)} with {workers} workers.",
        flush=True,
    )
    if not missing:
        return
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_spec = {
            executor.submit(_run_trial_worker, spec): spec for spec in missing
        }
        completed = len(specs) - len(missing)
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                result = future.result()
                _atomic_json(
                    CHECKPOINT_DIRECTORY / f"{spec['run_id']}.json", result
                )
            except Exception as error:  # pragma: no cover - process failures are retained
                failures += 1
                _atomic_json(
                    FAILURE_DIRECTORY / f"{spec['run_id']}.json",
                    {
                        "failed_at_utc": _utc_now(),
                        "specification": spec,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                )
            completed += 1
            print(
                f"K16 progress {completed}/{len(specs)}; failures this launch={failures}",
                flush=True,
            )
    if failures:
        raise RuntimeError(f"{failures} K=16 trials failed; resumable records preserved")


def _load_checkpoints(expected: int) -> list[dict[str, Any]]:
    paths = sorted(CHECKPOINT_DIRECTORY.glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} K=16 checkpoints, found {len(paths)}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _scaling_tables(
    k16_runs: pd.DataFrame, source: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    k8 = pd.read_csv(source / "qaoa_runs.csv")
    k8 = k8[
        (k8["K"] == 8)
        & (k8["topology"].isin(config["topologies"]))
        & (k8["matched_seed_id"].isin(config["matched_seed_ids"]))
        & (k8["uncertainty"].isin(config["uncertainty_regimes"]))
        & (k8["depth"].isin(config["depths"]))
        & (k8["method"].isin(config["mixers"]))
    ].copy()
    keys = ["network_id", "topology", "matched_seed_id", "uncertainty", "depth", "method"]
    metrics = [
        "feasibility_probability",
        "optimal_state_probability",
        "optimal_probability_given_feasible",
        "normalized_feasible_entropy",
        "risk_regret",
        "wall_time_seconds",
    ]
    pairs = k8[keys + metrics].merge(
        k16_runs[keys + metrics],
        on=keys,
        suffixes=("_K8", "_K16"),
        validate="one_to_one",
    )
    if len(pairs) != 72:
        raise RuntimeError("K8/K16 matched comparison is incomplete")
    rows: list[dict[str, object]] = []
    for method in map(str, config["mixers"]):
        subset = pairs[pairs["method"] == method]
        for metric in metrics:
            difference = subset[f"{metric}_K16"] - subset[f"{metric}_K8"]
            network_difference = difference.groupby(subset["network_id"]).mean()
            estimate, lower, upper = percentile_bootstrap_interval(
                network_difference.to_numpy(float),
                resamples=int(config["bootstrap_resamples"]),
                seed=int(config["bootstrap_seed"]) + len(rows),
            )
            rows.append(
                {
                    "contrast": "K16_minus_K8",
                    "method": method,
                    "metric": metric,
                    "matched_trial_count": int(len(difference)),
                    "independent_network_count": int(len(network_difference)),
                    "mean_network_difference": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    return pairs, pd.DataFrame(rows)


def finalize(config: dict[str, Any], source: Path, verified: dict[str, str]) -> None:
    records = _load_checkpoints(72)
    runs = pd.DataFrame([record["metric"] for record in records])
    distributions = pd.DataFrame(
        [row for record in records for row in record["feasible_distribution"]]
    )
    history = pd.DataFrame(
        [row for record in records for row in record["training_history"]]
    )
    if runs["run_id"].duplicated().any() or len(runs) != 72:
        raise RuntimeError("K=16 trial IDs are not unique and complete")
    if len(distributions) != 72 * 16 or len(history) != 72 * 16:
        raise RuntimeError("K=16 distribution/history row count is incomplete")
    xy = runs[runs["method"] == "feasible_xy"]["feasibility_probability"]
    if not np.allclose(xy, 1.0, atol=1e-8):
        raise RuntimeError("ideal K=16 XY trial left the one-hot subspace")
    numeric = runs[
        [
            "feasibility_probability",
            "optimal_state_probability",
            "optimal_probability_given_feasible",
            "normalized_feasible_entropy",
            "energy_regret",
            "risk_regret",
            "wall_time_seconds",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all() or np.any(runs["energy_regret"] < -1e-8):
        raise RuntimeError("K=16 result contains invalid metrics")
    aggregate = (
        runs.groupby(["depth", "method"], sort=True)[
            [
                "feasibility_probability",
                "optimal_state_probability",
                "optimal_probability_given_feasible",
                "normalized_feasible_entropy",
                "risk_regret",
                "wall_time_seconds",
            ]
        ]
        .agg(["count", "mean", "median", "std"])
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    pairs, effects = _scaling_tables(runs, source, config)
    checkpoint_hashes = {
        path.name: file_sha256(path)
        for path in sorted(CHECKPOINT_DIRECTORY.glob("*.json"))
    }

    artifacts: dict[str, pd.DataFrame] = {
        "qaoa_runs.csv": runs.sort_values("run_id"),
        "feasible_route_distributions.csv": distributions.sort_values(
            ["run_id", "route_index"]
        ),
        "qaoa_training_history.csv": history.sort_values(["run_id", "iteration"]),
        "k16_aggregate.csv": aggregate,
        "k8_k16_matched_pairs.csv": pairs,
        "k16_minus_k8_effects.csv": effects,
    }
    for name, table in artifacts.items():
        _atomic_csv(OUTPUT_DIRECTORY / name, table)
    _atomic_json(OUTPUT_DIRECTORY / "checkpoint_hashes.json", checkpoint_hashes)

    xy_summary = aggregate[aggregate["method"] == "feasible_xy"]
    effect_summary = effects[
        (effects["method"] == "feasible_xy")
        & effects["metric"].isin(
            ["optimal_state_probability", "normalized_feasible_entropy"]
        )
    ]
    report = f"""# Validation Report — Targeted K=16 Scaling

## Status

- Status: **PASS** for 72 targeted ideal-statevector K=16 trials.
- Independent networks: 9 (three topology families × three matched seeds).
- Uncertainty: low/high; depths: p=1/3; mixers: standard-X/feasible-XY.
- Every K=16 problem exactly reproduces its accepted K=8 first-route prefix, risk scores, realized outcomes, and deadline to 1e-9.
- Optimizer: the same 15-step Adam budget as the primary K=4/8 study.

## K=16 feasible-XY summary

{xy_summary[['depth', 'feasibility_probability_mean', 'optimal_state_probability_mean', 'optimal_probability_given_feasible_mean', 'normalized_feasible_entropy_mean', 'risk_regret_mean', 'wall_time_seconds_mean']].to_markdown(index=False, floatfmt='.6f')}

## Matched K=16 minus K=8 effects

{effect_summary[['metric', 'mean_network_difference', 'ci95_lower', 'ci95_upper', 'independent_network_count']].to_markdown(index=False, floatfmt='.6f')}

## Guardrails

1. K=16 has only 65,536 computational states and is classically trivial; exact candidate-risk minimization is immediate.
2. This is an indicative scaling trend across nine controlled synthetic networks, not a claim about useful quantum scale.
3. The statevector simulator and exact enumeration are classical computations. Runtime is reported only for reproducibility, never as a speed comparison.
4. The fixed optimizer budget may under-optimize the larger circuit, but it is held equal to the primary study instead of being tuned for a favorable K=16 result.
"""
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    artifact_names = [
        *artifacts,
        "checkpoint_hashes.json",
        "problem_summary.csv",
        "candidate_routes.csv",
        "k8_prefix_reproduction.csv",
        "network_registry.csv",
        "runtime_benchmark.csv",
        "runtime_estimate_by_cell.csv",
        "runtime_estimate.json",
        "VALIDATION_REPORT.md",
    ]
    manifest = build_experiment_manifest(
        str(config["experiment_name"]),
        REPOSITORY_ROOT,
        {
            "optimization_seed_base": int(config["optimization_seed_base"]),
            "bootstrap_seed": int(config["bootstrap_seed"]),
        },
        config,
        CONFIG_PATH,
    )
    manifest["source_sha256"] = verified | {
        "runner": file_sha256(Path(__file__)),
        "step9_2_runner": file_sha256(
            REPOSITORY_ROOT / "experiments" / "run_step9_2_topology_uncertainty_qaoa.py"
        ),
        "qaoa_diagnostics": file_sha256(
            REPOSITORY_ROOT / "src" / "qaoa_diagnostics.py"
        ),
        "feasibility_qaoa": file_sha256(
            REPOSITORY_ROOT / "src" / "feasibility_qaoa.py"
        ),
    }
    manifest["artifact_sha256"] = {
        name: file_sha256(OUTPUT_DIRECTORY / name) for name in artifact_names
    }
    manifest["checkpoint_sha256"] = checkpoint_hashes
    manifest["claim_exclusions"] = [
        "quantum_advantage",
        "quantum_speedup",
        "useful_quantum_scale",
        "population_representativeness",
    ]
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["prepare", "benchmark", "run", "finalize", "all"], nargs="?", default="all"
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    config = _load_config()
    source = _source_directory(config)
    verified = _verify_sources(source)
    problems, _, _, _ = _preparation_tables(config, source)
    if args.mode in {"benchmark", "all"}:
        benchmark(problems, config)
    if args.mode in {"run", "all"}:
        estimate_path = OUTPUT_DIRECTORY / "runtime_estimate.json"
        if not estimate_path.is_file():
            raise RuntimeError("run the runtime benchmark before K=16 trials")
        workers = int(config["default_workers"] if args.workers is None else args.workers)
        if workers <= 0:
            raise ValueError("workers must be positive")
        run_trials(problems, config, workers=workers)
    if args.mode in {"finalize", "all"}:
        finalize(config, source, verified)
    if args.mode == "prepare":
        print(f"Prepared {len(problems)} K=16 problems.", flush=True)


if __name__ == "__main__":
    main()
