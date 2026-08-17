"""Benchmark standard penalty QAOA against one-hot XY-style QAOA."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.feasibility_qaoa import (
    circuit_resource_record,
    create_expectation_qnode,
    create_probability_qnode,
    finite_shot_resample,
    one_hot_basis_indices,
    optimize_qaoa_angles,
    summarize_probabilities,
)
from src.reproducibility import build_experiment_manifest, write_json
from src.risk_qubo import build_risk_qubo, risk_qubo_to_ising


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "qaoa_benchmark.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step7_5_qaoa"


def _configurations(config: dict[str, object], scores: pd.DataFrame) -> list[dict[str, object]]:
    configurations: list[dict[str, object]] = []
    primary_networks = sorted(
        scores.loc[scores["K"] == int(config["primary_candidate_count"]), "network_id"].unique()
    )
    for network_id in primary_networks:
        for depth in config["primary_depths"]:
            for seed in config["primary_seeds"]:
                for method in config["methods"]:
                    configurations.append(
                        {
                            "study": "topology_primary",
                            "network_id": network_id,
                            "K": int(config["primary_candidate_count"]),
                            "depth": int(depth),
                            "seed": int(seed),
                            "method": method,
                        }
                    )
    for k_value in config["scaling_candidate_counts"]:
        for depth in config["scaling_depths"]:
            for seed in config["scaling_seeds"]:
                for method in config["methods"]:
                    configurations.append(
                        {
                            "study": "candidate_scaling",
                            "network_id": config["scaling_network_id"],
                            "K": int(k_value),
                            "depth": int(depth),
                            "seed": int(seed),
                            "method": method,
                        }
                    )
    return configurations


def run() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source = REPOSITORY_ROOT / config["routing_risk_artifacts"]
    scores = pd.read_csv(source / "route_risk_scores.csv")
    scores = scores[scores["variant"] == config["risk_variant"]].copy()
    configurations = _configurations(config, scores)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    history_tables: list[pd.DataFrame] = []
    count_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for run_index, run_config in enumerate(configurations, start=1):
        network_id = str(run_config["network_id"])
        k = int(run_config["K"])
        depth = int(run_config["depth"])
        seed = int(run_config["seed"])
        method = str(run_config["method"])
        group = scores[(scores["network_id"] == network_id) & (scores["K"] == k)]
        group = group.sort_values(
            "route_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int)
        )
        if len(group) != k:
            raise ValueError(f"expected {k} risk rows for {network_id}, found {len(group)}")
        topology = str(group.iloc[0]["topology"])
        risk_vector = group["risk_score"].to_numpy(dtype=float)
        qubo = build_risk_qubo(risk_vector)
        cost_hamiltonian, offset, _, _ = risk_qubo_to_ising(qubo)
        optimal_index = int(np.argmin(risk_vector))

        expectation_qnode = create_expectation_qnode(
            cost_hamiltonian,
            number_of_qubits=k,
            depth=depth,
            method=method,
        )
        probability_qnode = create_probability_qnode(
            cost_hamiltonian,
            number_of_qubits=k,
            depth=depth,
            method=method,
        )
        run_started = time.perf_counter()
        optimized = optimize_qaoa_angles(
            expectation_qnode,
            depth=depth,
            seed=seed,
            iterations=int(config["iterations"]),
            step_size=float(config["step_size"]),
        )
        probabilities = np.asarray(
            probability_qnode(optimized.gammas, optimized.betas), dtype=float
        )
        summary = summarize_probabilities(
            probabilities, qubo, optimal_route_index=optimal_index
        )
        sampling_seed = int(config["sampling_seed_offset"]) + seed + 100 * k + depth
        counts = finite_shot_resample(
            probabilities,
            number_of_qubits=k,
            shots=int(config["shots"]),
            seed=sampling_seed,
        )
        valid_indices = one_hot_basis_indices(k)
        optimal_basis_index = valid_indices[optimal_index]
        valid_shots = sum(
            counts.get(format(index, f"0{k}b"), 0) for index in valid_indices
        )
        optimal_shots = counts.get(format(optimal_basis_index, f"0{k}b"), 0)
        resources = circuit_resource_record(
            probability_qnode, optimized.gammas, optimized.betas
        )
        elapsed = time.perf_counter() - run_started
        run_id = f"qaoa_{run_index:04d}"
        metric_rows.append(
            {
                "run_id": run_id,
                **run_config,
                "topology": topology,
                "iterations": int(config["iterations"]),
                "step_size": float(config["step_size"]),
                "shots": int(config["shots"]),
                "sampling_seed": sampling_seed,
                "exact_optimal_route_index": optimal_index,
                "exact_optimal_route_id": f"x_{optimal_index}",
                "exact_optimal_risk": float(risk_vector[optimal_index]),
                "initial_expected_qubo_energy": float(
                    optimized.history.iloc[0]["hamiltonian_expectation"] + offset
                ),
                "final_expected_qubo_energy": summary["expected_qubo_energy"],
                "analytic_valid_probability": summary["valid_probability"],
                "analytic_optimal_probability": summary["optimal_probability"],
                "most_probable_valid_route_index": summary[
                    "most_probable_valid_route_index"
                ],
                "most_probable_valid_bitstring": summary[
                    "most_probable_valid_bitstring"
                ],
                "sampled_valid_probability": valid_shots / int(config["shots"]),
                "sampled_optimal_probability": optimal_shots / int(config["shots"]),
                "gammas_json": json.dumps([float(value) for value in optimized.gammas]),
                "betas_json": json.dumps([float(value) for value in optimized.betas]),
                "wall_time_seconds": elapsed,
                **resources,
            }
        )
        history = optimized.history.copy()
        history.insert(0, "run_id", run_id)
        history.insert(1, "method", method)
        history.insert(2, "network_id", network_id)
        history.insert(3, "topology", topology)
        history.insert(4, "K", k)
        history.insert(5, "depth", depth)
        history.insert(6, "seed", seed)
        history["expected_qubo_energy"] = history["hamiltonian_expectation"] + offset
        history_tables.append(history)
        count_rows.extend(
            {
                "run_id": run_id,
                "method": method,
                "network_id": network_id,
                "topology": topology,
                "K": k,
                "depth": depth,
                "seed": seed,
                "bitstring": bitstring,
                "count": count,
            }
            for bitstring, count in counts.items()
        )
        print(
            f"[{run_index}/{len(configurations)}] {network_id} K={k} p={depth} "
            f"{method} seed={seed}: valid={summary['valid_probability']:.4f}, "
            f"optimal={summary['optimal_probability']:.4f}, {elapsed:.1f}s",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    histories = pd.concat(history_tables, ignore_index=True)
    counts_table = pd.DataFrame(count_rows)
    metrics.to_csv(OUTPUT_DIRECTORY / "qaoa_runs.csv", index=False)
    histories.to_csv(OUTPUT_DIRECTORY / "training_history.csv", index=False)
    counts_table.to_csv(OUTPUT_DIRECTORY / "measurement_counts.csv", index=False)
    aggregate = (
        metrics.groupby(["study", "K", "depth", "method"], sort=True)
        .agg(
            run_count=("run_id", "size"),
            mean_valid_probability=("analytic_valid_probability", "mean"),
            std_valid_probability=("analytic_valid_probability", "std"),
            mean_optimal_probability=("analytic_optimal_probability", "mean"),
            std_optimal_probability=("analytic_optimal_probability", "std"),
            mean_expected_energy=("final_expected_qubo_energy", "mean"),
            mean_circuit_depth=("circuit_depth", "mean"),
            mean_gate_count=("gate_count", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(OUTPUT_DIRECTORY / "qaoa_aggregate.csv", index=False)
    topology_aggregate = (
        metrics[metrics["study"] == "topology_primary"]
        .groupby(["topology", "depth", "method"], sort=True)
        .agg(
            run_count=("run_id", "size"),
            mean_valid_probability=("analytic_valid_probability", "mean"),
            mean_optimal_probability=("analytic_optimal_probability", "mean"),
            std_optimal_probability=("analytic_optimal_probability", "std"),
        )
        .reset_index()
    )
    topology_aggregate.to_csv(
        OUTPUT_DIRECTORY / "qaoa_by_topology.csv", index=False
    )

    manifest = build_experiment_manifest(
        experiment_name=config["experiment_name"],
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "optimization_min": min(config["primary_seeds"]),
            "optimization_max": max(config["primary_seeds"]),
            "sampling_offset": int(config["sampling_seed_offset"]),
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest["total_runs"] = len(metrics)
    manifest["total_wall_time_seconds"] = time.perf_counter() - start_time
    manifest["warm_start_implemented"] = False
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    print(aggregate.to_string(index=False), flush=True)
    print(f"Artifacts: {OUTPUT_DIRECTORY}", flush=True)


if __name__ == "__main__":
    run()
