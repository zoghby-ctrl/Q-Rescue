"""Run rotating-event dynamic routing with four controlled policies."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.conformal import calibrate_locally_scaled_split_conformal
from src.deadline_risk import RiskWeights
from src.dynamic_rerouting import DynamicEvent, simulate_dynamic_route
from src.feasibility_qaoa import (
    create_expectation_qnode,
    create_probability_qnode,
    optimize_qaoa_angles,
    summarize_probabilities,
)
from src.reproducibility import build_experiment_manifest, write_json
from src.risk_qubo import build_risk_qubo, risk_qubo_to_ising
from src.synthetic_networks import SyntheticNetworkSpec, generate_synthetic_network


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "dynamic_rerouting_benchmark.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step8_dynamic_rerouting"


def _event_catalog(graph: nx.DiGraph) -> list[DynamicEvent]:
    base_path = nx.shortest_path(
        graph, graph.graph["origin"], graph.graph["destination"], weight="base_travel_time_min"
    )
    path_edges = list(zip(base_path[:-1], base_path[1:]))
    first_edge = path_edges[0]
    middle_edge = path_edges[len(path_edges) // 2]
    return [
        DynamicEvent("normal", global_congestion=0.20, actual_noise_fraction=0.03),
        DynamicEvent("rush_hour", global_congestion=0.78, actual_noise_fraction=0.05),
        DynamicEvent("rain", global_congestion=0.45, rain_indicator=1, actual_noise_fraction=0.06),
        DynamicEvent(
            "localized_congestion",
            global_congestion=0.28,
            localized_edges=frozenset({middle_edge}),
            localized_congestion_boost=0.65,
            actual_noise_fraction=0.05,
        ),
        DynamicEvent(
            "accident",
            global_congestion=0.42,
            incident_edges=frozenset({middle_edge}),
            actual_noise_fraction=0.10,
        ),
        DynamicEvent(
            "road_closure",
            global_congestion=0.35,
            closed_edges=frozenset({first_edge}),
            actual_noise_fraction=0.05,
        ),
        DynamicEvent(
            "prediction_noise",
            global_congestion=0.50,
            actual_noise_fraction=0.25,
        ),
    ]


class DynamicQAOASelector:
    """Actual ideal-simulator feasible-XY selector used at dynamic decisions."""

    def __init__(self, config: dict[str, object], scenario_seed: int) -> None:
        self.config = config
        self.scenario_seed = scenario_seed
        self.call_index = 0
        self.records: list[dict[str, object]] = []

    def __call__(self, candidates: pd.DataFrame) -> str:
        ordered = candidates.sort_values(
            "route_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int)
        )
        risk_vector = ordered["risk_score"].to_numpy(dtype=float)
        if len(ordered) == 1:
            return str(ordered.iloc[0]["route_id"])
        qubo = build_risk_qubo(risk_vector)
        hamiltonian, _, _, _ = risk_qubo_to_ising(qubo)
        depth = int(self.config["qaoa_depth"])
        seed = int(self.config["qaoa_seed"]) + self.scenario_seed * 100 + self.call_index
        expectation = create_expectation_qnode(
            hamiltonian,
            number_of_qubits=len(ordered),
            depth=depth,
            method="feasible_xy",
        )
        probability = create_probability_qnode(
            hamiltonian,
            number_of_qubits=len(ordered),
            depth=depth,
            method="feasible_xy",
        )
        optimized = optimize_qaoa_angles(
            expectation,
            depth=depth,
            seed=seed,
            iterations=int(self.config["qaoa_iterations"]),
            step_size=float(self.config["qaoa_step_size"]),
        )
        probabilities = probability(optimized.gammas, optimized.betas)
        optimal_index = int(np.argmin(risk_vector))
        summary = summarize_probabilities(
            probabilities, qubo, optimal_route_index=optimal_index
        )
        selected_index = int(summary["most_probable_valid_route_index"])
        selected_route = str(ordered.iloc[selected_index]["route_id"])
        self.records.append(
            {
                "qaoa_call_index": self.call_index,
                "seed": seed,
                "candidate_count": len(ordered),
                "exact_risk_route_id": str(ordered.iloc[optimal_index]["route_id"]),
                "selected_route_id": selected_route,
                "matched_exact_risk": selected_index == optimal_index,
                "analytic_valid_probability": summary["valid_probability"],
                "analytic_optimal_probability": summary["optimal_probability"],
                "final_expected_qubo_energy": summary["expected_qubo_energy"],
            }
        )
        self.call_index += 1
        return selected_route


def run() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    network_config = json.loads(
        (REPOSITORY_ROOT / config["network_config"]).read_text(encoding="utf-8")
    )
    step4 = REPOSITORY_ROOT / config["step4_artifacts"]
    step5 = REPOSITORY_ROOT / config["step5_artifacts"]
    base_model = joblib.load(step4 / "best_model.joblib")
    scale_model = joblib.load(step5 / "residual_scale_model.joblib")
    calibration = pd.read_csv(step4 / "calibration.csv")
    interval_model = calibrate_locally_scaled_split_conformal(
        base_model, scale_model, calibration, alpha=0.10
    )
    distance_normalization = float(calibration["distance_normalization_km"].iloc[0])
    weights = RiskWeights(**config["risk_weights"])

    summary_rows: list[dict[str, object]] = []
    decision_tables: list[pd.DataFrame] = []
    qaoa_rows: list[dict[str, object]] = []
    total_cases = len(network_config["networks"]) * 7 * len(config["policies"])
    case_index = 0
    started = time.perf_counter()
    for network_index, entry in enumerate(network_config["networks"]):
        graph = generate_synthetic_network(
            SyntheticNetworkSpec(entry["topology"], entry["seed"], entry["parameters"])
        )
        network_id = entry["network_id"]
        topology = entry["topology"]
        base_shortest_time = nx.shortest_path_length(
            graph,
            graph.graph["origin"],
            graph.graph["destination"],
            weight="base_travel_time_min",
        )
        deadline = float(
            base_shortest_time * config["deadline_multiplier_over_base_shortest_time"]
        )
        catalog = _event_catalog(graph)
        for rotation in range(len(catalog)):
            events = catalog[rotation:] + catalog[:rotation]
            scenario_name = f"starts_{events[0].name}"
            scenario_seed = network_index * 10 + rotation
            for policy in config["policies"]:
                selector = (
                    DynamicQAOASelector(config, scenario_seed)
                    if policy == "uncertainty_aware_qaoa"
                    else None
                )
                result = simulate_dynamic_route(
                    graph,
                    graph.graph["origin"],
                    graph.graph["destination"],
                    events,
                    policy=policy,
                    k=int(config["candidate_count"]),
                    deadline_minutes=deadline,
                    point_interval_model=interval_model,
                    distance_normalization_km=distance_normalization,
                    risk_weights=weights,
                    reroute_penalty_minutes=float(config["reroute_penalty_minutes"]),
                    random_seed=int(config["simulation_seed"]) + scenario_seed,
                    quantum_selector=selector,
                )
                case_index += 1
                run_id = f"dynamic_{case_index:04d}"
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "network_id": network_id,
                        "topology": topology,
                        "scenario": scenario_name,
                        "first_event": events[0].name,
                        "deadline_minutes": deadline,
                        **result.summary(),
                    }
                )
                decisions = result.decisions.copy()
                decisions.insert(0, "run_id", run_id)
                decisions.insert(1, "network_id", network_id)
                decisions.insert(2, "topology", topology)
                decisions.insert(3, "scenario", scenario_name)
                decisions.insert(4, "policy", policy)
                decision_tables.append(decisions)
                if selector is not None:
                    qaoa_rows.extend(
                        {
                            "run_id": run_id,
                            "network_id": network_id,
                            "topology": topology,
                            "scenario": scenario_name,
                            **record,
                        }
                        for record in selector.records
                    )
                print(
                    f"[{case_index}/{total_cases}] {network_id} {scenario_name} {policy}: "
                    f"arrived={result.arrived}, late={result.deadline_violation}, "
                    f"reroutes={result.reroute_count}",
                    flush=True,
                )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summaries = pd.DataFrame(summary_rows)
    decisions = pd.concat(decision_tables, ignore_index=True, sort=False)
    qaoa_decisions = pd.DataFrame(qaoa_rows)
    summaries.to_csv(OUTPUT_DIRECTORY / "dynamic_runs.csv", index=False)
    decisions.to_csv(OUTPUT_DIRECTORY / "dynamic_decisions.csv", index=False)
    qaoa_decisions.to_csv(OUTPUT_DIRECTORY / "dynamic_qaoa_decisions.csv", index=False)
    policy_summary = (
        summaries.groupby("policy", sort=True)
        .agg(
            run_count=("run_id", "size"),
            arrival_rate=("arrived", "mean"),
            deadline_violation_rate=("deadline_violation", "mean"),
            mean_arrival_time=("arrival_time_minutes", "mean"),
            mean_deadline_margin=("deadline_margin_minutes", "mean"),
            mean_reroutes=("reroute_count", "mean"),
            mean_unnecessary_reroutes=("unnecessary_reroutes", "mean"),
            mean_route_stability=("route_stability", "mean"),
        )
        .reset_index()
    )
    policy_summary.to_csv(OUTPUT_DIRECTORY / "dynamic_policy_summary.csv", index=False)
    by_topology = (
        summaries.groupby(["topology", "policy"], sort=True)
        .agg(
            run_count=("run_id", "size"),
            arrival_rate=("arrived", "mean"),
            deadline_violation_rate=("deadline_violation", "mean"),
            mean_deadline_margin=("deadline_margin_minutes", "mean"),
            mean_reroutes=("reroute_count", "mean"),
        )
        .reset_index()
    )
    by_topology.to_csv(OUTPUT_DIRECTORY / "dynamic_by_topology.csv", index=False)

    manifest = build_experiment_manifest(
        experiment_name=config["experiment_name"],
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "simulation": int(config["simulation_seed"]),
            "qaoa": int(config["qaoa_seed"]),
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest["total_wall_time_seconds"] = time.perf_counter() - started
    manifest["qaoa_selector"] = "p=1 feasible_xy ideal simulator, 12 Adam iterations"
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    print(policy_summary.to_string(index=False), flush=True)
    print(f"Artifacts: {OUTPUT_DIRECTORY}", flush=True)


if __name__ == "__main__":
    run()
