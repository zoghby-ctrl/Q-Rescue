"""Generate K routes, predict intervals, and run risk-objective ablations."""

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

from src.candidate_routes import (
    generate_candidate_routes,
    pairwise_route_diversity_table,
    summarize_route_diversity,
)
from src.conformal import calibrate_locally_scaled_split_conformal
from src.deadline_risk import RiskWeights, score_route_table
from src.reproducibility import build_experiment_manifest, write_json
from src.route_features import (
    RouteTrafficState,
    build_route_feature_table,
    estimate_pretrip_speed_kph,
)
from src.synthetic_networks import SyntheticNetworkSpec, generate_synthetic_network


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "routing_risk_benchmark.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "steps6_6_5_routing_risk"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _route_state(route: object, *, seed: int) -> RouteTrafficState:
    rng = np.random.default_rng(seed)
    congestion = float(rng.uniform(0.20, 0.82))
    rain = int(rng.random() < 0.35)
    incident = int(rng.random() < 0.22)
    historical_delay = float(rng.gamma(shape=1.5, scale=0.7))
    free_flow_speed = 60.0 * route.distance_km / route.base_travel_time_min
    sensor_adjustment = float(rng.normal(0.0, 0.03 * free_flow_speed))
    speed = estimate_pretrip_speed_kph(
        route.distance_km,
        route.base_travel_time_min,
        congestion,
        rain,
        incident,
        sensor_adjustment_kph=sensor_adjustment,
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


def run() -> None:
    config = _read_json(CONFIG_PATH)
    network_config = _read_json(REPOSITORY_ROOT / str(config["network_config"]))
    step4 = REPOSITORY_ROOT / str(config["step4_artifacts"])
    step5 = REPOSITORY_ROOT / str(config["step5_artifacts"])
    base_model = joblib.load(step4 / "best_model.joblib")
    scale_model = joblib.load(step5 / "residual_scale_model.joblib")
    calibration_data = pd.read_csv(step4 / "calibration.csv")
    conformal_model = calibrate_locally_scaled_split_conformal(
        base_model,
        scale_model,
        calibration_data,
        alpha=float(config["alpha"]),
    )
    distance_values = calibration_data["distance_normalization_km"].unique()
    if len(distance_values) != 1:
        raise ValueError("Step 4 must record exactly one distance normalization constant")
    distance_normalization_km = float(distance_values[0])
    weights = RiskWeights(**dict(config["risk_weights"]))

    candidate_rows: list[pd.DataFrame] = []
    score_rows: list[pd.DataFrame] = []
    decision_rows: list[dict[str, object]] = []
    diversity_rows: list[dict[str, object]] = []
    pairwise_rows: list[pd.DataFrame] = []
    maximum_k = max(int(value) for value in config["candidate_counts"])

    for network_index, entry in enumerate(network_config["networks"]):
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
        all_routes = generate_candidate_routes(
            graph,
            graph.graph["origin"],
            graph.graph["destination"],
            maximum_k,
        )
        all_states = {
            route.route_id: _route_state(
                route,
                seed=int(config["state_seed"]) + network_index * 100 + route.route_index,
            )
            for route in all_routes
        }

        for k_value in config["candidate_counts"]:
            k = int(k_value)
            routes = all_routes[:k]
            states = {route.route_id: all_states[route.route_id] for route in routes}
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
            intervals.insert(1, "topology", topology)
            intervals.insert(2, "K", k)
            intervals["deadline_minutes"] = deadline
            intervals["reroute_penalty_minutes"] = np.where(
                intervals["route_id"] == "x_0",
                0.0,
                float(config["reroute_penalty_minutes"]),
            )
            candidate_rows.append(intervals)

            for variant in "ABCDE":
                scored = score_route_table(
                    intervals,
                    deadline,
                    weights=weights,
                    variant=variant,
                )
                scored.insert(3, "variant", variant)
                scored["conservative_deadline_violation"] = scored["upper_bound"] > deadline
                score_rows.append(scored)
                chosen = scored.sort_values(
                    ["risk_score", "route_id"], kind="stable"
                ).iloc[0]
                decision_rows.append(
                    {
                        "network_id": network_id,
                        "topology": topology,
                        "K": k,
                        "variant": variant,
                        "selected_route_id": chosen["route_id"],
                        "prediction": float(chosen["prediction"]),
                        "interval_half_width": float(chosen["interval_half_width"]),
                        "lower_bound": float(chosen["lower_bound"]),
                        "upper_bound": float(chosen["upper_bound"]),
                        "deadline_minutes": deadline,
                        "deadline_margin": float(chosen["deadline_margin"]),
                        "conservative_deadline_violation": bool(
                            chosen["conservative_deadline_violation"]
                        ),
                        "risk_score": float(chosen["risk_score"]),
                    }
                )

            summary = summarize_route_diversity(routes)
            diversity_rows.append(
                {"network_id": network_id, "topology": topology, "K": k, **summary}
            )
            pairwise = pairwise_route_diversity_table(routes)
            pairwise.insert(0, "network_id", network_id)
            pairwise.insert(1, "topology", topology)
            pairwise.insert(2, "K", k)
            pairwise_rows.append(pairwise)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    candidates = pd.concat(candidate_rows, ignore_index=True)
    scores = pd.concat(score_rows, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)
    candidates.to_csv(OUTPUT_DIRECTORY / "candidate_route_predictions.csv", index=False)
    scores.to_csv(OUTPUT_DIRECTORY / "route_risk_scores.csv", index=False)
    decisions.to_csv(OUTPUT_DIRECTORY / "selected_routes_by_ablation.csv", index=False)
    pd.DataFrame(diversity_rows).to_csv(
        OUTPUT_DIRECTORY / "route_diversity_summary.csv", index=False
    )
    pd.concat(pairwise_rows, ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "route_diversity_pairwise.csv", index=False
    )

    expected_choices = decisions[decisions["variant"] == "A"].set_index(
        ["network_id", "K"]
    )["selected_route_id"]
    summary_rows: list[dict[str, object]] = []
    for variant, group in decisions.groupby("variant", sort=True):
        changed = sum(
            row.selected_route_id != expected_choices.loc[(row.network_id, row.K)]
            for row in group.itertuples()
        )
        summary_rows.append(
            {
                "variant": variant,
                "decision_count": len(group),
                "changed_from_expected_time_count": changed,
                "conservative_deadline_violation_count": int(
                    group["conservative_deadline_violation"].sum()
                ),
                "mean_deadline_margin": float(group["deadline_margin"].mean()),
                "mean_selected_interval_half_width": float(
                    group["interval_half_width"].mean()
                ),
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(OUTPUT_DIRECTORY / "ablation_summary.csv", index=False)

    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={"traffic_state": int(config["state_seed"])},
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest["conformal_calibration_quantile"] = conformal_model.calibration_quantile
    manifest["distance_normalization_km"] = distance_normalization_km
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)

    print(summary_table.to_string(index=False))
    print(f"Candidate decisions: {len(decisions)} across 5 networks and K=4,6,8")
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run()
