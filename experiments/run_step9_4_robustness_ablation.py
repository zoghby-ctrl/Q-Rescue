"""Reconcile inherited evidence and run the missing Step 9.4 ablations.

The runner deliberately keeps static proxy, controlled dynamic simulation,
QUBO ground-state, and ideal-simulator QAOA scopes separate.  It does not
report runtime comparisons and it does not synthesize a warm-start result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.conformal import (  # noqa: E402
    INTERVAL_COLUMNS,
    calibrate_locally_scaled_split_conformal,
)
from src.deadline_risk import RiskWeights, score_route_table  # noqa: E402
from src.dynamic_rerouting import DynamicEvent, simulate_dynamic_route  # noqa: E402
from src.feasibility_qaoa import (  # noqa: E402
    circuit_resource_record,
    create_expectation_qnode,
    create_probability_qnode,
    optimize_qaoa_angles,
    summarize_probabilities,
)
from src.reproducibility import (  # noqa: E402
    build_experiment_manifest,
    file_sha256,
    write_json,
)
from src.risk_qubo import build_risk_qubo, risk_qubo_to_ising  # noqa: E402
from src.synthetic_networks import (  # noqa: E402
    SyntheticNetworkSpec,
    generate_synthetic_network,
)
from src.traffic_model import FEATURE_COLUMNS  # noqa: E402


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "robustness_ablation_benchmark.json"


@dataclass(frozen=True)
class TableProvenance:
    generation_mode: str
    scope: str
    comparability_group: str
    sources: tuple[Path, ...]
    notes: str


class PointOnlyIntervalModel:
    """Expose point predictions as degenerate intervals for one ablation.

    This is not conformal inference.  It intentionally removes both calibrated
    uncertainty width and the conservative upper-bound deadline proxy.
    """

    def __init__(self, base_model: object) -> None:
        self.base_model = base_model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        predictions = np.asarray(self.base_model.predict(data), dtype=float)
        if np.any(predictions < 0.0) or not np.isfinite(predictions).all():
            raise ValueError("point-only dynamic predictions must be finite and non-negative")
        metadata = [
            column
            for column in data.columns
            if column not in FEATURE_COLUMNS and column not in INTERVAL_COLUMNS
        ]
        result = data.loc[:, metadata].reset_index(drop=True).copy()
        result["prediction"] = predictions
        result["lower_bound"] = predictions
        result["upper_bound"] = predictions
        result["interval_half_width"] = 0.0
        return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _route_sort(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values(
        "route_id",
        key=lambda values: values.str.extract(r"(\d+)")[0].astype(int),
        kind="stable",
    )


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval needs at least one observation")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _coverage_annotations(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    lows: list[float] = []
    highs: list[float] = []
    for row in result.itertuples(index=False):
        total = int(row.n_samples)
        successes = int(round(float(row.empirical_coverage) * total))
        low, high = _wilson_interval(successes, total)
        lows.append(low)
        highs.append(high)
    result["coverage_gap_empirical_minus_nominal"] = (
        result["empirical_coverage"] - result["nominal_coverage"]
    )
    result["wilson_95_lower"] = lows
    result["wilson_95_upper"] = highs
    result["observed_undercoverage_beyond_wilson_95"] = (
        result["wilson_95_upper"] < result["nominal_coverage"]
    )
    return result


def _weights_from_dict(values: dict[str, float]) -> RiskWeights:
    return RiskWeights(
        lambda_uncertainty=float(values["lambda_uncertainty"]),
        lambda_deadline=float(values["lambda_deadline"]),
        lambda_reroute=float(values["lambda_reroute"]),
    )


def _condition_parameters(
    condition: str,
    baseline: dict[str, float],
) -> tuple[RiskWeights, bool, bool]:
    values = dict(baseline)
    point_only = condition == "without_conformal_calibration"
    zero_reroute_input = condition == "without_reroute_penalty"
    if condition == "without_uncertainty":
        values["lambda_uncertainty"] = 0.0
    elif condition == "without_deadline_penalty":
        values["lambda_deadline"] = 0.0
    elif condition == "without_reroute_penalty":
        values["lambda_reroute"] = 0.0
    elif condition not in {"full", "without_conformal_calibration"}:
        raise ValueError(f"unsupported ablation condition {condition!r}")
    return _weights_from_dict(values), point_only, zero_reroute_input


def _risk_ablation_tables(
    candidates: pd.DataFrame,
    source_decisions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    static_config = dict(config["static_risk"])
    baseline = dict(static_config["baseline_weights"])
    score_tables: list[pd.DataFrame] = []
    decision_rows: list[dict[str, Any]] = []

    for condition in static_config["ablation_conditions"]:
        weights, point_only, _ = _condition_parameters(str(condition), baseline)
        for keys, group in candidates.groupby(["network_id", "topology", "K"], sort=True):
            network_id, topology, k_value = keys
            working = group.copy()
            deadline_values = working["deadline_minutes"].unique()
            if len(deadline_values) != 1:
                raise ValueError(f"non-unique deadline for {network_id} K={k_value}")
            deadline = float(deadline_values[0])
            if point_only:
                working["lower_bound"] = working["prediction"]
                working["upper_bound"] = working["prediction"]
                working["interval_half_width"] = 0.0
            scored = score_route_table(
                working,
                deadline,
                weights=weights,
                variant="E",
            )
            scored.insert(0, "condition", condition)
            scored.insert(1, "evidence_scope", "static_pretrip_deadline_proxy")
            scored["conformal_calibration_used"] = not point_only
            scored["risk_weights_json"] = json.dumps(
                {
                    "lambda_uncertainty": weights.lambda_uncertainty,
                    "lambda_deadline": weights.lambda_deadline,
                    "lambda_reroute": weights.lambda_reroute,
                },
                sort_keys=True,
            )
            scored["deadline_proxy_violation"] = scored["upper_bound"] > deadline
            score_tables.append(scored)
            chosen = scored.sort_values(["risk_score", "route_id"], kind="stable").iloc[0]
            decision_rows.append(
                {
                    "condition": condition,
                    "evidence_scope": "static_pretrip_deadline_proxy",
                    "network_id": network_id,
                    "topology": topology,
                    "K": int(k_value),
                    "selected_route_id": chosen["route_id"],
                    "prediction": float(chosen["prediction"]),
                    "lower_bound": float(chosen["lower_bound"]),
                    "upper_bound": float(chosen["upper_bound"]),
                    "interval_half_width": float(chosen["interval_half_width"]),
                    "deadline_minutes": deadline,
                    "deadline_margin": float(chosen["deadline_margin"]),
                    "deadline_proxy_violation": bool(chosen["deadline_proxy_violation"]),
                    "reroute_penalty": float(chosen["reroute_penalty"]),
                    "risk_score": float(chosen["risk_score"]),
                    "conformal_calibration_used": not point_only,
                    "risk_weights_json": scored.iloc[0]["risk_weights_json"],
                }
            )

    scores = pd.concat(score_tables, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)
    key_columns = ["network_id", "topology", "K"]
    full = decisions[decisions["condition"] == "full"].set_index(key_columns)
    decisions["full_selected_route_id"] = [
        full.loc[(row.network_id, row.topology, row.K), "selected_route_id"]
        for row in decisions.itertuples()
    ]
    decisions["changed_from_full"] = (
        decisions["selected_route_id"] != decisions["full_selected_route_id"]
    )

    summary_rows: list[dict[str, Any]] = []
    for condition, group in decisions.groupby("condition", sort=True):
        for scope, subset in [
            ("overall", group),
            *[(f"K={k}", group[group["K"] == k]) for k in sorted(group["K"].unique())],
        ]:
            summary_rows.append(
                {
                    "condition": condition,
                    "aggregation_scope": scope,
                    "decision_count": len(subset),
                    "selection_change_count_from_full": int(subset["changed_from_full"].sum()),
                    "selection_change_rate_from_full": float(subset["changed_from_full"].mean()),
                    "deadline_proxy_violation_count": int(
                        subset["deadline_proxy_violation"].sum()
                    ),
                    "deadline_proxy_violation_rate": float(
                        subset["deadline_proxy_violation"].mean()
                    ),
                    "mean_selected_deadline_margin_minutes": float(
                        subset["deadline_margin"].mean()
                    ),
                    "mean_selected_risk_score": float(subset["risk_score"].mean()),
                    "outcome_type": "proxy_not_realized_arrival",
                }
            )
    summary = pd.DataFrame(summary_rows)

    expected = source_decisions[source_decisions["variant"] == "E"].loc[
        :, ["network_id", "topology", "K", "selected_route_id"]
    ]
    actual = decisions[decisions["condition"] == "full"].loc[
        :, ["network_id", "topology", "K", "selected_route_id"]
    ]
    reconciliation = expected.merge(
        actual,
        on=["network_id", "topology", "K"],
        suffixes=("_source_E", "_recomputed_full"),
        validate="one_to_one",
    )
    reconciliation["selection_matches"] = (
        reconciliation["selected_route_id_source_E"]
        == reconciliation["selected_route_id_recomputed_full"]
    )
    checks = {
        "static_full_source_E_decisions": len(reconciliation),
        "static_full_matches_source_E": bool(reconciliation["selection_matches"].all()),
        "static_candidate_counts": sorted(int(value) for value in decisions["K"].unique()),
    }
    return scores, decisions, summary, checks


def _inherited_risk_variants(source: pd.DataFrame) -> pd.DataFrame:
    semantics = {
        "A": "expected time only; uncertainty, deadline, and reroute terms absent",
        "B": "expected time plus uncertainty; deadline and reroute terms absent",
        "C": "expected time plus deadline; uncertainty and reroute terms absent",
        "D": "expected time plus uncertainty and deadline; reroute term absent",
        "E": "full expected time, uncertainty, deadline, and reroute score",
    }
    result = source.copy()
    result.insert(0, "evidence_scope", "inherited_step6_static_factorial_variants")
    result["variant_semantics"] = result["variant"].map(semantics)
    result["causal_isolation_caveat"] = (
        "Variants A-D remove more than one component relative to E in some comparisons; "
        "use the targeted one-term table for isolated removal effects."
    )
    return result


def _alpha_sweep_tables(
    candidates: pd.DataFrame,
    calibration_quantiles: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    static_config = dict(config["static_risk"])
    baseline_alpha = float(static_config["baseline_alpha"])
    local = calibration_quantiles[calibration_quantiles["method"] == "local_scaled"]
    quantiles = {
        float(row.alpha): float(row.calibration_residual_quantile)
        for row in local.itertuples()
    }
    if baseline_alpha not in quantiles:
        raise ValueError("baseline local-scaled conformal quantile is missing")
    baseline_quantile = quantiles[baseline_alpha]
    weights = _weights_from_dict(dict(static_config["baseline_weights"]))
    decision_rows: list[dict[str, Any]] = []

    for alpha in (float(value) for value in static_config["alphas"]):
        if alpha not in quantiles:
            raise ValueError(f"local-scaled quantile missing for alpha={alpha}")
        multiplier = quantiles[alpha] / baseline_quantile
        for keys, group in candidates.groupby(["network_id", "topology", "K"], sort=True):
            network_id, topology, k_value = keys
            working = group.copy()
            half_width = working["interval_half_width"].to_numpy(dtype=float) * multiplier
            predictions = working["prediction"].to_numpy(dtype=float)
            working["lower_bound"] = np.maximum(0.0, predictions - half_width)
            working["upper_bound"] = predictions + half_width
            working["interval_half_width"] = half_width
            deadline = float(working["deadline_minutes"].iloc[0])
            scored = score_route_table(
                working,
                deadline,
                weights=weights,
                variant="E",
            )
            scored["deadline_proxy_violation"] = scored["upper_bound"] > deadline
            chosen = scored.sort_values(["risk_score", "route_id"], kind="stable").iloc[0]
            decision_rows.append(
                {
                    "alpha": alpha,
                    "nominal_coverage": 1.0 - alpha,
                    "calibration_quantile": quantiles[alpha],
                    "width_multiplier_vs_alpha_0_10": multiplier,
                    "deadline_policy": "fixed_at_source_alpha_0_10_to_isolate_interval_width",
                    "network_id": network_id,
                    "topology": topology,
                    "K": int(k_value),
                    "selected_route_id": chosen["route_id"],
                    "prediction": float(chosen["prediction"]),
                    "interval_half_width": float(chosen["interval_half_width"]),
                    "upper_bound": float(chosen["upper_bound"]),
                    "deadline_minutes": deadline,
                    "deadline_margin": float(chosen["deadline_margin"]),
                    "deadline_proxy_violation": bool(chosen["deadline_proxy_violation"]),
                    "risk_score": float(chosen["risk_score"]),
                }
            )

    decisions = pd.DataFrame(decision_rows)
    keys = ["network_id", "topology", "K"]
    baseline = decisions[np.isclose(decisions["alpha"], baseline_alpha)].set_index(keys)
    decisions["alpha_0_10_selected_route_id"] = [
        baseline.loc[(row.network_id, row.topology, row.K), "selected_route_id"]
        for row in decisions.itertuples()
    ]
    decisions["changed_from_alpha_0_10"] = (
        decisions["selected_route_id"] != decisions["alpha_0_10_selected_route_id"]
    )

    summary_rows: list[dict[str, Any]] = []
    for alpha, group in decisions.groupby("alpha", sort=True):
        for scope, subset in [
            ("overall", group),
            *[(f"K={k}", group[group["K"] == k]) for k in sorted(group["K"].unique())],
        ]:
            summary_rows.append(
                {
                    "alpha": float(alpha),
                    "nominal_coverage": 1.0 - float(alpha),
                    "aggregation_scope": scope,
                    "decision_count": len(subset),
                    "selection_change_count_from_alpha_0_10": int(
                        subset["changed_from_alpha_0_10"].sum()
                    ),
                    "selection_change_rate_from_alpha_0_10": float(
                        subset["changed_from_alpha_0_10"].mean()
                    ),
                    "deadline_proxy_violation_rate": float(
                        subset["deadline_proxy_violation"].mean()
                    ),
                    "mean_selected_interval_half_width": float(
                        subset["interval_half_width"].mean()
                    ),
                    "mean_selected_deadline_margin_minutes": float(
                        subset["deadline_margin"].mean()
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    checks = {
        "alpha_values": sorted(float(value) for value in decisions["alpha"].unique()),
        "alpha_0_10_decision_count": int(len(baseline)),
    }
    return decisions, summary, checks


def _weight_sweep_tables(
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    static_config = dict(config["static_risk"])
    baseline_values = {
        name: float(value)
        for name, value in dict(static_config["baseline_weights"]).items()
    }
    decision_rows: list[dict[str, Any]] = []
    sweep = dict(static_config["one_at_a_time_weight_sweep"])
    for parameter, values in sweep.items():
        for value in values:
            weight_values = dict(baseline_values)
            weight_values[str(parameter)] = float(value)
            weights = _weights_from_dict(weight_values)
            for keys, group in candidates.groupby(
                ["network_id", "topology", "K"], sort=True
            ):
                network_id, topology, k_value = keys
                deadline = float(group["deadline_minutes"].iloc[0])
                scored = score_route_table(group, deadline, weights=weights, variant="E")
                scored["deadline_proxy_violation"] = scored["upper_bound"] > deadline
                chosen = scored.sort_values(["risk_score", "route_id"], kind="stable").iloc[0]
                decision_rows.append(
                    {
                        "sweep_design": "one_at_a_time",
                        "sweep_parameter": parameter,
                        "sweep_value": float(value),
                        "baseline_value": baseline_values[str(parameter)],
                        "network_id": network_id,
                        "topology": topology,
                        "K": int(k_value),
                        "selected_route_id": chosen["route_id"],
                        "deadline_proxy_violation": bool(chosen["deadline_proxy_violation"]),
                        "deadline_margin": float(chosen["deadline_margin"]),
                        "risk_score": float(chosen["risk_score"]),
                        "risk_weights_json": json.dumps(weight_values, sort_keys=True),
                    }
                )
    decisions = pd.DataFrame(decision_rows)
    key_columns = ["network_id", "topology", "K"]
    baseline_selection = (
        decisions[
            (decisions["sweep_parameter"] == "lambda_uncertainty")
            & np.isclose(
                decisions["sweep_value"], baseline_values["lambda_uncertainty"]
            )
        ]
        .set_index(key_columns)["selected_route_id"]
    )
    decisions["baseline_selected_route_id"] = [
        baseline_selection.loc[(row.network_id, row.topology, row.K)]
        for row in decisions.itertuples()
    ]
    decisions["changed_from_baseline_weights"] = (
        decisions["selected_route_id"] != decisions["baseline_selected_route_id"]
    )

    rows: list[dict[str, Any]] = []
    for (parameter, value), group in decisions.groupby(
        ["sweep_parameter", "sweep_value"], sort=True
    ):
        for scope, subset in [
            ("overall", group),
            *[(f"K={k}", group[group["K"] == k]) for k in sorted(group["K"].unique())],
        ]:
            rows.append(
                {
                    "sweep_design": "one_at_a_time",
                    "sweep_parameter": parameter,
                    "sweep_value": float(value),
                    "aggregation_scope": scope,
                    "decision_count": len(subset),
                    "selection_change_count_from_baseline_weights": int(
                        subset["changed_from_baseline_weights"].sum()
                    ),
                    "selection_change_rate_from_baseline_weights": float(
                        subset["changed_from_baseline_weights"].mean()
                    ),
                    "deadline_proxy_violation_rate": float(
                        subset["deadline_proxy_violation"].mean()
                    ),
                    "mean_selected_deadline_margin_minutes": float(
                        subset["deadline_margin"].mean()
                    ),
                }
            )
    return decisions, pd.DataFrame(rows)


def _conformal_scope_coverage(
    step5_overall: pd.DataFrame,
    repeated_overall: pd.DataFrame,
    real_overall: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    step5 = step5_overall.copy()
    step5.insert(0, "source_scope", "step5_original_synthetic_networks")
    step5["population"] = step5["evaluation_set"].map(
        {
            "test_id": "held_out_rows_in_distribution",
            "test_ood_traffic": "controlled_traffic_distribution_shift",
        }
    )
    step5["source_method_label"] = step5["method"]

    repeated = repeated_overall.copy()
    repeated.insert(0, "source_scope", "step8_25_repeated_synthetic_topologies")
    repeated["population"] = "held_out_rows_known_networks"
    repeated["evaluation_set"] = "repeated_synthetic_test"
    repeated["method"] = "local_scaled"
    repeated["source_method_label"] = "locally_scaled_model_trained_in_step8_25_scope"

    real = real_overall.copy()
    real.insert(0, "source_scope", "step8_5_real_road_topology_controlled_simulated_traffic")
    real["population"] = "real_road_topology_with_controlled_simulated_traffic_shift"
    real["evaluation_set"] = "real_topology_controlled_simulated_traffic"
    real["method"] = "local_scaled"
    real["source_method_label"] = "frozen_step4_step5_local_scaled"

    columns = [
        "source_scope",
        "population",
        "evaluation_set",
        "method",
        "source_method_label",
        "n_samples",
        "alpha",
        "nominal_coverage",
        "empirical_coverage",
        "mean_interval_width",
        "median_interval_width",
    ]
    combined = pd.concat(
        [step5.loc[:, columns], repeated.loc[:, columns], real.loc[:, columns]],
        ignore_index=True,
    )
    combined = _coverage_annotations(combined)
    combined["pooling_status"] = "not_pooled_across_source_scopes"
    failures = combined[
        combined["observed_undercoverage_beyond_wilson_95"]
    ].reset_index(drop=True)
    return combined, failures


def _prediction_noise_tables(
    model_predictions: pd.DataFrame,
    best_model_name: str,
    repeated_intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = model_predictions[
        (model_predictions["model_name"] == best_model_name)
        & (model_predictions["split"] == "test")
    ].copy()
    model_rows: list[dict[str, Any]] = []
    for scenario, group in predictions.groupby("scenario", sort=True):
        actual = group["actual_travel_time_minutes"].to_numpy(dtype=float)
        predicted = group["predicted_travel_time_minutes"].to_numpy(dtype=float)
        residual = predicted - actual
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        r2 = math.nan if denominator == 0.0 else 1.0 - float(np.sum(residual**2)) / denominator
        model_rows.append(
            {
                "source_scope": "step4_best_model_held_out_test_rows",
                "model_name": best_model_name,
                "scenario": scenario,
                "is_prediction_noise_scenario": scenario == "prediction_noise",
                "n_samples": len(group),
                "mae_minutes": float(np.mean(np.abs(residual))),
                "rmse_minutes": float(np.sqrt(np.mean(residual**2))),
                "mean_signed_error_predicted_minus_actual_minutes": float(
                    np.mean(residual)
                ),
                "p95_absolute_error_minutes": float(np.quantile(np.abs(residual), 0.95)),
                "r2": r2,
            }
        )

    interval_rows: list[dict[str, Any]] = []
    for scenario, group in repeated_intervals.groupby("scenario", sort=True):
        actual = group["travel_time_minutes"].to_numpy(dtype=float)
        lower = group["lower_bound"].to_numpy(dtype=float)
        upper = group["upper_bound"].to_numpy(dtype=float)
        covered = (actual >= lower) & (actual <= upper)
        low, high = _wilson_interval(int(covered.sum()), len(group))
        interval_rows.append(
            {
                "source_scope": "step8_25_repeated_synthetic_held_out_test_rows",
                "method": "local_scaled",
                "alpha": 0.10,
                "nominal_coverage": 0.90,
                "scenario": scenario,
                "is_prediction_noise_scenario": scenario == "prediction_noise",
                "n_samples": len(group),
                "empirical_coverage": float(covered.mean()),
                "wilson_95_lower": low,
                "wilson_95_upper": high,
                "observed_undercoverage_beyond_wilson_95": high < 0.90,
                "mean_interval_width_minutes": float(np.mean(upper - lower)),
                "median_interval_width_minutes": float(np.median(upper - lower)),
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(interval_rows)


def _penalty_sweep_tables(
    risk_scores: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    penalty_config = dict(config["qubo_penalty"])
    selected = risk_scores[
        (risk_scores["variant"] == penalty_config["risk_variant"])
        & risk_scores["K"].isin(penalty_config["candidate_counts"])
    ].copy()
    tolerance = float(penalty_config["ground_tolerance"])
    rows: list[dict[str, Any]] = []

    for keys, group in selected.groupby(
        ["network_id", "topology", "K"], sort=True
    ):
        network_id, topology, k_value = keys
        ordered = _route_sort(group)
        risks = ordered["risk_score"].to_numpy(dtype=float)
        route_ids = ordered["route_id"].astype(str).tolist()
        minimum = float(risks.min())
        maximum = float(risks.max())
        settings = [
            (
                f"{float(multiplier):g}x_minimum_valid_cost",
                float(multiplier) * minimum,
                "multiplier_of_instance_minimum_valid_cost",
            )
            for multiplier in penalty_config["multipliers_of_minimum_valid_cost"]
        ]
        safe_penalty = maximum + float(
            penalty_config["safe_reference_margin_over_maximum_cost"]
        )
        settings.append(
            (
                "safe_maximum_cost_plus_margin",
                safe_penalty,
                "sufficient_safe_reference",
            )
        )

        assignments = list(product((0, 1), repeat=int(k_value)))
        for setting, penalty, definition in settings:
            costs = np.asarray(
                [
                    float(np.dot(risks, bits)) + penalty * (sum(bits) - 1) ** 2
                    for bits in assignments
                ],
                dtype=float,
            )
            minimum_cost = float(costs.min())
            ground_mask = np.isclose(costs, minimum_cost, atol=tolerance, rtol=0.0)
            ground_assignments = [
                bits for bits, is_ground in zip(assignments, ground_mask, strict=True) if is_ground
            ]
            validity = [sum(bits) == 1 for bits in ground_assignments]
            valid_ground_routes = {
                route_ids[bits.index(1)]
                for bits, valid in zip(ground_assignments, validity, strict=True)
                if valid
            }
            exact_route_id = route_ids[int(np.argmin(risks))]
            rows.append(
                {
                    "network_id": network_id,
                    "topology": topology,
                    "K": int(k_value),
                    "penalty_setting": setting,
                    "penalty_definition": definition,
                    "penalty": penalty,
                    "minimum_valid_route_cost": minimum,
                    "maximum_valid_route_cost": maximum,
                    "penalty_over_minimum_valid_cost": (
                        penalty / minimum if minimum > 0.0 else math.nan
                    ),
                    "violates_step7_sufficient_rule_penalty_gt_max": penalty <= maximum,
                    "ground_cost": minimum_cost,
                    "ground_state_count": len(ground_assignments),
                    "valid_ground_state_count": int(sum(validity)),
                    "invalid_ground_state_count": int(len(validity) - sum(validity)),
                    "ground_state_valid_fraction": float(np.mean(validity)),
                    "all_ground_states_valid": bool(all(validity)),
                    "any_valid_ground_state": bool(any(validity)),
                    "zero_state_is_ground": bool(
                        tuple(0 for _ in range(int(k_value))) in ground_assignments
                    ),
                    "exact_route_id": exact_route_id,
                    "exact_route_is_ground": exact_route_id in valid_ground_routes,
                    "actual_failure_region": not all(validity),
                }
            )

    detailed = pd.DataFrame(rows)
    aggregate = (
        detailed.groupby(["K", "penalty_setting", "penalty_definition"], sort=True)
        .agg(
            instance_count=("network_id", "size"),
            invalid_ground_state_instance_count=(
                "actual_failure_region",
                "sum",
            ),
            all_instances_all_ground_states_valid=("all_ground_states_valid", "all"),
            all_instances_have_exact_route_ground=("exact_route_is_ground", "all"),
            mean_ground_state_valid_fraction=("ground_state_valid_fraction", "mean"),
            minimum_ground_state_valid_fraction=("ground_state_valid_fraction", "min"),
            zero_state_ground_instance_count=("zero_state_is_ground", "sum"),
        )
        .reset_index()
    )
    failures = aggregate[
        aggregate["invalid_ground_state_instance_count"] > 0
    ].reset_index(drop=True)
    safe = detailed[detailed["penalty_setting"] == "safe_maximum_cost_plus_margin"]
    checks = {
        "qubo_penalty_instance_count": int(
            detailed[["network_id", "topology", "K"]].drop_duplicates().shape[0]
        ),
        "unsafe_low_penalty_failures_observed": bool(detailed["actual_failure_region"].any()),
        "safe_reference_all_ground_states_valid": bool(safe["all_ground_states_valid"].all()),
        "safe_reference_all_exact_routes_ground": bool(safe["exact_route_is_ground"].all()),
    }
    return detailed, aggregate, failures, checks


def _targeted_qaoa_runs(
    risk_scores: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qaoa_config = dict(config["qaoa"])
    source = risk_scores[risk_scores["variant"] == qaoa_config["risk_variant"]]
    metric_rows: list[dict[str, Any]] = []
    history_tables: list[pd.DataFrame] = []
    configurations = list(
        product(
            qaoa_config["target_candidate_counts"],
            qaoa_config["target_depths"],
            qaoa_config["target_seeds"],
            qaoa_config["methods"],
        )
    )
    for run_index, (k_value, depth, seed, method) in enumerate(configurations, start=1):
        group = source[
            (source["network_id"] == qaoa_config["target_network_id"])
            & (source["K"] == int(k_value))
        ]
        ordered = _route_sort(group)
        if len(ordered) != int(k_value):
            raise ValueError(f"target QAOA expected K={k_value}, found {len(ordered)} rows")
        risks = ordered["risk_score"].to_numpy(dtype=float)
        qubo = build_risk_qubo(risks)
        hamiltonian, offset, _, _ = risk_qubo_to_ising(qubo)
        expectation = create_expectation_qnode(
            hamiltonian,
            number_of_qubits=int(k_value),
            depth=int(depth),
            method=str(method),
        )
        probability = create_probability_qnode(
            hamiltonian,
            number_of_qubits=int(k_value),
            depth=int(depth),
            method=str(method),
        )
        optimized = optimize_qaoa_angles(
            expectation,
            depth=int(depth),
            seed=int(seed),
            iterations=int(qaoa_config["iterations"]),
            step_size=float(qaoa_config["step_size"]),
        )
        probabilities = np.asarray(
            probability(optimized.gammas, optimized.betas), dtype=float
        )
        optimum = int(np.argmin(risks))
        summary = summarize_probabilities(
            probabilities,
            qubo,
            optimal_route_index=optimum,
        )
        resources = circuit_resource_record(
            probability, optimized.gammas, optimized.betas
        )
        run_id = f"step9_4_qaoa_target_{run_index:03d}"
        metric_rows.append(
            {
                "record_origin": "targeted_step9_4_missing_depth_cell",
                "scope_id": "single_grid_network_candidate_scaling",
                "run_id": run_id,
                "network_id": qaoa_config["target_network_id"],
                "topology": str(ordered.iloc[0]["topology"]),
                "K": int(k_value),
                "depth": int(depth),
                "seed": int(seed),
                "method": method,
                "iterations": int(qaoa_config["iterations"]),
                "step_size": float(qaoa_config["step_size"]),
                "penalty": qubo.penalty,
                "exact_optimal_route_index": optimum,
                "exact_optimal_route_id": str(ordered.iloc[optimum]["route_id"]),
                "analytic_valid_probability": summary["valid_probability"],
                "analytic_optimal_probability": summary["optimal_probability"],
                "most_probable_valid_route_index": summary[
                    "most_probable_valid_route_index"
                ],
                "most_probable_valid_is_optimal": (
                    int(summary["most_probable_valid_route_index"]) == optimum
                ),
                "final_expected_qubo_energy": summary["expected_qubo_energy"],
                "warm_start_used": False,
                **resources,
            }
        )
        history = optimized.history.copy()
        history.insert(0, "run_id", run_id)
        history.insert(1, "record_origin", "targeted_step9_4_missing_depth_cell")
        history.insert(2, "network_id", qaoa_config["target_network_id"])
        history.insert(3, "K", int(k_value))
        history.insert(4, "depth", int(depth))
        history.insert(5, "seed", int(seed))
        history.insert(6, "method", method)
        history["expected_qubo_energy"] = history["hamiltonian_expectation"] + offset
        history_tables.append(history)
        print(
            f"[target QAOA {run_index}/{len(configurations)}] K={k_value} p={depth} "
            f"{method} seed={seed}: valid={summary['valid_probability']:.4f}, "
            f"optimal={summary['optimal_probability']:.4f}",
            flush=True,
        )
    return pd.DataFrame(metric_rows), pd.concat(history_tables, ignore_index=True)


def _combined_qaoa_tables(
    inherited: pd.DataFrame,
    targeted: pd.DataFrame,
    risk_scores: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    qaoa_config = dict(config["qaoa"])
    risk_source = risk_scores[risk_scores["variant"] == qaoa_config["risk_variant"]]
    penalty_lookup: dict[tuple[str, int], float] = {}
    for (network_id, k_value), group in risk_source.groupby(["network_id", "K"]):
        penalty_lookup[(str(network_id), int(k_value))] = float(group["risk_score"].max() + 1.0)

    inherited_rows: list[dict[str, Any]] = []
    for row in inherited.itertuples(index=False):
        scope_id = (
            "five_synthetic_topologies_K4"
            if row.study == "topology_primary"
            else "single_grid_network_candidate_scaling"
        )
        inherited_rows.append(
            {
                "record_origin": "inherited_step7_5_raw_run",
                "scope_id": scope_id,
                "run_id": row.run_id,
                "network_id": row.network_id,
                "topology": row.topology,
                "K": int(row.K),
                "depth": int(row.depth),
                "seed": int(row.seed),
                "method": row.method,
                "iterations": int(row.iterations),
                "step_size": float(row.step_size),
                "penalty": penalty_lookup[(str(row.network_id), int(row.K))],
                "exact_optimal_route_index": int(row.exact_optimal_route_index),
                "exact_optimal_route_id": row.exact_optimal_route_id,
                "analytic_valid_probability": float(row.analytic_valid_probability),
                "analytic_optimal_probability": float(row.analytic_optimal_probability),
                "most_probable_valid_route_index": int(
                    row.most_probable_valid_route_index
                ),
                "most_probable_valid_is_optimal": int(
                    row.most_probable_valid_route_index
                )
                == int(row.exact_optimal_route_index),
                "final_expected_qubo_energy": float(row.final_expected_qubo_energy),
                "warm_start_used": False,
                "circuit_depth": int(row.circuit_depth),
                "gate_count": int(row.gate_count),
                "gate_types_json": row.gate_types_json,
                "gate_sizes_json": row.gate_sizes_json,
            }
        )
    combined = pd.concat(
        [pd.DataFrame(inherited_rows), targeted], ignore_index=True, sort=False
    ).sort_values(
        ["scope_id", "K", "depth", "method", "network_id", "seed"],
        kind="stable",
    ).reset_index(drop=True)

    aggregate = (
        combined.groupby(["scope_id", "K", "depth", "method"], sort=True)
        .agg(
            run_count=("run_id", "size"),
            network_count=("network_id", "nunique"),
            seed_count=("seed", "nunique"),
            mean_valid_probability=("analytic_valid_probability", "mean"),
            minimum_valid_probability=("analytic_valid_probability", "min"),
            maximum_valid_probability=("analytic_valid_probability", "max"),
            mean_optimal_probability=("analytic_optimal_probability", "mean"),
            minimum_optimal_probability=("analytic_optimal_probability", "min"),
            maximum_optimal_probability=("analytic_optimal_probability", "max"),
            most_probable_valid_optimum_rate=("most_probable_valid_is_optimal", "mean"),
        )
        .reset_index()
    )
    aggregate["feasibility_leakage_observed"] = (
        aggregate["minimum_valid_probability"] < 1.0 - 1e-8
    )
    aggregate["most_probable_selection_failure_observed"] = (
        aggregate["most_probable_valid_optimum_rate"] < 1.0
    )
    failures = aggregate[
        aggregate["feasibility_leakage_observed"]
        | aggregate["most_probable_selection_failure_observed"]
    ].reset_index(drop=True)

    expected_cells = {
        (int(k), int(depth), str(method), int(seed))
        for k in qaoa_config["target_candidate_counts"]
        for depth in (1, 2, 3)
        for method in qaoa_config["methods"]
        for seed in qaoa_config["target_seeds"]
    }
    scaling = combined[
        combined["scope_id"] == "single_grid_network_candidate_scaling"
    ]
    observed_cells = {
        (int(row.K), int(row.depth), str(row.method), int(row.seed))
        for row in scaling.itertuples()
    }
    feasible = combined[combined["method"] == "feasible_xy"]
    checks = {
        "qaoa_inherited_run_count": int(len(inherited)),
        "qaoa_targeted_run_count": int(len(targeted)),
        "qaoa_combined_run_count": int(len(combined)),
        "qaoa_scaling_K6_K8_p1_p2_p3_cells_complete": expected_cells.issubset(
            observed_cells
        ),
        "qaoa_feasible_xy_validity_preserved": bool(
            np.allclose(feasible["analytic_valid_probability"], 1.0, atol=1e-8)
        ),
        "qaoa_warm_start_used_anywhere": bool(combined["warm_start_used"].any()),
    }
    return combined, aggregate, failures, checks


def _event_catalog(graph: nx.DiGraph) -> list[DynamicEvent]:
    """Return the same topology-agnostic controlled events used in Step 8."""

    base_path = nx.shortest_path(
        graph,
        graph.graph["origin"],
        graph.graph["destination"],
        weight="base_travel_time_min",
    )
    path_edges = list(zip(base_path[:-1], base_path[1:]))
    first_edge = path_edges[0]
    middle_edge = path_edges[len(path_edges) // 2]
    return [
        DynamicEvent("normal", global_congestion=0.20, actual_noise_fraction=0.03),
        DynamicEvent("rush_hour", global_congestion=0.78, actual_noise_fraction=0.05),
        DynamicEvent(
            "rain",
            global_congestion=0.45,
            rain_indicator=1,
            actual_noise_fraction=0.06,
        ),
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


def _targeted_dynamic_tables(
    config: dict[str, Any],
    network_config: dict[str, Any],
    interval_model: object,
    point_only_model: PointOnlyIntervalModel,
    distance_normalization_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dynamic_config = dict(config["dynamic_targeted"])
    baseline_weights = dict(config["static_risk"]["baseline_weights"])
    run_rows: list[dict[str, Any]] = []
    decision_tables: list[pd.DataFrame] = []
    total = (
        len(network_config["networks"])
        * 7
        * len(dynamic_config["conditions"])
    )
    run_index = 0

    for network_index, entry in enumerate(network_config["networks"]):
        graph = generate_synthetic_network(
            SyntheticNetworkSpec(
                str(entry["topology"]), int(entry["seed"]), dict(entry["parameters"])
            )
        )
        base_shortest_time = float(
            nx.shortest_path_length(
                graph,
                graph.graph["origin"],
                graph.graph["destination"],
                weight="base_travel_time_min",
            )
        )
        deadline = base_shortest_time * float(
            dynamic_config["deadline_multiplier_over_base_shortest_time"]
        )
        catalog = _event_catalog(graph)
        for rotation in range(len(catalog)):
            events = catalog[rotation:] + catalog[:rotation]
            scenario = f"starts_{events[0].name}"
            paired_seed = int(dynamic_config["simulation_seed"]) + network_index * 10 + rotation
            for condition in dynamic_config["conditions"]:
                weights, point_only, zero_reroute = _condition_parameters(
                    str(condition), baseline_weights
                )
                result = simulate_dynamic_route(
                    graph,
                    graph.graph["origin"],
                    graph.graph["destination"],
                    events,
                    policy="uncertainty_aware_classical",
                    k=int(dynamic_config["candidate_count"]),
                    deadline_minutes=deadline,
                    point_interval_model=(point_only_model if point_only else interval_model),
                    distance_normalization_km=distance_normalization_km,
                    risk_weights=weights,
                    reroute_penalty_minutes=(
                        0.0
                        if zero_reroute
                        else float(dynamic_config["reroute_penalty_minutes"])
                    ),
                    random_seed=paired_seed,
                )
                run_index += 1
                run_id = f"step9_4_dynamic_{run_index:04d}"
                decisions = result.decisions.copy()
                initial_route = (
                    decisions.iloc[0].get("selected_route_id")
                    if not decisions.empty
                    else None
                )
                run_rows.append(
                    {
                        "run_id": run_id,
                        "condition": condition,
                        "evidence_scope": "targeted_controlled_dynamic_simulation",
                        "network_id": entry["network_id"],
                        "topology": entry["topology"],
                        "event_rotation": rotation,
                        "scenario": scenario,
                        "first_event": events[0].name,
                        "paired_simulation_seed": paired_seed,
                        "deadline_minutes": deadline,
                        "initial_selected_route_id": initial_route,
                        "conformal_calibration_used": not point_only,
                        "reroute_penalty_minutes": (
                            0.0
                            if zero_reroute
                            else float(dynamic_config["reroute_penalty_minutes"])
                        ),
                        **result.summary(),
                    }
                )
                decisions.insert(0, "run_id", run_id)
                decisions.insert(1, "condition", condition)
                decisions.insert(2, "network_id", entry["network_id"])
                decisions.insert(3, "topology", entry["topology"])
                decisions.insert(4, "event_rotation", rotation)
                decisions.insert(5, "scenario", scenario)
                decision_tables.append(decisions)
                print(
                    f"[target dynamic {run_index}/{total}] {entry['network_id']} "
                    f"{scenario} {condition}: late={result.deadline_violation}, "
                    f"reroutes={result.reroute_count}",
                    flush=True,
                )

    runs = pd.DataFrame(run_rows)
    decisions = pd.concat(decision_tables, ignore_index=True, sort=False)
    case_keys = ["network_id", "topology", "event_rotation"]
    full = runs[runs["condition"] == "full"].set_index(case_keys)
    runs["full_initial_selected_route_id"] = [
        full.loc[(row.network_id, row.topology, row.event_rotation), "initial_selected_route_id"]
        for row in runs.itertuples()
    ]
    runs["changed_initial_selection_from_full"] = (
        runs["initial_selected_route_id"] != runs["full_initial_selected_route_id"]
    )

    summary_rows: list[dict[str, Any]] = []
    for condition, group in runs.groupby("condition", sort=True):
        populations = [("overall", group)] + [
            (f"first_event={event}", group[group["first_event"] == event])
            for event in sorted(group["first_event"].unique())
        ]
        for scope, subset in populations:
            summary_rows.append(
                {
                    "condition": condition,
                    "aggregation_scope": scope,
                    "run_count": len(subset),
                    "arrival_rate": float(subset["arrived"].mean()),
                    "realized_deadline_violation_rate": float(
                        subset["deadline_violation"].mean()
                    ),
                    "mean_arrival_time_minutes_among_arrivals": float(
                        subset["arrival_time_minutes"].mean()
                    ),
                    "mean_realized_deadline_margin_minutes_among_arrivals": float(
                        subset["deadline_margin_minutes"].mean()
                    ),
                    "mean_reroutes": float(subset["reroute_count"].mean()),
                    "mean_unnecessary_reroutes": float(
                        subset["unnecessary_reroutes"].mean()
                    ),
                    "initial_selection_change_rate_from_full": float(
                        subset["changed_initial_selection_from_full"].mean()
                    ),
                    "outcome_type": "realized_controlled_simulation_not_real_traffic",
                }
            )
    summary = pd.DataFrame(summary_rows)

    paired_rows: list[dict[str, Any]] = []
    full_columns = case_keys + [
        "arrived",
        "arrival_time_minutes",
        "deadline_violation",
        "reroute_count",
        "initial_selected_route_id",
    ]
    full_cases = runs[runs["condition"] == "full"].loc[:, full_columns]
    for condition, group in runs[runs["condition"] != "full"].groupby(
        "condition", sort=True
    ):
        paired = group.merge(
            full_cases,
            on=case_keys,
            suffixes=("_condition", "_full"),
            validate="one_to_one",
        )
        both_arrived = paired["arrived_condition"] & paired["arrived_full"]
        paired_rows.append(
            {
                "condition": condition,
                "reference_condition": "full",
                "paired_case_count": len(paired),
                "initial_selection_change_count": int(
                    (
                        paired["initial_selected_route_id_condition"]
                        != paired["initial_selected_route_id_full"]
                    ).sum()
                ),
                "realized_deadline_outcome_change_count": int(
                    (
                        paired["deadline_violation_condition"]
                        != paired["deadline_violation_full"]
                    ).sum()
                ),
                "realized_deadline_improvement_count": int(
                    (
                        ~paired["deadline_violation_condition"]
                        & paired["deadline_violation_full"]
                    ).sum()
                ),
                "realized_deadline_worsening_count": int(
                    (
                        paired["deadline_violation_condition"]
                        & ~paired["deadline_violation_full"]
                    ).sum()
                ),
                "mean_arrival_time_delta_minutes_condition_minus_full_both_arrived": float(
                    (
                        paired.loc[both_arrived, "arrival_time_minutes_condition"]
                        - paired.loc[both_arrived, "arrival_time_minutes_full"]
                    ).mean()
                ),
                "mean_reroute_delta_condition_minus_full": float(
                    (
                        paired["reroute_count_condition"]
                        - paired["reroute_count_full"]
                    ).mean()
                ),
                "outcome_type": "paired_realized_controlled_simulation",
            }
        )
    paired_summary = pd.DataFrame(paired_rows)
    expected_runs = total
    checks = {
        "dynamic_targeted_expected_runs": expected_runs,
        "dynamic_targeted_actual_runs": int(len(runs)),
        "dynamic_targeted_complete": len(runs) == expected_runs,
        "dynamic_pairing_unique": not runs.duplicated(
            ["condition", *case_keys]
        ).any(),
    }
    return runs, decisions, summary, paired_summary, checks


def _inherited_dynamic_context(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, group in source.groupby("policy", sort=True):
        populations = [("overall", group)] + [
            (f"first_event={event}", group[group["first_event"] == event])
            for event in sorted(group["first_event"].unique())
        ]
        for scope, subset in populations:
            rows.append(
                {
                    "source_scope": "inherited_step8_controlled_dynamic_simulation",
                    "policy": policy,
                    "aggregation_scope": scope,
                    "run_count": len(subset),
                    "arrival_rate": float(subset["arrived"].mean()),
                    "realized_deadline_violation_rate": float(
                        subset["deadline_violation"].mean()
                    ),
                    "mean_arrival_time_minutes_among_arrivals": float(
                        subset["arrival_time_minutes"].mean()
                    ),
                    "mean_realized_deadline_margin_minutes_among_arrivals": float(
                        subset["deadline_margin_minutes"].mean()
                    ),
                    "mean_reroutes": float(subset["reroute_count"].mean()),
                    "mean_unnecessary_reroutes": float(
                        subset["unnecessary_reroutes"].mean()
                    ),
                    "causal_ablation_status": (
                        "context_only; policy differences do not isolate one risk term"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _classical_k_context(source: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "method",
        "K",
        "decision_count",
        "comparable_decision_count",
        "exact_risk_optimal_count",
        "exact_risk_optimal_rate",
        "mean_risk_regret",
        "median_risk_regret",
        "maximum_risk_regret",
        "deadline_proxy_violation_count",
        "deadline_proxy_violation_rate",
        "mean_deadline_margin_min",
        "tied_decision_count",
        "stable_tie_check_count",
    ]
    result = source.loc[:, keep].copy()
    result.insert(0, "source_scope", "inherited_step9_3_static_candidate_baselines")
    result["runtime_metrics_excluded"] = True
    result["selection_outcome_type"] = "static_candidate_or_graph_baseline_proxy"
    return result


def _warm_start_status(config: dict[str, Any]) -> pd.DataFrame:
    implemented = bool(config["qaoa"]["warm_start_implemented"])
    if implemented:
        raise ValueError("this runner has no validated warm-start implementation")
    return pd.DataFrame(
        [
            {
                "ablation": "warm_start_qaoa",
                "implemented": False,
                "executed": False,
                "comparison_valid": False,
                "status": "not_implemented_not_run",
                "reason": (
                    "A warm start requires a distinct, specified initializer and paired "
                    "optimization protocol. Re-labeling cold-start angles would be invalid."
                ),
                "invalid_action_avoided": "no fabricated warm-start baseline",
            }
        ]
    )


def _markdown_table(data: pd.DataFrame, columns: list[str], maximum_rows: int = 20) -> str:
    subset = data.loc[:, columns].head(maximum_rows)

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(format_value(row[column]) for column in columns) + " |"
        for _, row in subset.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _validation_report(
    tables: dict[str, pd.DataFrame],
    checks: dict[str, Any],
) -> str:
    static_overall = tables["static_risk_ablation_summary.csv"].query(
        "aggregation_scope == 'overall'"
    )
    alpha_overall = tables["alpha_sweep_summary.csv"].query(
        "aggregation_scope == 'overall'"
    )
    dynamic_overall = tables["dynamic_ablation_summary.csv"].query(
        "aggregation_scope == 'overall'"
    )
    prediction_model = tables["prediction_noise_model_metrics.csv"]
    prediction_intervals = tables["prediction_noise_interval_metrics.csv"]
    qaoa = tables["qaoa_aggregate.csv"]
    qaoa_scaling = qaoa[qaoa["scope_id"] == "single_grid_network_candidate_scaling"]
    penalty_failures = tables["qubo_penalty_failure_regions.csv"]
    coverage_failures = tables["conformal_failure_regions.csv"]
    risk_failures = tables["risk_weight_sweep_summary.csv"].query(
        "aggregation_scope == 'overall' and "
        "(selection_change_count_from_baseline_weights > 0 or "
        "deadline_proxy_violation_rate > 0)"
    )
    dynamic_pairs = tables["dynamic_ablation_paired_deltas.csv"]

    prediction_noise_model = prediction_model[
        prediction_model["is_prediction_noise_scenario"]
    ].iloc[0]
    prediction_noise_interval = prediction_intervals[
        prediction_intervals["is_prediction_noise_scenario"]
    ].iloc[0]

    return f"""# Step 9.4 Robustness and Ablation Validation Report

## Outcome

The package reconciles inherited raw artifacts with targeted missing cells. Static
deadline results are explicitly proxies; dynamic results are realized outcomes in a
controlled simulator, not real traffic. Ideal statevector QAOA probabilities are not
hardware results. No runtime or speed comparison is made.

All validation checks passed: **{all(bool(value) for key, value in checks.items() if key.endswith(('_complete', '_matches_source_E', '_preserved', '_valid', '_ground', '_unique')))}**.

## Isolated static risk removals

Deadlines are held fixed within each network/K instance. `without_conformal_calibration`
uses degenerate point intervals and therefore removes both calibrated width and the
conservative conformal upper-bound proxy; it is not conformal inference.

{_markdown_table(static_overall, ['condition', 'decision_count', 'selection_change_count_from_full', 'deadline_proxy_violation_rate', 'mean_selected_deadline_margin_minutes'])}

The inherited A-E table is retained separately because several legacy variants remove
multiple terms at once and are not a one-term causal isolation.

## Alpha and coverage robustness

Alpha selection effects use the same candidate states and keep the original alpha=0.10
deadline fixed so interval width is the changed factor.

{_markdown_table(alpha_overall, ['alpha', 'decision_count', 'selection_change_count_from_alpha_0_10', 'deadline_proxy_violation_rate', 'mean_selected_interval_half_width'])}

Observed coverage failure regions use the strict descriptive rule that the nominal
coverage lies above the empirical Wilson 95% upper bound. Source populations are never
pooled.

{_markdown_table(coverage_failures, ['source_scope', 'evaluation_set', 'method', 'alpha', 'nominal_coverage', 'empirical_coverage', 'wilson_95_upper'])}

## Prediction noise

For the frozen Step 4 best model, the held-out `prediction_noise` scenario has MAE
**{prediction_noise_model.mae_minutes:.4f} min** and RMSE
**{prediction_noise_model.rmse_minutes:.4f} min**. In the separate repeated-topology
scope, alpha=0.10 local intervals cover **{prediction_noise_interval.empirical_coverage:.4f}**
of prediction-noise rows (n={int(prediction_noise_interval.n_samples)}). These scopes are
reported separately and are not pooled.

## Risk-weight failure regions

The sweep is one-at-a-time around the baseline (1, 2, 1); it is not a full factorial
interaction study. A row is retained here when selections changed or at least one
selected route violated the static deadline proxy.

{_markdown_table(risk_failures, ['sweep_parameter', 'sweep_value', 'selection_change_count_from_baseline_weights', 'deadline_proxy_violation_rate', 'mean_selected_deadline_margin_minutes'])}

## QUBO penalty strength

Unsafe penalties are evaluated directly rather than passed through the safe Step 7
constructor. Ground-state ties count as a failure of guaranteed validity whenever any
invalid assignment shares the minimum.

{_markdown_table(penalty_failures, ['K', 'penalty_setting', 'instance_count', 'invalid_ground_state_instance_count', 'mean_ground_state_valid_fraction', 'zero_state_ground_instance_count'])}

The `safe_maximum_cost_plus_margin` reference preserves valid exact ground states for
all {checks['qubo_penalty_instance_count']} network/K instances. The sweep also shows
where the sufficient Step 7 rule is conservative versus where low penalties actually
fail.

## QAOA method, K, and p

Inherited Step 7.5 contains 106 runs. Eight targeted runs add the missing p=3 cells for
K=6/8 on the same single grid scaling instance and seeds. K=4 remains a separate
five-topology scope. Valid and optimal probabilities are analytic ideal-simulator
probabilities.

{_markdown_table(qaoa_scaling, ['K', 'depth', 'method', 'run_count', 'mean_valid_probability', 'mean_optimal_probability', 'most_probable_valid_optimum_rate'])}

Feasible-XY preserved one-hot validity across all inherited and targeted runs:
**{checks['qaoa_feasible_xy_validity_preserved']}**. Standard-X feasibility leakage and
most-probable selection misses are listed in `qaoa_failure_regions.csv`.

Warm start is explicitly **not implemented and not run**. A fabricated comparison from
cold-start angles would be invalid.

## Realized controlled dynamic outcomes

The targeted matrix contains {checks['dynamic_targeted_actual_runs']} paired classical
simulations: five ablations by five synthetic networks by seven event rotations. Each
condition uses the same per-case random seed; paths may diverge after a selection change.

{_markdown_table(dynamic_overall, ['condition', 'run_count', 'arrival_rate', 'realized_deadline_violation_rate', 'mean_reroutes', 'initial_selection_change_rate_from_full'])}

{_markdown_table(dynamic_pairs, ['condition', 'paired_case_count', 'initial_selection_change_count', 'realized_deadline_improvement_count', 'realized_deadline_worsening_count', 'mean_reroute_delta_condition_minus_full'])}

Inherited Step 8 policy outcomes are retained only as context; they do not isolate a
single risk term. Static deadline-proxy rates must not be compared numerically with
realized dynamic violation rates as if they shared a population.

## K=4/6/8 and classical context

Static risk, penalty-ground-state, and inherited classical-baseline tables cover
K=4/6/8. Classical baseline runtime columns were intentionally excluded, and this
report makes no speed or quantum-advantage claim.

## Reproducibility and claim boundaries

- `ablation_index.csv` maps every result/report to raw sources or targeted runs and SHA-256 hashes.
- `manifest.json` records configuration, environment, source hashes, and output hashes.
- `validation_checks.json` records machine-readable completeness and reconciliation checks.
- Source scopes remain distinct in every aggregate; there is no pooled ID/OOD, synthetic/real-topology, static/dynamic, or K4-topology/K6-8-scaling metric.
- No warm start, hardware execution, real-traffic inference, runtime comparison, speed claim, or quantum-advantage claim is present.
"""


def run(*, force: bool = False) -> None:
    config = _read_json(CONFIG_PATH)
    output_directory = REPOSITORY_ROOT / str(config["output_directory"])
    if output_directory.exists() and any(output_directory.iterdir()) and not force:
        raise FileExistsError(
            f"{output_directory} is nonempty; pass --force to replace named Step 9.4 files"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    source_directories = {
        name: REPOSITORY_ROOT / str(path)
        for name, path in dict(config["source_artifacts"]).items()
    }
    source_files = {
        "traffic_predictions": source_directories["traffic_model"] / "predictions.csv",
        "traffic_manifest": source_directories["traffic_model"] / "manifest.json",
        "traffic_best_model": source_directories["traffic_model"] / "best_model.joblib",
        "traffic_calibration": source_directories["traffic_model"] / "calibration.csv",
        "conformal_quantiles": source_directories["conformal"] / "calibration_quantiles.csv",
        "conformal_coverage": source_directories["conformal"] / "coverage_overall.csv",
        "conformal_scale_model": source_directories["conformal"] / "residual_scale_model.joblib",
        "risk_candidates": source_directories["routing_risk"] / "candidate_route_predictions.csv",
        "risk_scores": source_directories["routing_risk"] / "route_risk_scores.csv",
        "risk_decisions": source_directories["routing_risk"] / "selected_routes_by_ablation.csv",
        "risk_ablation_summary": source_directories["routing_risk"] / "ablation_summary.csv",
        "qaoa_runs": source_directories["qaoa"] / "qaoa_runs.csv",
        "qaoa_manifest": source_directories["qaoa"] / "manifest.json",
        "qaoa_warm_status": source_directories["qaoa"] / "WARM_START_STATUS.md",
        "dynamic_runs": source_directories["dynamic"] / "dynamic_runs.csv",
        "repeated_intervals": source_directories["repeated_topology"] / "conformal_test_intervals.csv",
        "repeated_coverage": source_directories["repeated_topology"] / "conformal_coverage_overall.csv",
        "real_coverage": source_directories["real_topology"] / "conformal_coverage_overall.csv",
        "classical_by_k": source_directories["classical_baselines"] / "baseline_by_k.csv",
        "network_config": REPOSITORY_ROOT / str(config["network_config"]),
    }
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required source artifacts are missing: {missing}")

    candidates = pd.read_csv(source_files["risk_candidates"])
    risk_scores = pd.read_csv(source_files["risk_scores"])
    source_risk_decisions = pd.read_csv(source_files["risk_decisions"])
    inherited_ablation_summary = pd.read_csv(source_files["risk_ablation_summary"])
    calibration_quantiles = pd.read_csv(source_files["conformal_quantiles"])

    static_scores, static_decisions, static_summary, static_checks = (
        _risk_ablation_tables(
            candidates,
            source_risk_decisions,
            config,
        )
    )
    inherited_variants = _inherited_risk_variants(inherited_ablation_summary)
    alpha_decisions, alpha_summary, alpha_checks = _alpha_sweep_tables(
        candidates,
        calibration_quantiles,
        config,
    )
    weight_decisions, weight_summary = _weight_sweep_tables(candidates, config)

    step5_coverage = pd.read_csv(source_files["conformal_coverage"])
    repeated_coverage = pd.read_csv(source_files["repeated_coverage"])
    real_coverage = pd.read_csv(source_files["real_coverage"])
    coverage_scopes, coverage_failures = _conformal_scope_coverage(
        step5_coverage,
        repeated_coverage,
        real_coverage,
    )

    traffic_manifest = _read_json(source_files["traffic_manifest"])
    best_model_name = str(traffic_manifest["best_validation_model"])
    prediction_model, prediction_intervals = _prediction_noise_tables(
        pd.read_csv(source_files["traffic_predictions"]),
        best_model_name,
        pd.read_csv(source_files["repeated_intervals"]),
    )

    penalty_details, penalty_aggregate, penalty_failures, penalty_checks = (
        _penalty_sweep_tables(risk_scores, config)
    )

    targeted_qaoa, targeted_qaoa_history = _targeted_qaoa_runs(risk_scores, config)
    qaoa_runs, qaoa_aggregate, qaoa_failures, qaoa_checks = _combined_qaoa_tables(
        pd.read_csv(source_files["qaoa_runs"]),
        targeted_qaoa,
        risk_scores,
        config,
    )

    base_model = joblib.load(source_files["traffic_best_model"])
    scale_model = joblib.load(source_files["conformal_scale_model"])
    calibration = pd.read_csv(source_files["traffic_calibration"])
    interval_model = calibrate_locally_scaled_split_conformal(
        base_model,
        scale_model,
        calibration,
        alpha=float(config["static_risk"]["baseline_alpha"]),
    )
    normalization_values = calibration["distance_normalization_km"].unique()
    if len(normalization_values) != 1:
        raise ValueError("Step 4 calibration must have one distance normalization")
    distance_normalization = float(normalization_values[0])
    network_config = _read_json(source_files["network_config"])
    dynamic_runs, dynamic_decisions, dynamic_summary, dynamic_pairs, dynamic_checks = (
        _targeted_dynamic_tables(
            config,
            network_config,
            interval_model,
            PointOnlyIntervalModel(base_model),
            distance_normalization,
        )
    )
    inherited_dynamic = _inherited_dynamic_context(
        pd.read_csv(source_files["dynamic_runs"])
    )
    classical_k = _classical_k_context(pd.read_csv(source_files["classical_by_k"]))
    warm_start = _warm_start_status(config)

    tables: dict[str, pd.DataFrame] = {
        "static_risk_ablation_scores.csv": static_scores,
        "static_risk_ablation_decisions.csv": static_decisions,
        "static_risk_ablation_summary.csv": static_summary,
        "inherited_factorial_risk_summary.csv": inherited_variants,
        "alpha_sweep_decisions.csv": alpha_decisions,
        "alpha_sweep_summary.csv": alpha_summary,
        "risk_weight_sweep_decisions.csv": weight_decisions,
        "risk_weight_sweep_summary.csv": weight_summary,
        "conformal_scope_coverage.csv": coverage_scopes,
        "conformal_failure_regions.csv": coverage_failures,
        "prediction_noise_model_metrics.csv": prediction_model,
        "prediction_noise_interval_metrics.csv": prediction_intervals,
        "qubo_penalty_sweep.csv": penalty_details,
        "qubo_penalty_aggregate.csv": penalty_aggregate,
        "qubo_penalty_failure_regions.csv": penalty_failures,
        "qaoa_runs.csv": qaoa_runs,
        "qaoa_aggregate.csv": qaoa_aggregate,
        "qaoa_failure_regions.csv": qaoa_failures,
        "qaoa_targeted_training_history.csv": targeted_qaoa_history,
        "dynamic_ablation_runs.csv": dynamic_runs,
        "dynamic_ablation_decisions.csv": dynamic_decisions,
        "dynamic_ablation_summary.csv": dynamic_summary,
        "dynamic_ablation_paired_deltas.csv": dynamic_pairs,
        "inherited_dynamic_policy_context.csv": inherited_dynamic,
        "classical_K_context.csv": classical_k,
        "warm_start_status.csv": warm_start,
    }

    checks: dict[str, Any] = {
        **static_checks,
        **alpha_checks,
        **penalty_checks,
        **qaoa_checks,
        **dynamic_checks,
        "risk_weight_candidate_counts_complete": sorted(
            int(value) for value in weight_decisions["K"].unique()
        )
        == [4, 6, 8],
        "prediction_noise_model_row_present": bool(
            prediction_model["is_prediction_noise_scenario"].any()
        ),
        "prediction_noise_interval_row_present": bool(
            prediction_intervals["is_prediction_noise_scenario"].any()
        ),
        "conformal_scope_metrics_not_pooled": bool(
            (coverage_scopes["pooling_status"] == "not_pooled_across_source_scopes").all()
        ),
        "warm_start_explicitly_not_implemented": bool(
            (~warm_start["implemented"] & ~warm_start["executed"]).all()
        ),
        "hardware_runs_performed": False,
        "runtime_comparisons_reported": False,
    }
    required_boolean_checks = [
        "static_full_matches_source_E",
        "unsafe_low_penalty_failures_observed",
        "safe_reference_all_ground_states_valid",
        "safe_reference_all_exact_routes_ground",
        "qaoa_scaling_K6_K8_p1_p2_p3_cells_complete",
        "qaoa_feasible_xy_validity_preserved",
        "dynamic_targeted_complete",
        "dynamic_pairing_unique",
        "risk_weight_candidate_counts_complete",
        "prediction_noise_model_row_present",
        "prediction_noise_interval_row_present",
        "conformal_scope_metrics_not_pooled",
        "warm_start_explicitly_not_implemented",
    ]
    failed_checks = [name for name in required_boolean_checks if not bool(checks[name])]
    checks["required_checks_passed"] = not failed_checks
    checks["failed_required_checks"] = failed_checks
    if failed_checks:
        raise RuntimeError(f"Step 9.4 validation checks failed: {failed_checks}")

    table_sources = {
        "static": (
            source_files["risk_candidates"],
            source_files["risk_decisions"],
        ),
        "inherited_risk": (source_files["risk_ablation_summary"],),
        "alpha": (source_files["risk_candidates"], source_files["conformal_quantiles"]),
        "weights": (source_files["risk_candidates"],),
        "coverage": (
            source_files["conformal_coverage"],
            source_files["repeated_coverage"],
            source_files["real_coverage"],
        ),
        "prediction_model": (
            source_files["traffic_predictions"],
            source_files["traffic_manifest"],
        ),
        "prediction_interval": (source_files["repeated_intervals"],),
        "penalty": (source_files["risk_scores"],),
        "qaoa": (source_files["qaoa_runs"], source_files["risk_scores"]),
        "dynamic_targeted": (
            source_files["traffic_best_model"],
            source_files["conformal_scale_model"],
            source_files["traffic_calibration"],
            source_files["network_config"],
        ),
        "dynamic_inherited": (source_files["dynamic_runs"],),
        "classical": (source_files["classical_by_k"],),
        "warm": (source_files["qaoa_manifest"], source_files["qaoa_warm_status"]),
    }
    provenance: dict[str, TableProvenance] = {}
    for name in (
        "static_risk_ablation_scores.csv",
        "static_risk_ablation_decisions.csv",
        "static_risk_ablation_summary.csv",
    ):
        provenance[name] = TableProvenance(
            "targeted_recalculation",
            "static_pretrip_synthetic_candidate_decisions_K4_K6_K8",
            "static_deadline_proxy",
            table_sources["static"],
            "One-term removals use the same candidates and deadlines.",
        )
    provenance["inherited_factorial_risk_summary.csv"] = TableProvenance(
        "inherited_reanalysis",
        "step6_static_factorial_variants",
        "legacy_static_factorial",
        table_sources["inherited_risk"],
        "Legacy A-E variants are not all one-term isolations.",
    )
    for name in ("alpha_sweep_decisions.csv", "alpha_sweep_summary.csv"):
        provenance[name] = TableProvenance(
            "targeted_recalculation",
            "static_local_scaled_alpha_sweep_K4_K6_K8",
            "static_alpha_fixed_deadline",
            table_sources["alpha"],
            "Deadline fixed at source alpha=0.10 value.",
        )
    for name in ("risk_weight_sweep_decisions.csv", "risk_weight_sweep_summary.csv"):
        provenance[name] = TableProvenance(
            "targeted_recalculation",
            "static_one_at_a_time_risk_weight_sweep_K4_K6_K8",
            "static_weight_sweep",
            table_sources["weights"],
            "One-at-a-time, not factorial.",
        )
    for name in ("conformal_scope_coverage.csv", "conformal_failure_regions.csv"):
        provenance[name] = TableProvenance(
            "inherited_reanalysis",
            "three_separate_conformal_evaluation_populations",
            "coverage_by_unpooled_source_scope",
            table_sources["coverage"],
            "No source populations are pooled.",
        )
    provenance["prediction_noise_model_metrics.csv"] = TableProvenance(
        "inherited_reanalysis",
        "step4_best_model_held_out_test",
        "prediction_noise_model",
        table_sources["prediction_model"],
        "Scenarios summarized separately.",
    )
    provenance["prediction_noise_interval_metrics.csv"] = TableProvenance(
        "inherited_reanalysis",
        "step8_25_repeated_synthetic_test",
        "prediction_noise_intervals",
        table_sources["prediction_interval"],
        "Prediction-noise scenario was part of this controlled generator scope.",
    )
    for name in (
        "qubo_penalty_sweep.csv",
        "qubo_penalty_aggregate.csv",
        "qubo_penalty_failure_regions.csv",
    ):
        provenance[name] = TableProvenance(
            "targeted_exact_enumeration",
            "step6_variant_D_risk_vectors_K4_K6_K8",
            "qubo_ground_state_penalty_sweep",
            table_sources["penalty"],
            "Unsafe penalties are intentionally evaluated outside the safe constructor.",
        )
    for name in (
        "qaoa_runs.csv",
        "qaoa_aggregate.csv",
        "qaoa_failure_regions.csv",
        "qaoa_targeted_training_history.csv",
    ):
        provenance[name] = TableProvenance(
            "inherited_plus_targeted_ideal_simulator",
            "K4_five_topologies_separate_from_K6_K8_single_grid_scaling",
            "qaoa_scope_specific",
            table_sources["qaoa"],
            "Eight targeted p=3 K6/K8 runs; no hardware and no speed claim.",
        )
    for name in (
        "dynamic_ablation_runs.csv",
        "dynamic_ablation_decisions.csv",
        "dynamic_ablation_summary.csv",
        "dynamic_ablation_paired_deltas.csv",
    ):
        provenance[name] = TableProvenance(
            "targeted_controlled_simulation",
            "five_synthetic_networks_seven_event_rotations",
            "realized_dynamic_ablation",
            table_sources["dynamic_targeted"],
            "Controlled simulated outcomes, not real traffic.",
        )
    provenance["inherited_dynamic_policy_context.csv"] = TableProvenance(
        "inherited_reanalysis",
        "step8_controlled_dynamic_policy_context",
        "inherited_dynamic_context",
        table_sources["dynamic_inherited"],
        "Context only; not a one-term causal ablation.",
    )
    provenance["classical_K_context.csv"] = TableProvenance(
        "inherited_reanalysis",
        "step9_3_static_classical_baselines_K4_K6_K8",
        "classical_K_context",
        table_sources["classical"],
        "Runtime columns excluded; no speed claim.",
    )
    provenance["warm_start_status.csv"] = TableProvenance(
        "status_only",
        "qaoa_warm_start",
        "invalid_comparison_not_executed",
        table_sources["warm"],
        "Warm start is not implemented and was not fabricated.",
    )

    for filename, table in tables.items():
        table.to_csv(output_directory / filename, index=False)

    validation_path = output_directory / "validation_checks.json"
    write_json(validation_path, checks)
    report_path = output_directory / "VALIDATION_REPORT.md"
    report_path.write_text(
        _validation_report(tables, checks), encoding="utf-8"
    )

    common_sources = (CONFIG_PATH, Path(__file__).resolve())
    index_rows: list[dict[str, Any]] = []
    for filename, metadata in provenance.items():
        result_path = output_directory / filename
        sources = tuple(dict.fromkeys((*metadata.sources, *common_sources)))
        index_rows.append(
            {
                "result_id": Path(filename).stem,
                "result_file": _relative(result_path),
                "result_sha256": file_sha256(result_path),
                "row_count": len(tables[filename]),
                "generation_mode": metadata.generation_mode,
                "scope": metadata.scope,
                "comparability_group": metadata.comparability_group,
                "source_artifacts_json": json.dumps(
                    [_relative(path) for path in sources], sort_keys=True
                ),
                "source_sha256_json": json.dumps(
                    {_relative(path): file_sha256(path) for path in sources},
                    sort_keys=True,
                ),
                "notes": metadata.notes,
            }
        )
    all_sources = tuple(dict.fromkeys((*source_files.values(), *common_sources)))
    for result_id, result_path, mode, scope in (
        (
            "validation_checks",
            validation_path,
            "machine_readable_validation",
            "all_step9_4_scopes",
        ),
        (
            "validation_report",
            report_path,
            "human_readable_synthesis",
            "all_step9_4_scopes_without_pooling",
        ),
    ):
        index_rows.append(
            {
                "result_id": result_id,
                "result_file": _relative(result_path),
                "result_sha256": file_sha256(result_path),
                "row_count": None,
                "generation_mode": mode,
                "scope": scope,
                "comparability_group": "cross_scope_index_only_not_metric_pooling",
                "source_artifacts_json": json.dumps(
                    [_relative(path) for path in all_sources], sort_keys=True
                ),
                "source_sha256_json": json.dumps(
                    {_relative(path): file_sha256(path) for path in all_sources},
                    sort_keys=True,
                ),
                "notes": "Synthesis preserves source-scope boundaries.",
            }
        )
    index = pd.DataFrame(index_rows).sort_values("result_id").reset_index(drop=True)
    index_path = output_directory / "ablation_index.csv"
    index.to_csv(index_path, index=False)

    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "qaoa_target_min": min(int(value) for value in config["qaoa"]["target_seeds"]),
            "qaoa_target_max": max(int(value) for value in config["qaoa"]["target_seeds"]),
            "dynamic_base": int(config["dynamic_targeted"]["simulation_seed"]),
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_path": _relative(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "source_artifact_sha256": {
                _relative(path): file_sha256(path) for path in all_sources
            },
            "validation_checks": checks,
            "scope_guards": {
                "metrics_pooled_across_incomparable_scopes": False,
                "static_proxy_labeled_separately_from_dynamic_realized": True,
                "K4_topology_scope_labeled_separately_from_K6_K8_scaling": True,
                "real_topology_data_labeled_as_controlled_simulated_traffic": True,
            },
            "warm_start": {
                "implemented": False,
                "executed": False,
                "fabricated": False,
            },
            "hardware_execution": False,
            "runtime_or_speed_comparison": False,
            "quantum_advantage_claim": False,
            "output_artifact_sha256": {
                _relative(path): file_sha256(path)
                for path in sorted(output_directory.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        }
    )
    write_json(output_directory / "manifest.json", manifest)

    print(static_summary.query("aggregation_scope == 'overall'").to_string(index=False))
    print(qaoa_aggregate.to_string(index=False))
    print(dynamic_summary.query("aggregation_scope == 'overall'").to_string(index=False))
    print(f"Artifacts: {output_directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the named Step 9.4 output files in its dedicated directory",
    )
    run(force=parser.parse_args().force)
