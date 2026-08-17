"""Evaluate static absolute/local split conformal against one ACI baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.adaptive_conformal import AdaptiveMiscoverageConformal  # noqa: E402
from src.controlled_objectives import controlled_objective_table  # noqa: E402
from src.reproducibility import (  # noqa: E402
    build_experiment_manifest,
    file_sha256,
    write_json,
)
from src.statistical_inference import (  # noqa: E402
    percentile_bootstrap_interval,
    wilson_interval,
)
from src.step9_source_replay import regenerate_step9_observations  # noqa: E402


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "adaptive_conformal_baseline.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step9_7_adaptive_conformal"


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _source_path(config: dict[str, Any], key: str) -> Path:
    path = (REPOSITORY_ROOT / config["source_paths"][key]).resolve()
    if REPOSITORY_ROOT.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _verify_sources(config: dict[str, Any]) -> dict[str, str]:
    manifest_path = _source_path(config, "step9_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_hashes = dict(manifest["artifact_sha256"])
    verified: dict[str, str] = {}
    for key in [
        "traffic_observations",
        "conformal_models",
        "existing_intervals",
        "route_candidates",
        "effective_config",
    ]:
        path = _source_path(config, key)
        actual = file_sha256(path)
        if artifact_hashes.get(path.name) != actual:
            raise RuntimeError(f"inherited Step 9 source hash mismatch: {path.name}")
        verified[path.name] = actual
    verified[manifest_path.name] = file_sha256(manifest_path)
    source_hashes = dict(manifest.get("source_sha256", {}))
    for relative, expected in source_hashes.items():
        if not str(relative).endswith(".py"):
            continue
        path = REPOSITORY_ROOT / str(relative)
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"inherited Step 9 source hash mismatch: {relative}")
        verified[f"source::{relative}"] = actual
    return verified


def _regenerate_step9_observations(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    return regenerate_step9_observations(
        _source_path(config, "effective_config"),
        _source_path(config, "traffic_observations"),
    )


def _adaptive_state(config: dict[str, Any], scores: np.ndarray) -> AdaptiveMiscoverageConformal:
    return AdaptiveMiscoverageConformal(
        scores,
        target_alpha=float(config["target_alpha"]),
        gamma=float(config["gamma"]),
        minimum_alpha=float(config["minimum_alpha"]),
        maximum_alpha=float(config["maximum_alpha"]),
    )


def _interval_row(
    row: pd.Series,
    *,
    method: str,
    prediction: float,
    half_width: float,
    alpha_before: float,
    alpha_after: float,
    quantile: float,
    scale: float,
) -> dict[str, object]:
    actual = float(row["travel_time_minutes"])
    lower = max(0.0, prediction - half_width)
    upper = prediction + half_width
    return {
        "evaluation_group": row["evaluation_group"],
        "method": method,
        "observation_id": row["observation_id"],
        "network_id": row["network_id"],
        "topology": row["topology"],
        "scenario": row["scenario"],
        "sequence_index_within_network": int(row["sequence_index_within_network"]),
        "prediction": prediction,
        "travel_time_minutes": actual,
        "residual_scale_minutes": scale,
        "score_quantile": half_width if method == "absolute_split" else quantile,
        "interval_half_width": half_width,
        "lower_bound": lower,
        "upper_bound": upper,
        "interval_width": upper - lower,
        "covered": lower <= actual <= upper,
        "alpha_before": alpha_before,
        "alpha_after": alpha_after,
    }


def _coverage_intervals(
    config: dict[str, Any], observations: pd.DataFrame, models: dict[object, object]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    absolute_model = models[("absolute", float(config["target_alpha"]))]
    local_model = models[("local", float(config["target_alpha"]))]
    base_model = local_model.base_model
    scale_model = local_model.scale_model
    calibration = observations[observations["dataset_group"] == "calibration"].copy()
    calibration_prediction = np.asarray(base_model.predict(calibration), dtype=float)
    calibration_scale = np.asarray(scale_model.predict_scale(calibration), dtype=float)
    calibration_scores = (
        np.abs(calibration["travel_time_minutes"].to_numpy(dtype=float) - calibration_prediction)
        / calibration_scale
    )
    if not np.isclose(
        local_model.calibration_quantile,
        _adaptive_state(config, calibration_scores).current_quantile(),
    ):
        raise RuntimeError("reconstructed local calibration scores do not match artifact")

    interval_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for evaluation_group, dataset_group in config["evaluation_groups"].items():
        evaluation = observations[observations["dataset_group"] == dataset_group].copy()
        if evaluation.empty:
            raise RuntimeError(f"empty evaluation group: {evaluation_group}")
        evaluation["evaluation_group"] = evaluation_group
        evaluation = evaluation.sort_values(
            ["network_id", "observation_id"], kind="stable"
        )
        evaluation["sequence_index_within_network"] = evaluation.groupby(
            "network_id", sort=False
        ).cumcount()
        predictions = np.asarray(base_model.predict(evaluation), dtype=float)
        scales = np.asarray(scale_model.predict_scale(evaluation), dtype=float)
        evaluation["_prediction"] = predictions
        evaluation["_scale"] = scales
        for _, row in evaluation.iterrows():
            prediction = float(row["_prediction"])
            scale = float(row["_scale"])
            for method, half_width, quantile in [
                (
                    "absolute_split",
                    float(absolute_model.calibration_quantile),
                    float(absolute_model.calibration_quantile),
                ),
                (
                    "local_scaled_split",
                    float(local_model.calibration_quantile) * scale,
                    float(local_model.calibration_quantile),
                ),
            ]:
                interval_rows.append(
                    _interval_row(
                        row,
                        method=method,
                        prediction=prediction,
                        half_width=half_width,
                        alpha_before=float(config["target_alpha"]),
                        alpha_after=float(config["target_alpha"]),
                        quantile=quantile,
                        scale=scale,
                    )
                )
        for network_id, stream in evaluation.groupby("network_id", sort=True):
            adaptive = _adaptive_state(config, calibration_scores)
            for _, row in stream.iterrows():
                prediction = float(row["_prediction"])
                scale = float(row["_scale"])
                quantile = adaptive.current_quantile()
                half_width = quantile * scale
                score = abs(float(row["travel_time_minutes"]) - prediction) / scale
                alpha_before = float(adaptive.alpha_t)
                trace = adaptive.observe(score, quantile_used=quantile)
                interval_rows.append(
                    _interval_row(
                        row,
                        method="adaptive_local_aci",
                        prediction=prediction,
                        half_width=half_width,
                        alpha_before=alpha_before,
                        alpha_after=float(adaptive.alpha_t),
                        quantile=quantile,
                        scale=scale,
                    )
                )
                trace_rows.append(
                    {
                        "evaluation_group": evaluation_group,
                        "network_id": network_id,
                        "observation_id": row["observation_id"],
                        "sequence_index_within_network": int(
                            row["sequence_index_within_network"]
                        ),
                        **trace,
                    }
                )
    intervals = pd.DataFrame(interval_rows).sort_values(
        ["evaluation_group", "network_id", "observation_id", "method"],
        kind="stable",
    )
    trace = pd.DataFrame(trace_rows).sort_values(
        ["evaluation_group", "network_id", "sequence_index_within_network"],
        kind="stable",
    )
    calibration_details = {
        "calibration_count": int(len(calibration_scores)),
        "absolute_quantile_minutes": float(absolute_model.calibration_quantile),
        "local_score_quantile": float(local_model.calibration_quantile),
        "calibration_score_mean": float(np.mean(calibration_scores)),
        "calibration_score_median": float(np.median(calibration_scores)),
    }
    return intervals.reset_index(drop=True), trace.reset_index(drop=True), calibration_details


def _verify_static_reproduction(
    intervals: pd.DataFrame, existing_path: Path, alpha: float
) -> dict[str, float]:
    existing = pd.read_csv(existing_path)
    existing = existing[
        (existing["configured_alpha"] == alpha)
        & (existing["method"].isin(["absolute", "local"]))
    ].copy()
    method_map = {"absolute": "absolute_split", "local": "local_scaled_split"}
    existing["method"] = existing["method"].map(method_map)
    columns = ["evaluation_group", "observation_id", "method"]
    static = intervals[intervals["method"] != "adaptive_local_aci"]
    merged = static.merge(
        existing[columns + ["lower_bound", "upper_bound", "interval_half_width"]],
        on=columns,
        suffixes=("_new", "_existing"),
        validate="one_to_one",
    )
    if len(merged) != len(static):
        raise RuntimeError("static interval reproduction rows did not match")
    differences = {
        name: float(np.max(np.abs(merged[f"{name}_new"] - merged[f"{name}_existing"])))
        for name in ["lower_bound", "upper_bound", "interval_half_width"]
    }
    if max(differences.values()) > 1e-9:
        raise RuntimeError(f"static interval reproduction failed: {differences}")
    return differences


def _coverage_summary(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = (
        intervals.groupby(["evaluation_group", "method"], sort=True)
        .agg(
            n_samples=("covered", "size"),
            empirical_coverage=("covered", "mean"),
            average_interval_width=("interval_width", "mean"),
            median_interval_width=("interval_width", "median"),
            average_alpha=("alpha_before", "mean"),
            final_alpha_mean_across_rows=("alpha_after", "mean"),
        )
        .reset_index()
    )
    interval_rows: list[dict[str, object]] = []
    for row in aggregate.itertuples(index=False):
        subset = intervals[
            (intervals["evaluation_group"] == row.evaluation_group)
            & (intervals["method"] == row.method)
        ]
        successes = int(subset["covered"].sum())
        _, lower, upper = wilson_interval(successes, len(subset))
        interval_rows.append(
            {
                **row._asdict(),
                "nominal_coverage": 0.90,
                "covered_count": successes,
                "wilson_ci95_lower": lower,
                "wilson_ci95_upper": upper,
            }
        )
    by_network = (
        intervals.groupby(["evaluation_group", "method", "network_id"], sort=True)
        .agg(
            n_samples=("covered", "size"),
            empirical_coverage=("covered", "mean"),
            average_interval_width=("interval_width", "mean"),
            median_interval_width=("interval_width", "median"),
            final_alpha=("alpha_after", "last"),
        )
        .reset_index()
    )
    return pd.DataFrame(interval_rows), by_network


def _route_decisions(
    config: dict[str, Any],
    route_candidates: pd.DataFrame,
    models: dict[object, object],
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha = float(config["target_alpha"])
    absolute_q = float(models[("absolute", alpha)].calibration_quantile)
    local_q = float(models[("local", alpha)].calibration_quantile)
    calibration = observations[observations["dataset_group"] == "calibration"]
    local_model = models[("local", alpha)]
    calibration_scores = (
        np.abs(
            calibration["travel_time_minutes"].to_numpy(dtype=float)
            - np.asarray(local_model.base_model.predict(calibration), dtype=float)
        )
        / np.asarray(local_model.scale_model.predict_scale(calibration), dtype=float)
    )
    objective = dict(config["objective"])
    candidates_out: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    feedback_rows: list[dict[str, object]] = []
    route_candidates = route_candidates.sort_values(
        ["evaluation_group", "network_id", "event_id", "route_id"], kind="stable"
    )
    for (evaluation_group, network_id), stream in route_candidates.groupby(
        ["evaluation_group", "network_id"], sort=True
    ):
        adaptive = _adaptive_state(config, calibration_scores)
        for event_id, event in stream.groupby("event_id", sort=True):
            event = event.sort_values("route_id", kind="stable").reset_index(drop=True)
            if len(event) < 2:
                raise RuntimeError("route event has fewer than two candidates")
            predictions = event["prediction"].to_numpy(dtype=float)
            actuals = event["actual_travel_time_minutes"].to_numpy(dtype=float)
            scales = event["interval_half_width"].to_numpy(dtype=float) / local_q
            deadline = float(event["deadline_minutes"].iloc[0])
            adaptive_q = adaptive.current_quantile()
            widths = {
                "absolute_split": np.full(len(event), absolute_q, dtype=float),
                "local_scaled_split": local_q * scales,
                "adaptive_local_aci": adaptive_q * scales,
            }
            selections: dict[str, int] = {}
            scores: dict[str, np.ndarray] = {}
            for method in config["methods"]:
                table = controlled_objective_table(
                    predictions,
                    widths[method],
                    deadline_minutes=deadline,
                    lambda_deadline=float(objective["lambda_deadline"]),
                    lambda_uncertainty=float(objective["lambda_uncertainty"]),
                )
                vector = table[
                    "C_expected_plus_deadline_plus_uncertainty"
                ].to_numpy(dtype=float)
                selected = int(np.argmin(vector))
                selections[method] = selected
                scores[method] = vector
            expected_index = int(np.argmin(predictions))
            actual_fastest = int(np.argmin(actuals))
            local_index = selections["local_scaled_split"]
            for method in config["methods"]:
                selected = selections[method]
                covered = np.abs(actuals - predictions) <= widths[method]
                for index, row in event.iterrows():
                    candidates_out.append(
                        {
                            "evaluation_group": evaluation_group,
                            "network_id": network_id,
                            "topology": row["topology"],
                            "scenario": row["scenario"],
                            "event_id": event_id,
                            "route_id": row["route_id"],
                            "method": method,
                            "prediction": predictions[index],
                            "residual_scale_minutes": scales[index],
                            "interval_half_width": widths[method][index],
                            "deadline_minutes": deadline,
                            "controlled_risk_score": scores[method][index],
                            "actual_travel_time_minutes": actuals[index],
                            "covered": bool(covered[index]),
                            "selected": index == selected,
                        }
                    )
                selected_actual = float(actuals[selected])
                decision_rows.append(
                    {
                        "evaluation_group": evaluation_group,
                        "network_id": network_id,
                        "topology": event["topology"].iloc[0],
                        "scenario": event["scenario"].iloc[0],
                        "event_id": event_id,
                        "method": method,
                        "selected_route_id": event.loc[selected, "route_id"],
                        "expected_time_route_id": event.loc[expected_index, "route_id"],
                        "local_scaled_route_id": event.loc[local_index, "route_id"],
                        "selection_changed_from_expected": selected != expected_index,
                        "selection_changed_from_local_scaled": selected != local_index,
                        "selected_actual_travel_time_minutes": selected_actual,
                        "actual_deadline_violation": selected_actual > deadline,
                        "actual_deadline_margin_minutes": deadline - selected_actual,
                        "actual_regret_vs_fastest_minutes": selected_actual
                        - float(actuals[actual_fastest]),
                        "selected_modeled_risk": float(scores[method][selected]),
                        "adaptive_alpha_before": (
                            float(adaptive.alpha_t)
                            if method == "adaptive_local_aci"
                            else alpha
                        ),
                    }
                )
            # Only the selected adaptive route outcome becomes available.  Its
            # update occurs after every method's interval and decision is fixed.
            adaptive_selected = selections["adaptive_local_aci"]
            selected_score = abs(
                actuals[adaptive_selected] - predictions[adaptive_selected]
            ) / scales[adaptive_selected]
            trace = adaptive.observe(selected_score, quantile_used=adaptive_q)
            feedback_rows.append(
                {
                    "evaluation_group": evaluation_group,
                    "network_id": network_id,
                    "event_id": event_id,
                    "selected_route_id": event.loc[adaptive_selected, "route_id"],
                    **trace,
                }
            )
    candidate_table = pd.DataFrame(candidates_out)
    decisions = pd.DataFrame(decision_rows)
    feedback = pd.DataFrame(feedback_rows)
    summary = (
        decisions.groupby(["evaluation_group", "method"], sort=True)
        .agg(
            event_count=("event_id", "size"),
            network_count=("network_id", "nunique"),
            deadline_violation_rate=("actual_deadline_violation", "mean"),
            mean_actual_travel_time_minutes=(
                "selected_actual_travel_time_minutes",
                "mean",
            ),
            median_actual_travel_time_minutes=(
                "selected_actual_travel_time_minutes",
                "median",
            ),
            mean_deadline_margin_minutes=("actual_deadline_margin_minutes", "mean"),
            mean_actual_regret_vs_fastest_minutes=(
                "actual_regret_vs_fastest_minutes",
                "mean",
            ),
            route_change_from_expected_rate=(
                "selection_changed_from_expected",
                "mean",
            ),
            route_change_from_local_scaled_rate=(
                "selection_changed_from_local_scaled",
                "mean",
            ),
        )
        .reset_index()
    )
    return candidate_table, decisions, summary, feedback


def _decision_differences(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [
        "actual_deadline_violation",
        "selected_actual_travel_time_minutes",
        "actual_deadline_margin_minutes",
        "actual_regret_vs_fastest_minutes",
    ]
    for evaluation_group, subset in decisions.groupby("evaluation_group", sort=True):
        for comparison in ["absolute_split", "local_scaled_split"]:
            for metric in metrics:
                wide = subset.pivot(
                    index=["network_id", "event_id"], columns="method", values=metric
                )
                difference = (
                    wide["adaptive_local_aci"].astype(float)
                    - wide[comparison].astype(float)
                )
                network_difference = difference.groupby(level="network_id").mean()
                estimate, lower, upper = percentile_bootstrap_interval(
                    network_difference.to_numpy(dtype=float),
                    resamples=20_000,
                    seed=970000 + len(rows),
                )
                rows.append(
                    {
                        "evaluation_group": evaluation_group,
                        "contrast": f"adaptive_local_aci_minus_{comparison}",
                        "metric": metric,
                        "event_count": len(difference),
                        "network_count": len(network_difference),
                        "mean_network_difference": estimate,
                        "cluster_bootstrap_ci95_lower": lower,
                        "cluster_bootstrap_ci95_upper": upper,
                    }
                )
    return pd.DataFrame(rows)


def _report(
    coverage: pd.DataFrame,
    route_summary: pd.DataFrame,
    reproduction: dict[str, float],
    config: dict[str, Any],
) -> str:
    coverage_pivot = coverage.pivot(
        index="evaluation_group", columns="method", values="empirical_coverage"
    )
    adaptive_routes = route_summary[route_summary["method"] == "adaptive_local_aci"]
    return f"""# Validation Report — Step 9.7 Adaptive Conformal Baseline

## Status

- Status: **PASS** for the frozen ACI-style sequential baseline.
- Methods: absolute split conformal, locally scaled split conformal, and adaptive local ACI.
- Target coverage: {1.0 - float(config['target_alpha']):.2f}; adaptation rate gamma: {float(config['gamma']):.3f}, fixed before evaluation.
- Point model, residual-scale model, and calibration-score pool remain frozen.
- Original pre-CSV Step 9 floating-point observations are deterministically regenerated from hash-verified sources; saved static intervals reproduce within tolerance.
- Each current interval is constructed before its current outcome updates the state.
- Maximum static-artifact reproduction error: {max(reproduction.values()):.3e}.

## Coverage

{coverage_pivot.to_markdown(floatfmt='.4f')}

## Route-decision scope

The controlled route objective holds the point-prediction deadline term fixed
and changes only interval half-width between methods. Adaptive route feedback
uses only the selected adaptive route after the decision; unselected candidate
outcomes are not used to update future decisions. Across the four evaluation
groups, adaptive decisions cover {int(adaptive_routes['event_count'].sum())}
controlled route events. Detailed violation, width, route-change, and realized
regret metrics are saved in CSV form.

## Limitations

1. This is one ACI-style baseline, not an adaptive-conformal benchmark suite.
2. Test streams are controlled synthetic sequences in stable recorded order; order sensitivity is intrinsic to online adaptation.
3. The update changes nominal alpha but does not repair a misspecified point or scale model.
4. Route evaluation has four events per network (16 per evaluation group), so network-cluster intervals are exploratory.
5. Coverage under arbitrary distribution shift is not guaranteed or claimed.
"""


def main() -> None:
    config = _load_config()
    verified = _verify_sources(config)
    observations, regeneration_checks = _regenerate_step9_observations(config)
    models = joblib.load(_source_path(config, "conformal_models"))
    intervals, adaptive_trace, calibration_details = _coverage_intervals(
        config, observations, models
    )
    reproduction = _verify_static_reproduction(
        intervals,
        _source_path(config, "existing_intervals"),
        float(config["target_alpha"]),
    )
    coverage, coverage_network = _coverage_summary(intervals)
    route_candidates = pd.read_csv(_source_path(config, "route_candidates"))
    route_candidate_results, decisions, route_summary, route_feedback = _route_decisions(
        config, route_candidates, models, observations
    )
    decision_differences = _decision_differences(decisions)
    if len(intervals) != 4 * 1344 * 3 or len(decisions) != 4 * 16 * 3:
        raise RuntimeError("unexpected Step 9.7 sample size")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, pd.DataFrame] = {
        "coverage_intervals.csv": intervals,
        "adaptive_update_trace.csv": adaptive_trace,
        "coverage_overall.csv": coverage,
        "coverage_by_network.csv": coverage_network,
        "route_candidate_intervals.csv": route_candidate_results,
        "route_decisions.csv": decisions,
        "route_decision_summary.csv": route_summary,
        "route_decision_differences.csv": decision_differences,
        "adaptive_route_feedback_trace.csv": route_feedback,
    }
    for name, table in artifacts.items():
        table.to_csv(OUTPUT_DIRECTORY / name, index=False)
    write_json(OUTPUT_DIRECTORY / "calibration_details.json", calibration_details)
    write_json(OUTPUT_DIRECTORY / "static_reproduction_checks.json", reproduction)
    write_json(OUTPUT_DIRECTORY / "source_regeneration_checks.json", regeneration_checks)
    report = _report(coverage, route_summary, reproduction, config)
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={"decision_bootstrap_seed_base": 970000},
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "source_sha256": verified
            | {
                "runner": file_sha256(Path(__file__)),
                "adaptive_conformal": file_sha256(
                    REPOSITORY_ROOT / "src" / "adaptive_conformal.py"
                ),
                "controlled_objectives": file_sha256(
                    REPOSITORY_ROOT / "src" / "controlled_objectives.py"
                ),
                "step9_source_replay": file_sha256(
                    REPOSITORY_ROOT / "src" / "step9_source_replay.py"
                ),
            },
            "artifact_sha256": {
                name: file_sha256(OUTPUT_DIRECTORY / name) for name in artifacts
            }
            | {
                name: file_sha256(OUTPUT_DIRECTORY / name)
                for name in [
                    "calibration_details.json",
                    "static_reproduction_checks.json",
                    "source_regeneration_checks.json",
                    "VALIDATION_REPORT.md",
                ]
            },
            "claim_exclusions": [
                "universal_shift_coverage",
                "adaptive_method_optimality",
                "real_traffic_validation",
            ],
        }
    )
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    print(
        f"Step 9.7 complete: {len(intervals)} intervals and {len(decisions)} route decisions.",
        flush=True,
    )


if __name__ == "__main__":
    main()
