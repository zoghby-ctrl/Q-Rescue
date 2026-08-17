"""Build final paper tables from validated source artifacts without retyping values."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_ARTIFACTS = [
    "results/research/step9_2_topology_uncertainty_qaoa/network_characteristics.csv",
    "results/research/step9_cross_network/point_metrics_overall.csv",
    "results/research/step9_7_adaptive_conformal/coverage_overall.csv",
    "results/research/step9_6_statistics/descriptive_primary.csv",
    "results/research/step8_dynamic_rerouting/dynamic_policy_summary.csv",
    "results/research/step9_6_statistics/dynamic_policy_effects.csv",
    "results/research/step9_6_statistics/major_mixer_effects.csv",
    "results/research/step9_6_statistics/factorial_contrasts.csv",
    "results/research/step9_8_noise/noise_aggregate.csv",
    "results/research/step9_8_noise/noise_paired_effects.csv",
    "results/research/step9_6_statistics/objective_ablation_effects.csv",
    "results/research/step9_7_adaptive_conformal/route_decision_summary.csv",
    "results/research/standard_x_penalty_sensitivity/validation_selection.csv",
    "results/research/standard_x_penalty_sensitivity/test_summary.csv",
    "results/research/targeted_k16_scaling/k16_aggregate.csv",
    "results/research/targeted_k16_scaling/k16_minus_k8_effects.csv",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_csv(name: str, table: pd.DataFrame) -> Path:
    path = OUT / name
    table.to_csv(path, index=False)
    return path


def _display(table: pd.DataFrame) -> str:
    copy = table.copy()
    numeric = copy.select_dtypes(include=[np.number]).columns
    copy[numeric] = copy[numeric].round(4)
    return copy.to_markdown(index=False)


def _write_markdown(name: str, title: str, panels: list[tuple[str, pd.DataFrame]]) -> Path:
    lines = [f"# {title}", "", "Values are display-rounded; linked CSV files preserve full precision."]
    for heading, table in panels:
        lines.extend(["", f"## {heading}", "", _display(table)])
    path = OUT / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def topology_statistics() -> list[Path]:
    source = pd.read_csv(
        ROOT
        / "results/research/step9_2_topology_uncertainty_qaoa/network_characteristics.csv"
    )
    metrics = [
        "node_count",
        "directed_edge_count",
        "directed_simple_density",
        "undirected_average_degree",
        "undirected_degree_variance",
        "undirected_average_shortest_path_length",
        "undirected_average_clustering",
        "route_mean_jaccard_similarity",
    ]
    rows: list[dict[str, object]] = []
    for topology, group in source.groupby("topology", sort=True):
        row: dict[str, object] = {
            "topology": topology,
            "network_count": int(group["network_id"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    table = pd.DataFrame(rows)
    csv_path = _save_csv("table01_topology_statistics.csv", table)
    md_path = _write_markdown(
        "table01_topology_statistics.md",
        "Table 1 — Controlled topology-family statistics",
        [("Fifty primary network instances", table)],
    )
    return [csv_path, md_path]


def prediction_conformal() -> list[Path]:
    point = pd.read_csv(
        ROOT / "results/research/step9_cross_network/point_metrics_overall.csv"
    )
    # Step 9.7 replays the accepted static baselines and adds the adaptive
    # method, so this one source avoids presenting the same static estimates
    # twice under historical and replay labels.
    conformal = pd.read_csv(
        ROOT / "results/research/step9_7_adaptive_conformal/coverage_overall.csv"
    )[
        [
            "evaluation_group",
            "method",
            "n_samples",
            "nominal_coverage",
            "empirical_coverage",
            "average_interval_width",
            "median_interval_width",
        ]
    ].rename(columns={"average_interval_width": "mean_interval_width"})
    conformal = conformal.sort_values(["evaluation_group", "method"]).reset_index(drop=True)
    point_path = _save_csv("table02a_prediction_metrics.csv", point)
    conformal_path = _save_csv("table02b_conformal_metrics.csv", conformal)
    md_path = _write_markdown(
        "table02_prediction_conformal.md",
        "Table 2 — Prediction and conformal calibration",
        [("Point prediction", point), ("Static and adaptive coverage", conformal)],
    )
    return [point_path, conformal_path, md_path]


def qaoa_diagnostics() -> list[Path]:
    source = pd.read_csv(
        ROOT / "results/research/step9_6_statistics/descriptive_primary.csv"
    )
    metrics = [
        "feasibility_probability",
        "optimal_state_probability",
        "optimal_probability_given_feasible",
        "normalized_feasible_entropy",
        "risk_regret",
    ]
    source = source[source["metric"].isin(metrics)]
    rows: list[dict[str, object]] = []
    for keys, group in source.groupby(["K", "depth", "method"], sort=True):
        row: dict[str, object] = {
            "K": int(keys[0]),
            "depth": int(keys[1]),
            "method": keys[2],
            "row_count": int(group["row_count"].min()),
            "independent_network_count": int(group["independent_cluster_count"].min()),
        }
        for metric in metrics:
            value = group[group["metric"] == metric].iloc[0]
            row[f"{metric}_mean"] = float(value["cluster_equal_weight_mean"])
            row[f"{metric}_ci95_lower"] = float(value["cluster_bootstrap_ci95_lower"])
            row[f"{metric}_ci95_upper"] = float(value["cluster_bootstrap_ci95_upper"])
        rows.append(row)
    table = pd.DataFrame(rows)
    csv_path = _save_csv("table03_qaoa_diagnostics.csv", table)
    md_path = _write_markdown(
        "table03_qaoa_diagnostics.md",
        "Table 3 — QAOA feasibility, concentration, entropy, and regret",
        [("Primary factorial summaries", table)],
    )
    return [csv_path, md_path]


def dynamic_outcomes() -> list[Path]:
    policy = pd.read_csv(
        ROOT / "results/research/step8_dynamic_rerouting/dynamic_policy_summary.csv"
    )
    effects = pd.read_csv(
        ROOT / "results/research/step9_6_statistics/dynamic_policy_effects.csv"
    )
    policy_path = _save_csv("table04a_dynamic_policy_outcomes.csv", policy)
    effects_path = _save_csv("table04b_dynamic_matched_effects.csv", effects)
    md_path = _write_markdown(
        "table04_dynamic_outcomes.md",
        "Table 4 — Dynamic operational outcomes",
        [("Policy summaries", policy), ("Matched exploratory effects", effects)],
    )
    return [policy_path, effects_path, md_path]


def statistical_findings() -> list[Path]:
    mixer = pd.read_csv(
        ROOT / "results/research/step9_6_statistics/major_mixer_effects.csv"
    )
    factorial = pd.read_csv(
        ROOT / "results/research/step9_6_statistics/factorial_contrasts.csv"
    )
    factorial = factorial[
        factorial["metric"].isin(
            [
                "feasibility_probability",
                "optimal_state_probability",
                "normalized_feasible_entropy",
                "risk_regret",
                "selected_realized_deadline_violation",
            ]
        )
    ]
    mixer_path = _save_csv("table05a_mixer_effects.csv", mixer)
    factorial_path = _save_csv("table05b_factorial_contrasts.csv", factorial)
    md_path = _write_markdown(
        "table05_statistical_findings.md",
        "Table 5 — Matched cluster-aware statistical findings",
        [("Feasible-XY minus standard-X", mixer), ("Prespecified factorial contrasts", factorial)],
    )
    return [mixer_path, factorial_path, md_path]


def noise_results() -> list[Path]:
    aggregate = pd.read_csv(ROOT / "results/research/step9_8_noise/noise_aggregate.csv")
    effects = pd.read_csv(ROOT / "results/research/step9_8_noise/noise_paired_effects.csv")
    xy_effects = effects[
        (effects["method"] == "feasible_xy")
        & (effects["contrast"] == "noisy_shot_minus_ideal_analytic")
    ]
    aggregate_path = _save_csv("table06a_noise_aggregate.csv", aggregate)
    effect_path = _save_csv("table06b_xy_noise_effects.csv", xy_effects)
    md_path = _write_markdown(
        "table06_noise_robustness.md",
        "Table 6 — Local noise-model robustness",
        [("Ideal, finite-shot, and noisy summaries", aggregate), ("Feasible-XY noisy minus ideal", xy_effects)],
    )
    return [aggregate_path, effect_path, md_path]


def ablation_results() -> list[Path]:
    objective = pd.read_csv(
        ROOT / "results/research/step9_6_statistics/objective_ablation_effects.csv"
    )
    adaptive = pd.read_csv(
        ROOT / "results/research/step9_7_adaptive_conformal/route_decision_summary.csv"
    )
    penalty_validation = pd.read_csv(
        ROOT / "results/research/standard_x_penalty_sensitivity/validation_selection.csv"
    )
    penalty_test = pd.read_csv(
        ROOT / "results/research/standard_x_penalty_sensitivity/test_summary.csv"
    )
    penalty_test = penalty_test[
        penalty_test["metric"].isin(
            [
                "feasibility_probability",
                "optimal_state_probability",
                "optimal_probability_given_feasible",
                "normalized_feasible_entropy",
            ]
        )
    ]
    paths = [
        _save_csv("table07a_objective_ablation.csv", objective),
        _save_csv("table07b_adaptive_route_ablation.csv", adaptive),
        _save_csv("table07c_penalty_validation.csv", penalty_validation),
        _save_csv("table07d_penalty_test.csv", penalty_test),
    ]
    paths.append(
        _write_markdown(
            "table07_ablation_summary.md",
            "Table 7 — Objective, calibration, and penalty ablations",
            [
                ("Objective A/B/C effects", objective),
                ("Adaptive route decisions", adaptive),
                ("Penalty validation selection", penalty_validation),
                ("Complete held-out penalty grid", penalty_test),
            ],
        )
    )
    return paths


def k16_supplement() -> list[Path]:
    aggregate = pd.read_csv(
        ROOT / "results/research/targeted_k16_scaling/k16_aggregate.csv"
    )
    effects = pd.read_csv(
        ROOT / "results/research/targeted_k16_scaling/k16_minus_k8_effects.csv"
    )
    aggregate_path = _save_csv("tableS01a_k16_aggregate.csv", aggregate)
    effects_path = _save_csv("tableS01b_k16_minus_k8.csv", effects)
    md_path = _write_markdown(
        "tableS01_k16_scaling.md",
        "Supplementary Table S1 — Indicative K=16 scaling",
        [("K=16 summaries", aggregate), ("Matched K=16 minus K=8", effects)],
    )
    return [aggregate_path, effects_path, md_path]


def main() -> None:
    builders = [
        topology_statistics,
        prediction_conformal,
        qaoa_diagnostics,
        dynamic_outcomes,
        statistical_findings,
        noise_results,
        ablation_results,
        k16_supplement,
    ]
    paths: list[Path] = []
    for builder in builders:
        paths.extend(builder())
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": _sha256(Path(__file__)),
        "source_sha256": {
            name: _sha256(ROOT / name) for name in SOURCE_ARTIFACTS
        },
        "artifact_sha256": {
            path.name: _sha256(path) for path in sorted(set(paths))
        },
        "manual_numeric_inputs": False,
    }
    (OUT / "science_freeze_table_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(paths)} science-freeze table artifacts to {OUT}")


if __name__ == "__main__":
    main()
