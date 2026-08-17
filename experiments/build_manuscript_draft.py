"""Render the science-freeze manuscript from audited machine-readable results."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "paper" / "MANUSCRIPT_TEMPLATE.md"
OUTPUT = ROOT / "paper" / "MANUSCRIPT_DRAFT.md"
ABSTRACT_TEMPLATE = ROOT / "paper" / "ABSTRACT_TEMPLATE.md"
ABSTRACT_OUTPUT = ROOT / "paper" / "ABSTRACT_DRAFT.md"
MANIFEST = ROOT / "paper" / "MANUSCRIPT_DRAFT_MANIFEST.json"

SOURCES = {
    "step4": "results/research/step4_traffic_ai/model_metrics_overall.csv",
    "cross_point": "results/research/step9_cross_network/point_metrics_overall.csv",
    "cross_conformal": "results/research/step9_cross_network/conformal_metrics_overall.csv",
    "lono": "results/research/step9_1_leave_one_network_out/conformal_metrics_aggregate.csv",
    "real_point": "results/research/step8_5_real_topology/model_metrics_overall.csv",
    "real_conformal": "results/research/step8_5_real_topology/conformal_coverage_overall.csv",
    "primary_runs": "results/research/step9_2_topology_uncertainty_qaoa/qaoa_runs.csv",
    "major_mixer": "results/research/step9_6_statistics/major_mixer_effects.csv",
    "factorial": "results/research/step9_6_statistics/factorial_contrasts.csv",
    "entropy": "results/research/step9_6_statistics/entropy_associations.csv",
    "objective": "results/research/step9_6_statistics/objective_ablation_effects.csv",
    "dynamic_summary": "results/research/step8_dynamic_rerouting/dynamic_policy_summary.csv",
    "dynamic_effects": "results/research/step9_6_statistics/dynamic_policy_effects.csv",
    "adaptive_coverage": "results/research/step9_7_adaptive_conformal/coverage_overall.csv",
    "adaptive_routes": "results/research/step9_7_adaptive_conformal/route_decision_differences.csv",
    "noise": "results/research/step9_8_noise/noise_paired_effects.csv",
    "k16_aggregate": "results/research/targeted_k16_scaling/k16_aggregate.csv",
    "k16_effects": "results/research/targeted_k16_scaling/k16_minus_k8_effects.csv",
    "penalty_selection": "results/research/standard_x_penalty_sensitivity/selected_penalty.json",
    "penalty_validation": "results/research/standard_x_penalty_sensitivity/validation_selection.csv",
    "penalty_summary": "results/research/standard_x_penalty_sensitivity/test_summary.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def row(table: pd.DataFrame, **filters: object) -> pd.Series:
    selected = table
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def effect_tokens(prefix: str, record: pd.Series, *, value: str = "mean_difference") -> dict[str, str]:
    return {
        prefix: f(record[value]),
        f"{prefix}_LO": f(record["ci95_lower"]),
        f"{prefix}_HI": f(record["ci95_upper"]),
    }


def build_tokens() -> dict[str, str]:
    tables: dict[str, pd.DataFrame] = {
        key: pd.read_csv(ROOT / path)
        for key, path in SOURCES.items()
        if Path(path).suffix == ".csv"
    }
    primary = tables["primary_runs"]
    if len(primary) != 1_800 or primary["network_id"].nunique() != 50:
        raise RuntimeError("primary design identity check failed")
    if len(tables["adaptive_coverage"]) != 12:
        raise RuntimeError("adaptive coverage design identity check failed")

    tokens: dict[str, str] = {}

    step4 = row(
        tables["step4"], model_name="hist_gradient_boosting", split="test"
    )
    tokens.update(
        STEP4_MAE=f(step4["mae"]),
        STEP4_RMSE=f(step4["rmse"]),
        STEP4_R2=f(step4["r2"]),
        STEP4_N=str(int(step4["n_samples"])),
    )
    point_id = row(tables["cross_point"], evaluation_group="TEST-ID")
    tokens.update(ID_MAE=f(point_id["mae"]), ID_R2=f(point_id["r2"]))

    static = tables["cross_conformal"]
    static = static[
        (static["method"] == "local")
        & np.isclose(static["configured_alpha"], 0.10)
    ]
    adaptive = tables["adaptive_coverage"]
    group_names = {
        "ID": "TEST-ID",
        "TOPO": "OOD-topology",
        "TRAFFIC": "OOD-traffic",
        "COMBINED": "OOD-combined",
    }
    for short, group in group_names.items():
        static_row = row(static, evaluation_group=group)
        adaptive_row = row(
            adaptive, evaluation_group=group, method="adaptive_local_aci"
        )
        tokens[f"LOCAL_{short}_COV"] = f(static_row["empirical_coverage"])
        tokens[f"LOCAL_{short}_WIDTH"] = f(static_row["mean_interval_width"])
        tokens[f"ADAPT_{short}_COV"] = f(adaptive_row["empirical_coverage"])
        tokens[f"ADAPT_{short}_WIDTH"] = f(
            adaptive_row["average_interval_width"]
        )
    adaptive_vs_local = tables["adaptive_routes"]
    local_contrast = adaptive_vs_local[
        adaptive_vs_local["contrast"]
        == "adaptive_local_aci_minus_local_scaled_split"
    ]
    if not np.allclose(
        local_contrast["mean_network_difference"].fillna(0.0), 0.0
    ):
        raise RuntimeError("manuscript route-identity statement is no longer true")

    real_point = tables["real_point"].iloc[0]
    real_conformal = tables["real_conformal"].iloc[0]
    lono_local = row(tables["lono"], method="local")
    tokens.update(
        REAL_R2=f(real_point["r2"]),
        REAL_COV=f(real_conformal["empirical_coverage"]),
        LONO_LOCAL_COV=f(lono_local["empirical_coverage"]),
    )

    xy = primary[primary["method"] == "feasible_xy"]
    x = primary[primary["method"] == "standard_x"]
    tokens.update(
        XY_PFEAS_MEAN=f(xy["feasibility_probability"].mean()),
        X_PFEAS_MEAN=f(x["feasibility_probability"].mean()),
        XY_POPT_MEAN=f(xy["optimal_state_probability"].mean()),
    )
    for depth in [1, 3, 5]:
        subset = xy[(xy["K"] == 8) & (xy["depth"] == depth)]
        tokens[f"K8_P{depth}_POPT"] = f(subset["optimal_state_probability"].mean())

    major = tables["major_mixer"]
    for metric, prefix in [
        ("feasibility_probability", "MIXER_PFEAS_DIFF"),
        ("optimal_state_probability", "MIXER_POPT_DIFF"),
        ("optimal_probability_given_feasible", "MIXER_COND_DIFF"),
    ]:
        tokens.update(effect_tokens(prefix, row(major, metric=metric)))

    factorial = tables["factorial"]
    factorial_specs = [
        ("optimal_state_probability", "K_8_minus_4", "K_EFFECT_POPT"),
        ("normalized_feasible_entropy", "K_8_minus_4", "K_EFFECT_ENT"),
        ("optimal_state_probability", "uncertainty_high_minus_low", "UNC_EFFECT_POPT"),
        ("normalized_feasible_entropy", "uncertainty_high_minus_low", "UNC_EFFECT_ENT"),
        ("optimal_state_probability", "depth_5_minus_1", "DEPTH_POPT"),
        ("normalized_feasible_entropy", "depth_5_minus_1", "DEPTH_ENT"),
        ("selected_realized_deadline_violation", "depth_5_minus_1", "DEPTH_VIOL"),
        ("selected_realized_regret_vs_actual_fastest_minutes", "depth_5_minus_1", "DEPTH_REALREG"),
    ]
    for metric, contrast, prefix in factorial_specs:
        tokens.update(
            effect_tokens(
                prefix,
                row(
                    factorial,
                    metric=metric,
                    contrast=contrast,
                    method_scope="feasible_xy",
                ),
            )
        )

    entropy = tables["entropy"]
    entropy_specs = [
        ("optimal_state_probability", "ENT_POPT_RHO"),
        ("risk_regret", "ENT_RISK_RHO"),
        ("selected_realized_deadline_violation", "ENT_VIOL_RHO"),
        ("selected_realized_regret_vs_actual_fastest_minutes", "ENT_REALREG_RHO"),
    ]
    for outcome, prefix in entropy_specs:
        record = row(entropy, outcome=outcome)
        tokens[prefix] = f(record["mean_within_network_spearman"])
        tokens[f"{prefix}_LO"] = f(record["ci95_lower"])
        tokens[f"{prefix}_HI"] = f(record["ci95_upper"])

    objective = tables["objective"]
    c_prefix = "C_expected_plus_deadline_plus_uncertainty_minus_B_expected_plus_deadline"
    selection_change = row(
        objective, comparison=c_prefix, metric="route_selection_change_rate"
    )["mean_difference"]
    tokens["C_SELECTION_CHANGE_PCT"] = f"{100.0 * float(selection_change):.1f}%"
    tokens.update(
        effect_tokens(
            "C_TIME_DIFF",
            row(objective, comparison=c_prefix, metric="realized_travel_time_minutes"),
        )
    )
    tokens.update(
        effect_tokens(
            "C_VIOL_DIFF",
            row(objective, comparison=c_prefix, metric="realized_deadline_violation"),
        )
    )

    k16 = tables["k16_aggregate"]
    for depth in [1, 3]:
        record = row(k16, depth=depth, method="feasible_xy")
        tokens[f"K16_P{depth}_POPT"] = f(
            record["optimal_state_probability_mean"]
        )
        tokens[f"K16_P{depth}_ENT"] = f(
            record["normalized_feasible_entropy_mean"]
        )
    k16_effects = tables["k16_effects"]
    for metric, prefix in [
        ("optimal_state_probability", "K16_POPT_DIFF"),
        ("normalized_feasible_entropy", "K16_ENT_DIFF"),
    ]:
        record = row(k16_effects, method="feasible_xy", metric=metric)
        tokens[prefix] = f(record["mean_network_difference"])
        tokens[f"{prefix}_LO"] = f(record["ci95_lower"])
        tokens[f"{prefix}_HI"] = f(record["ci95_upper"])

    selection = json.loads(
        (ROOT / SOURCES["penalty_selection"]).read_text(encoding="utf-8")
    )["selected_penalty_condition"]
    penalty = tables["penalty_summary"]
    eligible_conditions = tables["penalty_validation"].loc[
        tables["penalty_validation"][
            "global_ground_states_all_feasible_every_problem"
        ].astype(bool),
        "penalty_condition",
    ]
    tokens["PENALTY_SELECTED"] = str(selection)
    for condition, label in [
        (selection, "PENALTY_SELECTED"),
        ("validated_safe_default", "PENALTY_DEFAULT"),
    ]:
        tokens[f"{label}_PFEAS"] = f(
            row(penalty, penalty_condition=condition, metric="feasibility_probability")[
                "mean"
            ]
        )
        tokens[f"{label}_POPT"] = f(
            row(penalty, penalty_condition=condition, metric="optimal_state_probability")[
                "mean"
            ]
        )
    valid_test = penalty[penalty["penalty_condition"].isin(eligible_conditions)]
    tokens["PENALTY_VALID_MAX_PFEAS"] = f(
        valid_test[valid_test["metric"] == "feasibility_probability"]["mean"].max()
    )
    tokens["PENALTY_VALID_MAX_POPT"] = f(
        valid_test[valid_test["metric"] == "optimal_state_probability"]["mean"].max()
    )

    dynamic = tables["dynamic_summary"]
    for policy, label in [
        ("expected_time_rerouting", "EXPECT"),
        ("uncertainty_aware_classical", "CLASSICAL"),
        ("uncertainty_aware_qaoa", "QAOA"),
    ]:
        record = row(dynamic, policy=policy)
        tokens[f"DYN_{label}_VIOL"] = f(record["deadline_violation_rate"])
        tokens[f"DYN_{label}_REROUTES"] = f(record["mean_reroutes"])
    dynamic_effects = tables["dynamic_effects"]
    for metric, prefix in [
        ("deadline_violation", "DYN_DIFF_VIOL"),
        ("arrival_time_minutes", "DYN_DIFF_ARRIVAL"),
    ]:
        tokens.update(effect_tokens(prefix, row(dynamic_effects, metric=metric)))

    noise = tables["noise"]
    for depth in [1, 3]:
        for metric, suffix in [
            ("feasibility_probability", "PFEAS"),
            ("optimal_state_probability", "POPT"),
        ]:
            record = row(
                noise,
                depth=depth,
                method="feasible_xy",
                contrast="noisy_shot_minus_ideal_analytic",
                metric=metric,
            )
            tokens.update(effect_tokens(f"NOISE_P{depth}_{suffix}", record))

    return tokens


def main() -> None:
    missing = [name for name, path in SOURCES.items() if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"missing manuscript source artifacts: {missing}")
    tokens = build_tokens()
    rendered_outputs: dict[Path, str] = {
        OUTPUT: TEMPLATE.read_text(encoding="utf-8"),
        ABSTRACT_OUTPUT: ABSTRACT_TEMPLATE.read_text(encoding="utf-8"),
    }
    for path, rendered in rendered_outputs.items():
        for key, value in sorted(tokens.items(), key=lambda item: -len(item[0])):
            rendered = rendered.replace(f"@@{key}@@", value)
        rendered_outputs[path] = rendered
    unresolved = sorted(
        {
            token
            for rendered in rendered_outputs.values()
            for token in re.findall(r"@@[A-Z0-9_]+@@", rendered)
        }
    )
    if unresolved:
        raise RuntimeError(f"unresolved manuscript tokens: {unresolved}")
    for path, rendered in rendered_outputs.items():
        path.write_text(rendered, encoding="utf-8")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": sha256(Path(__file__)),
        "template_sha256": sha256(TEMPLATE),
        "abstract_template_sha256": sha256(ABSTRACT_TEMPLATE),
        "source_sha256": {
            path: sha256(ROOT / path) for path in SOURCES.values()
        },
        "output_sha256": sha256(OUTPUT),
        "abstract_output_sha256": sha256(ABSTRACT_OUTPUT),
        "numeric_tokens_populated": len(tokens),
        "unresolved_tokens": unresolved,
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Rendered manuscript and abstract with {len(tokens)} "
        "artifact-derived numeric tokens"
    )


if __name__ == "__main__":
    main()
