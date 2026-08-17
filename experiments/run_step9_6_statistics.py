"""Run matched, cluster-aware Step 9.6 inference on finalized Step 9.2 data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.reproducibility import (  # noqa: E402
    build_experiment_manifest,
    file_sha256,
    write_json,
)
from src.statistical_inference import (  # noqa: E402
    holm_adjust,
    paired_sign_flip_test,
    percentile_bootstrap_interval,
    wilson_interval,
)


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "statistical_rigor.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step9_6_statistics"


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _source_directory(config: dict[str, Any], key: str) -> Path:
    path = (REPOSITORY_ROOT / config["source_directories"][key]).resolve()
    if REPOSITORY_ROOT.resolve() not in path.parents or not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _verify_step9_2(step9_2: Path) -> dict[str, str]:
    manifest_path = step9_2 / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Step 9.2 must be finalized before Step 9.6")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = dict(manifest["artifact_sha256"])
    required = [
        "qaoa_runs.csv",
        "qaoa_mixer_pairs.csv",
        "candidate_route_problems.csv",
        "inherited_dynamic_qaoa_link.csv",
        "validation_checks.json",
    ]
    verified: dict[str, str] = {}
    for name in required:
        path = step9_2 / name
        actual = file_sha256(path)
        if expected.get(name) != actual:
            raise RuntimeError(f"Step 9.2 artifact hash mismatch: {name}")
        verified[name] = actual
    checks = json.loads((step9_2 / "validation_checks.json").read_text())
    if not checks.get("all_checks_pass") or checks.get("trial_count") != 1800:
        raise RuntimeError("Step 9.2 validation checks are not publication-complete")
    return verified


def _effect_summary(
    values: np.ndarray,
    *,
    config: dict[str, Any],
    seed_offset: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("effect summary requires at least two finite clusters")
    estimate, lower, upper = percentile_bootstrap_interval(
        values,
        confidence=float(config["confidence"]),
        resamples=int(config["bootstrap_resamples"]),
        seed=int(config["bootstrap_seed"]) + seed_offset,
    )
    standard_deviation = float(np.std(values, ddof=1))
    return {
        "cluster_count": int(len(values)),
        "mean_difference": estimate,
        "median_cluster_difference": float(np.median(values)),
        "standard_deviation_cluster_difference": standard_deviation,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "standardized_mean_difference_dz": (
            estimate / standard_deviation if standard_deviation > 0.0 else np.nan
        ),
        "p_value_sign_flip": paired_sign_flip_test(
            values,
            permutations=int(config["sign_flip_permutations"]),
            seed=int(config["permutation_seed"]) + seed_offset,
        ),
    }


def _cluster_differences(
    data: pd.DataFrame,
    *,
    metric: str,
    factor: str,
    low: object,
    high: object,
) -> np.ndarray:
    means = (
        data.groupby(["network_id", factor], sort=True)[metric]
        .mean()
        .unstack(factor)
    )
    if low not in means or high not in means:
        raise RuntimeError(f"missing factor levels for {factor}: {low}, {high}")
    differences = means[high] - means[low]
    if differences.isna().any():
        raise RuntimeError(f"unmatched clusters in {factor} contrast")
    return differences.to_numpy(dtype=float)


def _descriptive_tables(
    runs: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = list(map(str, config["primary_mixer_effect_metrics"]))
    cell_rows: list[dict[str, object]] = []
    for keys, group in runs.groupby(
        ["topology", "uncertainty", "K", "depth", "method"], sort=True
    ):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            cell_rows.append(
                {
                    "topology": keys[0],
                    "uncertainty": keys[1],
                    "K": int(keys[2]),
                    "depth": int(keys[3]),
                    "method": keys[4],
                    "metric": metric,
                    "n_network_instances": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                }
            )
    cell_table = pd.DataFrame(cell_rows)

    primary_rows: list[dict[str, object]] = []
    seed_offset = 0
    for keys, group in runs.groupby(["K", "depth", "method"], sort=True):
        for metric in metrics:
            seed_offset += 1
            cluster_means = group.groupby("network_id", sort=True)[metric].mean()
            estimate, lower, upper = percentile_bootstrap_interval(
                cluster_means.to_numpy(dtype=float),
                confidence=float(config["confidence"]),
                resamples=int(config["bootstrap_resamples"]),
                seed=int(config["bootstrap_seed"]) + seed_offset,
            )
            values = group[metric].to_numpy(dtype=float)
            primary_rows.append(
                {
                    "K": int(keys[0]),
                    "depth": int(keys[1]),
                    "method": keys[2],
                    "metric": metric,
                    "row_count": len(values),
                    "independent_cluster_count": len(cluster_means),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                    "cluster_equal_weight_mean": estimate,
                    "cluster_bootstrap_ci95_lower": lower,
                    "cluster_bootstrap_ci95_upper": upper,
                }
            )
    primary_table = pd.DataFrame(primary_rows)

    rate_rows: list[dict[str, object]] = []
    for keys, group in runs.groupby(["K", "depth", "method"], sort=True):
        values = group["selected_realized_deadline_violation"].astype(int)
        proportion, wilson_lower, wilson_upper = wilson_interval(
            int(values.sum()), len(values), confidence=float(config["confidence"])
        )
        cluster_rates = group.groupby("network_id")[
            "selected_realized_deadline_violation"
        ].mean()
        _, cluster_lower, cluster_upper = percentile_bootstrap_interval(
            cluster_rates.to_numpy(dtype=float),
            confidence=float(config["confidence"]),
            resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]) + 1_000 + len(rate_rows),
        )
        rate_rows.append(
            {
                "K": int(keys[0]),
                "depth": int(keys[1]),
                "method": keys[2],
                "violations": int(values.sum()),
                "row_count": len(values),
                "independent_cluster_count": len(cluster_rates),
                "violation_rate": proportion,
                "wilson_ci95_lower_descriptive": wilson_lower,
                "wilson_ci95_upper_descriptive": wilson_upper,
                "cluster_bootstrap_ci95_lower": cluster_lower,
                "cluster_bootstrap_ci95_upper": cluster_upper,
            }
        )
    return cell_table, primary_table, pd.DataFrame(rate_rows)


def _major_mixer_effects(
    runs: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, metric in enumerate(config["primary_mixer_effect_metrics"]):
        means = (
            runs.groupby(["network_id", "method"], sort=True)[str(metric)]
            .mean()
            .unstack("method")
        )
        differences = means["feasible_xy"] - means["standard_x"]
        rows.append(
            {
                "contrast": "feasible_xy_minus_standard_x",
                "metric": str(metric),
                **_effect_summary(
                    differences.to_numpy(dtype=float),
                    config=config,
                    seed_offset=2_000 + index,
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["p_value_holm"] = holm_adjust(table["p_value_sign_flip"])
    return table


def _stratified_mixer_effects(
    runs: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strata: list[tuple[str, list[str]]] = [
        ("K_depth", ["K", "depth"]),
        ("topology", ["topology"]),
        ("uncertainty", ["uncertainty"]),
    ]
    seed_offset = 3_000
    for stratum_name, columns in strata:
        grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
        for keys, subset in runs.groupby(grouper, sort=True):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            labels = {
                column: value.item() if isinstance(value, np.generic) else value
                for column, value in zip(columns, key_tuple, strict=True)
            }
            for metric in config["primary_mixer_effect_metrics"]:
                seed_offset += 1
                means = (
                    subset.groupby(["network_id", "method"], sort=True)[str(metric)]
                    .mean()
                    .unstack("method")
                )
                difference = means["feasible_xy"] - means["standard_x"]
                rows.append(
                    {
                        "stratum_type": stratum_name,
                        "stratum_json": json.dumps(labels, sort_keys=True),
                        "metric": str(metric),
                        "contrast": "feasible_xy_minus_standard_x",
                        **_effect_summary(
                            difference.to_numpy(dtype=float),
                            config=config,
                            seed_offset=seed_offset,
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    table["p_value_holm_within_metric"] = np.nan
    for metric, indices in table.groupby("metric").groups.items():
        table.loc[indices, "p_value_holm_within_metric"] = holm_adjust(
            table.loc[indices, "p_value_sign_flip"].to_numpy(dtype=float)
        )
    return table


def _factorial_contrasts(runs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    levels = dict(config["prespecified_factor_contrasts"])
    metrics = list(map(str, config["primary_mixer_effect_metrics"]))
    rows: list[dict[str, object]] = []
    seed_offset = 5_000
    for metric in metrics:
        for method in ["standard_x", "feasible_xy"]:
            subset = runs[runs["method"] == method]
            for factor in ["depth", "K", "uncertainty"]:
                low, high = levels[factor]
                seed_offset += 1
                rows.append(
                    {
                        "metric": metric,
                        "contrast": f"{factor}_{high}_minus_{low}",
                        "method_scope": method,
                        **_effect_summary(
                            _cluster_differences(
                                subset, metric=metric, factor=factor, low=low, high=high
                            ),
                            config=config,
                            seed_offset=seed_offset,
                        ),
                    }
                )
        for factor in ["depth", "K", "uncertainty"]:
            low, high = levels[factor]
            by_network_factor_method = (
                runs.groupby(["network_id", factor, "method"], sort=True)[metric]
                .mean()
                .unstack("method")
            )
            by_network_factor_method["mixer_effect"] = (
                by_network_factor_method["feasible_xy"]
                - by_network_factor_method["standard_x"]
            )
            effect = by_network_factor_method["mixer_effect"].unstack(factor)
            differences = effect[high] - effect[low]
            seed_offset += 1
            rows.append(
                {
                    "metric": metric,
                    "contrast": f"mixer_by_{factor}_interaction_{high}_minus_{low}",
                    "method_scope": "feasible_xy_minus_standard_x",
                    **_effect_summary(
                        differences.to_numpy(dtype=float),
                        config=config,
                        seed_offset=seed_offset,
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table["p_value_holm_within_metric"] = np.nan
    for metric, indices in table.groupby("metric").groups.items():
        table.loc[indices, "p_value_holm_within_metric"] = holm_adjust(
            table.loc[indices, "p_value_sign_flip"].to_numpy(dtype=float)
        )
    return table


def _objective_ablation(
    candidates: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidates.copy()
    fastest = candidates.groupby("problem_id")["realized_travel_time_minutes"].transform(
        "min"
    )
    candidates["realized_regret_vs_fastest_minutes"] = (
        candidates["realized_travel_time_minutes"] - fastest
    )
    variants = {
        "A_expected_time": "selected_by_A",
        "B_expected_plus_deadline": "selected_by_B",
        "C_expected_plus_deadline_plus_uncertainty": "selected_by_C",
    }
    selected_tables: list[pd.DataFrame] = []
    for variant, flag in variants.items():
        selected = candidates[candidates[flag]].copy()
        if selected["problem_id"].duplicated().any():
            raise RuntimeError(f"objective {variant} selected multiple routes")
        selected["objective_variant"] = variant
        selected_tables.append(selected)
    selected = pd.concat(selected_tables, ignore_index=True)
    summary = (
        selected.groupby("objective_variant", sort=True)
        .agg(
            problem_count=("problem_id", "size"),
            network_count=("network_id", "nunique"),
            mean_realized_time_minutes=("realized_travel_time_minutes", "mean"),
            median_realized_time_minutes=("realized_travel_time_minutes", "median"),
            deadline_violation_rate=("realized_deadline_violation", "mean"),
            mean_realized_regret_vs_fastest_minutes=(
                "realized_regret_vs_fastest_minutes",
                "mean",
            ),
        )
        .reset_index()
    )
    comparisons = [
        ("A_expected_time", "B_expected_plus_deadline"),
        ("B_expected_plus_deadline", "C_expected_plus_deadline_plus_uncertainty"),
    ]
    rows: list[dict[str, object]] = []
    seed_offset = 7_000
    metrics = [
        "realized_travel_time_minutes",
        "realized_deadline_violation",
        "realized_regret_vs_fastest_minutes",
    ]
    index_columns = ["problem_id", "network_id", "objective_variant"]
    compact = selected[index_columns + ["route_id"] + metrics]
    for low, high in comparisons:
        low_table = compact[compact["objective_variant"] == low].set_index("problem_id")
        high_table = compact[compact["objective_variant"] == high].set_index("problem_id")
        aligned = low_table.join(high_table, lsuffix="_low", rsuffix="_high")
        if len(aligned) != 300:
            raise RuntimeError("objective ablation problems are not matched")
        selection_change = aligned["route_id_low"] != aligned["route_id_high"]
        rows.append(
            {
                "comparison": f"{high}_minus_{low}",
                "metric": "route_selection_change_rate",
                "cluster_count": aligned["network_id_low"].nunique(),
                "mean_difference": float(selection_change.mean()),
                "median_cluster_difference": np.nan,
                "standard_deviation_cluster_difference": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "standardized_mean_difference_dz": np.nan,
                "p_value_sign_flip": np.nan,
            }
        )
        for metric in metrics:
            seed_offset += 1
            aligned["difference"] = aligned[f"{metric}_high"].astype(float) - aligned[
                f"{metric}_low"
            ].astype(float)
            cluster_difference = aligned.groupby("network_id_low")["difference"].mean()
            rows.append(
                {
                    "comparison": f"{high}_minus_{low}",
                    "metric": metric,
                    **_effect_summary(
                        cluster_difference.to_numpy(dtype=float),
                        config=config,
                        seed_offset=seed_offset,
                    ),
                }
            )
    effects = pd.DataFrame(rows)
    mask = effects["p_value_sign_flip"].notna()
    effects.loc[mask, "p_value_holm"] = holm_adjust(
        effects.loc[mask, "p_value_sign_flip"].to_numpy(dtype=float)
    )
    return summary, effects


def _entropy_associations(runs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    xy = runs[runs["method"] == "feasible_xy"].copy()
    rows: list[dict[str, object]] = []
    for index, outcome in enumerate(config["entropy_association_outcomes"]):
        per_network: list[float] = []
        for _, group in xy.groupby("network_id", sort=True):
            predictor_values = group["normalized_feasible_entropy"].to_numpy(dtype=float)
            outcome_values = group[str(outcome)].to_numpy(dtype=float)
            correlation = (
                spearmanr(predictor_values, outcome_values).statistic
                if np.ptp(predictor_values) > 0.0 and np.ptp(outcome_values) > 0.0
                else np.nan
            )
            if np.isfinite(correlation):
                per_network.append(float(correlation))
        values = np.asarray(per_network, dtype=float)
        overall_predictor = xy["normalized_feasible_entropy"].to_numpy(dtype=float)
        overall_outcome = xy[str(outcome)].to_numpy(dtype=float)
        overall = (
            float(spearmanr(overall_predictor, overall_outcome).statistic)
            if np.ptp(overall_predictor) > 0.0 and np.ptp(overall_outcome) > 0.0
            else np.nan
        )
        row = {
            "predictor": "normalized_feasible_entropy",
            "outcome": str(outcome),
            "method_scope": "feasible_xy",
            "overall_row_level_spearman_descriptive": overall,
            "network_correlations_defined": len(values),
            **_effect_summary(values, config=config, seed_offset=8_000 + index),
        }
        row["mean_within_network_spearman"] = row.pop("mean_difference")
        row["median_within_network_spearman"] = row.pop(
            "median_cluster_difference"
        )
        row["standard_deviation_within_network_spearman"] = row.pop(
            "standard_deviation_cluster_difference"
        )
        row["p_value_sign_flip_vs_zero"] = row.pop("p_value_sign_flip")
        rows.append(row)
    table = pd.DataFrame(rows)
    table["p_value_holm"] = holm_adjust(table["p_value_sign_flip_vs_zero"])
    return table


def _dynamic_effects(
    dynamic_directory: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(dynamic_directory / "dynamic_runs.csv")
    decisions = pd.read_csv(dynamic_directory / "dynamic_qaoa_decisions.csv")
    policies = ["expected_time_rerouting", "uncertainty_aware_qaoa"]
    subset = runs[runs["policy"].isin(policies)].copy()
    key = ["network_id", "topology", "scenario"]
    if subset.duplicated(key + ["policy"]).any():
        raise RuntimeError("dynamic policy rows are not unique per matched scenario")
    metrics = [
        "deadline_violation",
        "arrival_time_minutes",
        "deadline_margin_minutes",
        "reroute_count",
        "route_stability",
    ]
    rows: list[dict[str, object]] = []
    seed_offset = 9_000
    for metric in metrics:
        wide = subset.pivot(index=key, columns="policy", values=metric).dropna()
        difference = (
            wide["uncertainty_aware_qaoa"].astype(float)
            - wide["expected_time_rerouting"].astype(float)
        ).rename("difference")
        network_difference = difference.groupby(level="network_id").mean()
        seed_offset += 1
        rows.append(
            {
                "contrast": "uncertainty_aware_qaoa_minus_expected_time_rerouting",
                "metric": metric,
                "matched_scenario_count": len(wide),
                "scope_status": "exploratory_five_network_controlled_simulation",
                **_effect_summary(
                    network_difference.to_numpy(dtype=float),
                    config=config,
                    seed_offset=seed_offset,
                ),
            }
        )
    effects = pd.DataFrame(rows)
    effects["p_value_holm"] = holm_adjust(effects["p_value_sign_flip"])

    qaoa_runs = subset[subset["policy"] == "uncertainty_aware_qaoa"].copy()
    qaoa_summary = (
        decisions.groupby("run_id", sort=True)
        .agg(
            mean_dynamic_P_opt=("analytic_optimal_probability", "mean"),
            mean_dynamic_P_feasible=("analytic_valid_probability", "mean"),
        )
        .reset_index()
    )
    linked = qaoa_runs.merge(qaoa_summary, on="run_id", validate="one_to_one")
    association_rows: list[dict[str, object]] = []
    for predictor in ["mean_dynamic_P_opt", "mean_dynamic_P_feasible"]:
        for outcome in ["deadline_violation", "deadline_margin_minutes"]:
            predictor_values = linked[predictor].to_numpy(dtype=float)
            outcome_values = linked[outcome].to_numpy(dtype=float)
            statistic = (
                float(spearmanr(predictor_values, outcome_values).statistic)
                if np.ptp(predictor_values) > 0.0 and np.ptp(outcome_values) > 0.0
                else np.nan
            )
            association_rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "run_count": len(linked),
                    "network_count": linked["network_id"].nunique(),
                    "spearman_correlation": statistic,
                    "scope_status": "exploratory_unadjusted_five_network_association",
                }
            )
    return effects, pd.DataFrame(association_rows)


def _validation_report(
    runs: pd.DataFrame,
    major: pd.DataFrame,
    associations: pd.DataFrame,
    dynamic: pd.DataFrame,
) -> str:
    feasibility = major[major["metric"] == "feasibility_probability"].iloc[0]
    optimality = major[major["metric"] == "optimal_state_probability"].iloc[0]
    entropy_opt = associations[
        associations["outcome"] == "optimal_state_probability"
    ].iloc[0]
    dynamic_deadline = dynamic[dynamic["metric"] == "deadline_violation"].iloc[0]
    return f"""# Validation Report — Step 9.6 Statistical Rigor

## Status

- Status: **PASS** for matched, cluster-aware execution.
- Step 9.2 rows analyzed: {len(runs)}; independent network-instance clusters: {runs['network_id'].nunique()}.
- Bootstrap resampling unit: complete network instance.
- Multiple-comparison control: Holm family-wise adjustment within declared families.
- No naive row-independent factorial ANOVA was used.

## Primary matched mixer effects

- XY − X feasibility probability: {feasibility.mean_difference:.6f}, 95% cluster-bootstrap CI [{feasibility.ci95_lower:.6f}, {feasibility.ci95_upper:.6f}].
- XY − X optimal-state probability: {optimality.mean_difference:.6f}, 95% CI [{optimality.ci95_lower:.6f}, {optimality.ci95_upper:.6f}].
- Mean within-network entropy/P_opt Spearman association (XY): {entropy_opt.mean_within_network_spearman:.6f}, 95% CI [{entropy_opt.ci95_lower:.6f}, {entropy_opt.ci95_upper:.6f}].
- Dynamic QAOA − expected-time deadline-violation rate: {dynamic_deadline.mean_difference:.6f}, 95% five-network cluster-bootstrap CI [{dynamic_deadline.ci95_lower:.6f}, {dynamic_deadline.ci95_upper:.6f}].

Consult the CSV tables for all effect sizes, sample sizes, unadjusted and Holm-adjusted p-values. P-values are secondary to effects and intervals.

## Assumptions and limitations

1. Network instances are treated as independent; repeated uncertainty, K, depth, and mixer observations within an instance are not.
2. The five-topology design has 50 independent instances overall but only 10 per topology. High-order mixed models would be unstable, and XY feasibility is on a boundary; prespecified marginal within-network contrasts are used instead.
3. Percentile cluster-bootstrap intervals quantify finite-design uncertainty and do not turn the controlled synthetic networks into a population-representative road sample.
4. Dynamic inference has only five network clusters and remains exploratory even when an interval excludes zero.
5. Spearman associations are diagnostic, not causal mediation estimates.
"""


def main() -> None:
    config = _load_config()
    step9_2 = _source_directory(config, "step9_2")
    dynamic = _source_directory(config, "dynamic")
    verified_sources = _verify_step9_2(step9_2)
    runs = pd.read_csv(step9_2 / "qaoa_runs.csv")
    candidates = pd.read_csv(step9_2 / "candidate_route_problems.csv")
    if len(runs) != 1800 or runs["run_id"].duplicated().any():
        raise RuntimeError("Step 9.2 run table is not the expected 1,800 unique trials")
    if runs["network_id"].nunique() != int(config["overall_cluster_count"]):
        raise RuntimeError("unexpected number of independent network clusters")

    cell, primary, rates = _descriptive_tables(runs, config)
    major = _major_mixer_effects(runs, config)
    stratified = _stratified_mixer_effects(runs, config)
    factorial = _factorial_contrasts(runs, config)
    ablation_summary, ablation_effects = _objective_ablation(candidates, config)
    associations = _entropy_associations(runs, config)
    dynamic_effects, dynamic_associations = _dynamic_effects(dynamic, config)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "descriptive_by_factorial_cell.csv": cell,
        "descriptive_primary.csv": primary,
        "deadline_rate_intervals.csv": rates,
        "major_mixer_effects.csv": major,
        "stratified_mixer_effects.csv": stratified,
        "factorial_contrasts.csv": factorial,
        "objective_ablation_summary.csv": ablation_summary,
        "objective_ablation_effects.csv": ablation_effects,
        "entropy_associations.csv": associations,
        "dynamic_policy_effects.csv": dynamic_effects,
        "dynamic_probability_associations.csv": dynamic_associations,
    }
    for name, table in artifacts.items():
        table.to_csv(OUTPUT_DIRECTORY / name, index=False)
    report = _validation_report(runs, major, associations, dynamic_effects)
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "bootstrap_seed": int(config["bootstrap_seed"]),
            "permutation_seed": int(config["permutation_seed"]),
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "source_sha256": {
                **verified_sources,
                "dynamic_runs.csv": file_sha256(dynamic / "dynamic_runs.csv"),
                "dynamic_qaoa_decisions.csv": file_sha256(
                    dynamic / "dynamic_qaoa_decisions.csv"
                ),
                "runner": file_sha256(Path(__file__)),
                "statistical_inference": file_sha256(
                    REPOSITORY_ROOT / "src" / "statistical_inference.py"
                ),
            },
            "artifact_sha256": {
                name: file_sha256(OUTPUT_DIRECTORY / name) for name in artifacts
            }
            | {"VALIDATION_REPORT.md": file_sha256(OUTPUT_DIRECTORY / "VALIDATION_REPORT.md")},
            "claim_exclusions": [
                "causal_association",
                "population_representativeness",
                "independent_1800_samples",
                "quantum_advantage",
            ],
        }
    )
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    print(
        f"Step 9.6 complete: {len(runs)} trials, "
        f"{runs['network_id'].nunique()} independent network clusters.",
        flush=True,
    )


if __name__ == "__main__":
    main()
