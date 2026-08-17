"""Exploratory graph- and route-structure analysis for finalized Step 9.2 data."""

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
    spearman_bootstrap_interval,
)


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "topology_mechanism_analysis.json"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "research" / "step9_6b_topology_mechanisms"
)


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _verify_sources(source: Path) -> dict[str, str]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    required = [
        "qaoa_runs.csv",
        "problem_summary.csv",
        "network_characteristics.csv",
        "validation_checks.json",
    ]
    verified: dict[str, str] = {}
    for name in required:
        actual = file_sha256(source / name)
        if manifest["artifact_sha256"].get(name) != actual:
            raise RuntimeError(f"Step 9.2 artifact hash mismatch: {name}")
        verified[name] = actual
    checks = json.loads((source / "validation_checks.json").read_text())
    if not checks.get("all_checks_pass") or checks.get("trial_count") != 1800:
        raise RuntimeError("Step 9.2 is not complete and validated")
    return verified


def _summary_interval(
    values: np.ndarray, config: dict[str, Any], seed: int
) -> tuple[float, float, float]:
    return percentile_bootstrap_interval(
        values,
        confidence=float(config["confidence"]),
        resamples=int(config["bootstrap_resamples"]),
        seed=int(config["bootstrap_seed"]) + seed,
    )


def _topology_summary(runs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    metrics = list(config["outcomes"])
    rows: list[dict[str, Any]] = []
    seed = 0
    network_means = (
        runs.groupby(["network_id", "topology", "K", "method"], sort=True)[metrics]
        .mean()
        .reset_index()
    )
    for keys, group in network_means.groupby(
        ["topology", "K", "method"], sort=True
    ):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            estimate, lower, upper = _summary_interval(values, config, seed)
            rows.append(
                {
                    "topology": keys[0],
                    "K": int(keys[1]),
                    "method": keys[2],
                    "metric": metric,
                    "independent_network_count": int(len(values)),
                    "network_equal_weight_mean": estimate,
                    "median_network_mean": float(np.median(values)),
                    "standard_deviation_network_mean": float(np.std(values, ddof=1)),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
            seed += 1
    return pd.DataFrame(rows)


def _graph_associations(
    runs: pd.DataFrame,
    networks: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    outcomes = list(config["outcomes"])
    predictors = list(config["graph_predictors"])
    network_means = (
        runs.groupby(["network_id", "topology", "K", "method"], sort=True)[outcomes]
        .mean()
        .reset_index()
    )
    columns = ["network_id", "topology_family", *predictors]
    merged = network_means.merge(
        networks[columns], on="network_id", validate="many_to_one"
    )
    if not (merged["topology"] == merged["topology_family"]).all():
        raise RuntimeError("topology labels disagree across source tables")

    rows: list[dict[str, Any]] = []
    seed = 1000
    for (k_value, method), group in merged.groupby(["K", "method"], sort=True):
        for predictor in predictors:
            for outcome in outcomes:
                x = group[predictor].to_numpy(float)
                y = group[outcome].to_numpy(float)
                estimate, lower, upper, defined = spearman_bootstrap_interval(
                    x,
                    y,
                    confidence=float(config["confidence"]),
                    resamples=int(config["bootstrap_resamples"]),
                    seed=int(config["bootstrap_seed"]) + seed,
                )
                rows.append(
                    {
                        "K": int(k_value),
                        "method": method,
                        "predictor": predictor,
                        "outcome": outcome,
                        "independent_network_count": int(len(group)),
                        "spearman_rho": estimate,
                        "bootstrap_ci95_lower": lower,
                        "bootstrap_ci95_upper": upper,
                        "defined_bootstrap_draws": defined,
                        "scope_status": "exploratory_descriptive_topology_confounded",
                    }
                )
                seed += 1
    return pd.DataFrame(rows)


def _problem_associations(
    runs: pd.DataFrame,
    problems: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    outcomes = list(config["outcomes"])
    predictors = list(config["problem_predictors"])
    problem_means = (
        runs.groupby(["problem_id", "network_id", "method"], sort=True)[outcomes]
        .mean()
        .reset_index()
    )
    problem_columns = ["problem_id", "network_id", *predictors]
    merged = problem_means.merge(
        problems[problem_columns],
        on=["problem_id", "network_id"],
        validate="many_to_one",
    )

    rows: list[dict[str, Any]] = []
    seed = 5000
    for method in sorted(merged["method"].unique()):
        method_rows = merged[merged["method"] == method]
        for predictor in predictors:
            for outcome in outcomes:
                correlations: list[float] = []
                for _, group in method_rows.groupby("network_id", sort=True):
                    x = group[predictor].to_numpy(float)
                    y = group[outcome].to_numpy(float)
                    if np.unique(x).size < 2 or np.unique(y).size < 2:
                        continue
                    value = float(spearmanr(x, y).statistic)
                    if np.isfinite(value):
                        correlations.append(value)
                values = np.asarray(correlations, dtype=float)
                if len(values) < 2:
                    continue
                estimate, lower, upper = _summary_interval(values, config, seed)
                rows.append(
                    {
                        "method": method,
                        "predictor": predictor,
                        "outcome": outcome,
                        "networks_with_defined_correlation": int(len(values)),
                        "mean_within_network_spearman": estimate,
                        "median_within_network_spearman": float(np.median(values)),
                        "standard_deviation_within_network_spearman": float(
                            np.std(values, ddof=1)
                        ),
                        "bootstrap_ci95_lower": lower,
                        "bootstrap_ci95_upper": upper,
                        "p_value_sign_flip_vs_zero": paired_sign_flip_test(
                            values,
                            permutations=int(config["sign_flip_permutations"]),
                            seed=int(config["permutation_seed"]) + seed,
                        ),
                        "scope_status": "exploratory_six_problems_per_network",
                    }
                )
                seed += 1
    frame = pd.DataFrame(rows)
    frame["p_value_holm_exploratory_family"] = holm_adjust(
        frame["p_value_sign_flip_vs_zero"].to_numpy(float)
    )
    return frame


def _report(
    topology: pd.DataFrame,
    graph: pd.DataFrame,
    problem: pd.DataFrame,
) -> str:
    xy = topology[
        (topology["method"] == "feasible_xy")
        & (topology["metric"] == "optimal_state_probability")
        & (topology["K"] == 8)
    ].sort_values("network_equal_weight_mean")
    weakest = xy.iloc[0]
    strongest = xy.iloc[-1]
    strongest_problem = problem.iloc[
        problem["mean_within_network_spearman"].abs().argmax()
    ]
    return f"""# Validation Report — Step 9.6b Topology Mechanisms

## Status

- Status: **PASS** for the declared exploratory analysis.
- Independent network instances: 50 (10 per topology family).
- Topology-summary rows: {len(topology)}.
- Graph-level association rows: {len(graph)}.
- Within-network problem association rows: {len(problem)}.
- Design status: post-hoc exploratory; none of these associations is confirmatory or causal.

## Descriptive findings

- At K=8, the lowest topology-level feasible-XY mean P_opt was {weakest['topology']} ({weakest['network_equal_weight_mean']:.6f}); the highest was {strongest['topology']} ({strongest['network_equal_weight_mean']:.6f}).
- The largest absolute mean within-network route/problem association was {strongest_problem['predictor']} versus {strongest_problem['outcome']} for {strongest_problem['method']} (mean Spearman rho {strongest_problem['mean_within_network_spearman']:.6f}, 95% network-bootstrap CI [{strongest_problem['bootstrap_ci95_lower']:.6f}, {strongest_problem['bootstrap_ci95_upper']:.6f}]).

## Interpretation guardrails

1. Topology class and graph statistics are intentionally coupled, so graph-level correlations cannot identify a causal graph mechanism.
2. Each within-network correlation has only six K-by-uncertainty problems and is used as a diagnostic; the network, not the problem row, is the inferential unit.
3. Holm-adjusted exploratory p-values are secondary to effect size, interval width, and mechanistic plausibility.
4. These controlled synthetic networks are not a population sample of road networks.
"""


def main() -> None:
    config = _load_config()
    source = (REPOSITORY_ROOT / config["source_directory"]).resolve()
    verified = _verify_sources(source)
    runs = pd.read_csv(source / "qaoa_runs.csv")
    problems = pd.read_csv(source / "problem_summary.csv")
    networks = pd.read_csv(source / "network_characteristics.csv")
    if len(runs) != 1800 or runs["network_id"].nunique() != 50:
        raise RuntimeError("unexpected Step 9.2 shape")
    if problems["problem_id"].duplicated().any() or len(problems) != 300:
        raise RuntimeError("problem summary must contain 300 unique problems")
    if networks["network_id"].duplicated().any() or len(networks) != 50:
        raise RuntimeError("network characteristics must contain 50 unique networks")

    topology = _topology_summary(runs, config)
    graph = _graph_associations(runs, networks, config)
    problem = _problem_associations(runs, problems, config)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, pd.DataFrame] = {
        "topology_method_summary.csv": topology,
        "graph_structure_associations.csv": graph,
        "route_problem_associations.csv": problem,
    }
    for name, frame in artifacts.items():
        frame.to_csv(OUTPUT_DIRECTORY / name, index=False)
    report = _report(topology, graph, problem)
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    manifest = build_experiment_manifest(
        config["experiment_name"],
        REPOSITORY_ROOT,
        {
            "bootstrap_seed": int(config["bootstrap_seed"]),
            "permutation_seed": int(config["permutation_seed"]),
        },
        config,
        CONFIG_PATH,
    )
    manifest["source_sha256"] = verified | {
        "runner": file_sha256(Path(__file__)),
        "statistical_inference": file_sha256(
            REPOSITORY_ROOT / "src" / "statistical_inference.py"
        ),
    }
    manifest["artifact_sha256"] = {
        name: file_sha256(OUTPUT_DIRECTORY / name)
        for name in [*artifacts, "VALIDATION_REPORT.md"]
    }
    manifest["claim_exclusions"] = [
        "causal_graph_mechanism",
        "confirmatory_topology_hypothesis",
        "population_representativeness",
        "quantum_advantage",
    ]
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    print(report)


if __name__ == "__main__":
    main()
