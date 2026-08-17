"""Resumable validation/test standard-X penalty sensitivity analysis."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from pennylane import numpy as pnp


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.feasibility_qaoa import (  # noqa: E402
    circuit_resource_record,
    create_expectation_qnode,
    create_probability_qnode,
    optimize_qaoa_angles,
)
from src.penalty_sensitivity import (  # noqa: E402
    build_unrestricted_penalty_qubo,
    exact_zero_state_penalty_threshold,
    summarize_penalty_distribution,
)
from src.reproducibility import (  # noqa: E402
    build_experiment_manifest,
    file_sha256,
    write_json,
)
from src.risk_qubo import risk_qubo_to_ising  # noqa: E402
from src.statistical_inference import (  # noqa: E402
    holm_adjust,
    paired_sign_flip_test,
    percentile_bootstrap_interval,
)


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "standard_x_penalty_sensitivity.json"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "research" / "standard_x_penalty_sensitivity"
)
CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"
FAILURE_DIRECTORY = OUTPUT_DIRECTORY / "failures"


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars and undefined diagnostics to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
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
    required = ["problem_summary.csv", "qaoa_runs.csv", "validation_checks.json"]
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


def _penalty_value(condition: dict[str, Any], risks: np.ndarray) -> float:
    ratio = condition.get("ratio_to_exact_threshold")
    if ratio is None:
        return float(np.max(risks) + 1.0)
    return float(ratio) * exact_zero_state_penalty_threshold(risks)


def _specifications(
    config: dict[str, Any], source: Path
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    problems = pd.read_csv(source / "problem_summary.csv")
    source_runs = pd.read_csv(source / "qaoa_runs.csv")
    all_seeds = [
        *map(int, config["validation_seed_ids"]),
        *map(int, config["test_seed_ids"]),
    ]
    problems = problems[
        (problems["K"] == int(config["K"]))
        & (problems["uncertainty"] == str(config["uncertainty"]))
        & (problems["topology"].isin(config["topologies"]))
        & (problems["matched_seed_id"].isin(all_seeds))
    ].sort_values(["topology_index", "matched_seed_id"], kind="stable")
    if len(problems) != 50 or problems["network_id"].nunique() != 50:
        raise RuntimeError("penalty study requires 50 unique primary problems")
    source_subset = source_runs[
        (source_runs["K"] == int(config["K"]))
        & (source_runs["depth"] == int(config["depth"]))
        & (source_runs["method"] == str(config["method"]))
        & (source_runs["uncertainty"] == str(config["uncertainty"]))
        & (source_runs["topology"].isin(config["topologies"]))
        & (source_runs["matched_seed_id"].isin(all_seeds))
    ].set_index("problem_id")
    if len(source_subset) != 50:
        raise RuntimeError("accepted standard-X reference grid is incomplete")

    validation_ids = set(map(int, config["validation_seed_ids"]))
    specs: list[dict[str, Any]] = []
    design_rows: list[dict[str, object]] = []
    for problem in problems.itertuples(index=False):
        record = problem._asdict()
        risks = np.asarray(json.loads(record["risk_vector_json"]), dtype=float)
        reference = source_subset.loc[record["problem_id"]]
        split = (
            "validation"
            if int(record["matched_seed_id"]) in validation_ids
            else "test"
        )
        for condition_raw in config["penalty_conditions"]:
            condition = dict(condition_raw)
            penalty = _penalty_value(condition, risks)
            ratio = penalty / exact_zero_state_penalty_threshold(risks)
            run_id = f"{record['problem_id']}_{condition['name']}"
            spec = {
                "run_id": run_id,
                "problem_id": record["problem_id"],
                "problem_sha256": record["problem_sha256"],
                "network_id": record["network_id"],
                "topology": record["topology"],
                "topology_index": int(record["topology_index"]),
                "matched_seed_id": int(record["matched_seed_id"]),
                "data_split": split,
                "uncertainty": record["uncertainty"],
                "K": int(config["K"]),
                "depth": int(config["depth"]),
                "method": str(config["method"]),
                "penalty_condition": str(condition["name"]),
                "penalty_formula": str(condition["formula"]),
                "penalty_value": penalty,
                "penalty_ratio_to_exact_threshold": ratio,
                "exact_penalty_threshold": exact_zero_state_penalty_threshold(risks),
                "optimization_seed": int(reference["optimization_seed"]),
                "iterations": int(config["iterations"]),
                "step_size": float(config["step_size"]),
                "risk_vector": risks.tolist(),
                "actual_route_times": json.loads(record["actual_route_times_json"]),
                "deadline_minutes": float(record["deadline_minutes"]),
                "expected_time_route_index": int(record["expected_time_route_index"]),
                "exact_risk_route_index": int(record["exact_risk_route_index"]),
                "actual_fastest_route_index": int(
                    record["actual_fastest_route_index"]
                ),
            }
            specs.append(spec)
            design_rows.append(
                {
                    key: spec[key]
                    for key in [
                        "run_id",
                        "problem_id",
                        "problem_sha256",
                        "network_id",
                        "topology",
                        "matched_seed_id",
                        "data_split",
                        "K",
                        "depth",
                        "method",
                        "penalty_condition",
                        "penalty_formula",
                        "penalty_value",
                        "penalty_ratio_to_exact_threshold",
                        "exact_penalty_threshold",
                        "optimization_seed",
                    ]
                }
            )
    design = pd.DataFrame(design_rows).sort_values("run_id", kind="stable")
    if len(specs) != 350 or design["run_id"].duplicated().any():
        raise RuntimeError("penalty design must contain 350 unique trials")
    return specs, design


def _run_worker(spec: dict[str, Any]) -> dict[str, Any]:
    risks = np.asarray(spec["risk_vector"], dtype=float)
    actual_times = np.asarray(spec["actual_route_times"], dtype=float)
    qubo = build_unrestricted_penalty_qubo(risks, float(spec["penalty_value"]))
    hamiltonian, offset, _, _ = risk_qubo_to_ising(qubo)
    expectation = create_expectation_qnode(
        hamiltonian,
        number_of_qubits=int(spec["K"]),
        depth=int(spec["depth"]),
        method=str(spec["method"]),
    )
    probability = create_probability_qnode(
        hamiltonian,
        number_of_qubits=int(spec["K"]),
        depth=int(spec["depth"]),
        method=str(spec["method"]),
    )
    started = time.perf_counter()
    optimized = optimize_qaoa_angles(
        expectation,
        depth=int(spec["depth"]),
        seed=int(spec["optimization_seed"]),
        iterations=int(spec["iterations"]),
        step_size=float(spec["step_size"]),
    )
    probabilities = np.asarray(probability(optimized.gammas, optimized.betas), float)
    diagnostics = summarize_penalty_distribution(probabilities, qubo)
    resources = circuit_resource_record(probability, optimized.gammas, optimized.betas)
    elapsed = time.perf_counter() - started
    history = optimized.history.copy()
    history["expected_qubo_energy"] = history["hamiltonian_expectation"] + offset
    selected = int(diagnostics.metrics["selected_route_index"])
    selected_actual = float(actual_times[selected])
    deadline = float(spec["deadline_minutes"])
    metric = {
        key: spec[key]
        for key in [
            "run_id",
            "problem_id",
            "problem_sha256",
            "network_id",
            "topology",
            "topology_index",
            "matched_seed_id",
            "data_split",
            "uncertainty",
            "K",
            "depth",
            "method",
            "penalty_condition",
            "penalty_formula",
            "penalty_value",
            "penalty_ratio_to_exact_threshold",
            "exact_penalty_threshold",
            "optimization_seed",
            "iterations",
            "step_size",
        ]
    }
    metric.update(diagnostics.metrics)
    gate_sizes = json.loads(str(resources["gate_sizes_json"]))
    metric.update(
        {
            "hamiltonian_identity_offset": float(offset),
            "initial_expected_qubo_energy": float(
                history.iloc[0]["expected_qubo_energy"]
            ),
            "final_recorded_expected_qubo_energy": float(
                history.iloc[-1]["expected_qubo_energy"]
            ),
            "best_recorded_expected_qubo_energy": float(
                history["expected_qubo_energy"].min()
            ),
            "best_recorded_iteration": int(history["expected_qubo_energy"].idxmin()),
            "last_five_energy_range": float(
                history["expected_qubo_energy"].tail(5).max()
                - history["expected_qubo_energy"].tail(5).min()
            ),
            "gammas_json": json.dumps([float(value) for value in optimized.gammas]),
            "betas_json": json.dumps([float(value) for value in optimized.betas]),
            "wall_time_seconds": elapsed,
            "two_qubit_gate_count_logical": int(gate_sizes.get("2", 0)),
            "selected_realized_travel_time_minutes": selected_actual,
            "selected_realized_deadline_violation": selected_actual > deadline,
            "selected_realized_deadline_margin_minutes": deadline - selected_actual,
            "selected_realized_regret_vs_actual_fastest_minutes": selected_actual
            - float(actual_times[int(spec["actual_fastest_route_index"])]),
            **resources,
        }
    )
    distribution = diagnostics.feasible_distribution.copy()
    distribution.insert(0, "run_id", spec["run_id"])
    distribution.insert(1, "penalty_condition", spec["penalty_condition"])
    history.insert(0, "run_id", spec["run_id"])
    history.insert(1, "penalty_condition", spec["penalty_condition"])
    return {
        "completed_at_utc": _utc_now(),
        "metric": metric,
        "feasible_distribution": distribution.to_dict(orient="records"),
        "training_history": history.to_dict(orient="records"),
    }


def prepare(config: dict[str, Any], source: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    specs, design = _specifications(config, source)
    design_path = OUTPUT_DIRECTORY / "trial_design.csv"
    if design_path.is_file():
        existing = pd.read_csv(design_path)
        frozen = design.reset_index(drop=True)
        if list(existing.columns) != list(frozen.columns) or len(existing) != len(frozen):
            raise RuntimeError("existing penalty trial design differs from frozen design")
        for column in frozen.columns:
            if pd.api.types.is_numeric_dtype(frozen[column]):
                if not np.allclose(
                    existing[column].to_numpy(dtype=float),
                    frozen[column].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                ):
                    raise RuntimeError(
                        f"existing penalty trial design differs in numeric column {column}"
                    )
            elif not existing[column].astype(str).equals(frozen[column].astype(str)):
                raise RuntimeError(
                    f"existing penalty trial design differs in column {column}"
                )
    else:
        if any(CHECKPOINT_DIRECTORY.glob("*.json")):
            raise RuntimeError("cannot write a new design after checkpoints exist")
        _atomic_csv(design_path, design.reset_index(drop=True))
    return specs, design


def benchmark(specs: list[dict[str, Any]], config: dict[str, Any]) -> None:
    path = OUTPUT_DIRECTORY / "runtime_benchmark.csv"
    if path.is_file():
        return
    representative = next(
        spec
        for spec in specs
        if spec["topology"] == "grid"
        and spec["matched_seed_id"] == 0
        and spec["penalty_condition"] == "validated_safe_default"
    )
    trial = dict(representative)
    trial["iterations"] = 3
    result = _run_worker(trial)
    seconds = float(result["metric"]["wall_time_seconds"])
    estimated_sequential = seconds / 3.0 * int(config["iterations"]) * len(specs) * 1.25
    table = pd.DataFrame(
        [
            {
                "problem_id": representative["problem_id"],
                "penalty_condition": representative["penalty_condition"],
                "benchmark_iterations": 3,
                "wall_time_seconds": seconds,
                "estimated_sequential_seconds_with_25pct_margin": estimated_sequential,
                "estimated_parallel_seconds_at_default_workers": estimated_sequential
                / int(config["default_workers"]),
            }
        ]
    )
    _atomic_csv(path, table)
    print(table.to_string(index=False), flush=True)


def run_trials(specs: list[dict[str, Any]], *, workers: int) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FAILURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    missing = [
        spec
        for spec in specs
        if not (CHECKPOINT_DIRECTORY / f"{spec['run_id']}.json").is_file()
    ]
    completed = len(specs) - len(missing)
    print(
        f"Penalty checkpoints: {completed}/{len(specs)} complete; "
        f"launching {len(missing)} with {workers} workers.",
        flush=True,
    )
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_worker, spec): spec for spec in missing}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                _atomic_json(
                    CHECKPOINT_DIRECTORY / f"{spec['run_id']}.json",
                    future.result(),
                )
            except Exception as error:  # pragma: no cover
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
            if completed % 10 == 0 or completed == len(specs):
                print(
                    f"Penalty progress {completed}/{len(specs)}; "
                    f"failures this launch={failures}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"{failures} penalty trials failed; records preserved")


def _effect_summary(
    values: np.ndarray, config: dict[str, Any], seed_offset: int
) -> dict[str, object]:
    estimate, lower, upper = percentile_bootstrap_interval(
        values,
        resamples=int(config["bootstrap_resamples"]),
        seed=int(config["bootstrap_seed"]) + seed_offset,
    )
    return {
        "independent_network_count": int(len(values)),
        "mean_difference": estimate,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "p_value_sign_flip": paired_sign_flip_test(
            values,
            permutations=100_000,
            seed=int(config["bootstrap_seed"]) + 10_000 + seed_offset,
        ),
    }


def _analysis(
    runs: pd.DataFrame, config: dict[str, Any], source: Path
) -> dict[str, pd.DataFrame | dict[str, object]]:
    validation = runs[runs["data_split"] == "validation"]
    test = runs[runs["data_split"] == "test"]
    if len(validation) != 105 or len(test) != 245:
        raise RuntimeError("validation/test penalty split is incomplete")

    validation_selection = (
        validation.groupby("penalty_condition", sort=False)
        .agg(
            validation_problem_count=("problem_id", "size"),
            global_ground_states_all_feasible_every_problem=(
                "global_ground_states_all_feasible",
                "all",
            ),
            validation_mean_optimal_state_probability=(
                "optimal_state_probability",
                "mean",
            ),
            validation_mean_feasibility_probability=(
                "feasibility_probability",
                "mean",
            ),
            median_ratio_to_exact_threshold=(
                "penalty_ratio_to_exact_threshold",
                "median",
            ),
        )
        .reset_index()
    )
    eligible = validation_selection[
        validation_selection["global_ground_states_all_feasible_every_problem"]
    ].sort_values(
        [
            "validation_mean_optimal_state_probability",
            "median_ratio_to_exact_threshold",
        ],
        ascending=[False, True],
        kind="stable",
    )
    if eligible.empty:
        raise RuntimeError("no penalty condition passed validation eligibility")
    selected = str(eligible.iloc[0]["penalty_condition"])
    validation_selection["selected_by_frozen_validation_rule"] = (
        validation_selection["penalty_condition"] == selected
    )

    summary_metrics = [
        "feasibility_probability",
        "optimal_state_probability",
        "optimal_probability_given_feasible",
        "normalized_feasible_entropy",
        "energy_regret",
        "risk_regret",
        "selected_route_is_optimal",
        "last_five_energy_range",
        "wall_time_seconds",
    ]
    summary_rows: list[dict[str, object]] = []
    for condition, group in test.groupby("penalty_condition", sort=False):
        for metric in summary_metrics:
            values = group[metric].astype(float).to_numpy()
            estimate, lower, upper = percentile_bootstrap_interval(
                values,
                resamples=int(config["bootstrap_resamples"]),
                seed=int(config["bootstrap_seed"]) + len(summary_rows),
            )
            summary_rows.append(
                {
                    "penalty_condition": condition,
                    "metric": metric,
                    "test_problem_count": int(len(values)),
                    "mean": estimate,
                    "median": float(np.median(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    test_summary = pd.DataFrame(summary_rows)

    primary_metrics = [
        "feasibility_probability",
        "optimal_state_probability",
        "optimal_probability_given_feasible",
        "normalized_feasible_entropy",
        "energy_regret",
        "risk_regret",
    ]
    full_effect_rows: list[dict[str, object]] = []
    for condition in [
        str(value)
        for value in config_condition_order(config)
        if value != "validated_safe_default"
    ]:
        for metric in primary_metrics:
            wide = test.pivot(index="network_id", columns="penalty_condition", values=metric)
            differences = (
                wide[condition].astype(float)
                - wide["validated_safe_default"].astype(float)
            ).to_numpy()
            full_effect_rows.append(
                {
                    "contrast": f"{condition}_minus_validated_safe_default",
                    "metric": metric,
                    **_effect_summary(differences, config, len(full_effect_rows)),
                }
            )
    test_effects = pd.DataFrame(full_effect_rows)
    test_effects["p_value_holm_full_exploratory_family"] = holm_adjust(
        test_effects["p_value_sign_flip"].to_numpy(float)
    )
    confirmation = test_effects[
        test_effects["contrast"]
        == f"{selected}_minus_validated_safe_default"
    ].copy()
    confirmation["p_value_holm_selected_primary_family"] = holm_adjust(
        confirmation["p_value_sign_flip"].to_numpy(float)
    )

    source_reference = pd.read_csv(source / "qaoa_runs.csv")
    source_reference = source_reference[
        (source_reference["K"] == int(config["K"]))
        & (source_reference["depth"] == int(config["depth"]))
        & (source_reference["method"] == str(config["method"]))
        & (source_reference["uncertainty"] == str(config["uncertainty"]))
    ][
        [
            "problem_id",
            "feasibility_probability",
            "optimal_state_probability",
            "expected_hamiltonian_energy",
        ]
    ]
    default = runs[runs["penalty_condition"] == "validated_safe_default"]
    reproduction = default.merge(
        source_reference,
        on="problem_id",
        suffixes=("_new", "_source"),
        validate="one_to_one",
    )
    for metric in [
        "feasibility_probability",
        "optimal_state_probability",
        "expected_hamiltonian_energy",
    ]:
        reproduction[f"{metric}_absolute_error"] = np.abs(
            reproduction[f"{metric}_new"] - reproduction[f"{metric}_source"]
        )
    return {
        "validation_selection": validation_selection,
        "test_summary": test_summary,
        "test_effects_vs_default": test_effects,
        "selected_vs_default_confirmation": confirmation,
        "default_reproduction": reproduction[
            [
                "problem_id",
                "network_id",
                *[
                    f"{metric}_absolute_error"
                    for metric in [
                        "feasibility_probability",
                        "optimal_state_probability",
                        "expected_hamiltonian_energy",
                    ]
                ],
            ]
        ],
        "selection": {
            "selected_penalty_condition": selected,
            "selection_source": "validation seeds 0-2 only",
            "test_seed_ids": list(map(int, config["test_seed_ids"])),
        },
    }


def config_condition_order(config: dict[str, Any]) -> list[str]:
    return [str(dict(item)["name"]) for item in config["penalty_conditions"]]


def finalize(
    config: dict[str, Any], source: Path, verified: dict[str, str]
) -> None:
    paths = sorted(CHECKPOINT_DIRECTORY.glob("*.json"))
    if len(paths) != 350:
        raise RuntimeError(f"expected 350 penalty checkpoints, found {len(paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    runs = pd.DataFrame([record["metric"] for record in records])
    distributions = pd.DataFrame(
        [row for record in records for row in record["feasible_distribution"]]
    )
    history = pd.DataFrame(
        [row for record in records for row in record["training_history"]]
    )
    if runs["run_id"].duplicated().any() or len(distributions) != 350 * 8:
        raise RuntimeError("penalty results are not unique and complete")
    if len(history) != 350 * 16:
        raise RuntimeError("penalty training histories are incomplete")
    required = runs[
        [
            "penalty_value",
            "feasibility_probability",
            "optimal_state_probability",
            "expected_hamiltonian_energy",
            "ground_state_energy",
            "energy_regret",
            "normalized_feasible_entropy",
            "wall_time_seconds",
        ]
    ].to_numpy(float)
    if not np.isfinite(required).all() or np.any(runs["energy_regret"] < -1e-8):
        raise RuntimeError("penalty result contains invalid metrics")
    analysis = _analysis(runs, config, source)
    reproduction = analysis["default_reproduction"]
    assert isinstance(reproduction, pd.DataFrame)
    reproduction_max = float(
        reproduction.filter(like="absolute_error").to_numpy(float).max()
    )
    if reproduction_max > 1e-8:
        raise RuntimeError(
            f"safe-default reproduction differs from primary: {reproduction_max}"
        )

    artifacts: dict[str, pd.DataFrame] = {
        "penalty_runs.csv": runs.sort_values("run_id"),
        "feasible_route_distributions.csv": distributions.sort_values(
            ["run_id", "route_index"]
        ),
        "qaoa_training_history.csv": history.sort_values(["run_id", "iteration"]),
    }
    for key in [
        "validation_selection",
        "test_summary",
        "test_effects_vs_default",
        "selected_vs_default_confirmation",
        "default_reproduction",
    ]:
        value = analysis[key]
        assert isinstance(value, pd.DataFrame)
        artifacts[f"{key}.csv"] = value
    for name, table in artifacts.items():
        _atomic_csv(OUTPUT_DIRECTORY / name, table)
    selection = analysis["selection"]
    assert isinstance(selection, dict)
    _atomic_json(OUTPUT_DIRECTORY / "selected_penalty.json", selection)
    checkpoint_hashes = {
        path.name: file_sha256(path) for path in paths
    }
    _atomic_json(OUTPUT_DIRECTORY / "checkpoint_hashes.json", checkpoint_hashes)
    prior_failure_hashes = {
        path.name: file_sha256(path)
        for path in sorted(FAILURE_DIRECTORY.glob("*.json"))
    }
    _atomic_json(OUTPUT_DIRECTORY / "prior_failure_hashes.json", prior_failure_hashes)

    selected = str(selection["selected_penalty_condition"])
    validation = artifacts["validation_selection.csv"]
    test_summary = artifacts["test_summary.csv"]
    confirmation = artifacts["selected_vs_default_confirmation.csv"]
    compact_summary = test_summary[
        test_summary["metric"].isin(
            [
                "feasibility_probability",
                "optimal_state_probability",
                "optimal_probability_given_feasible",
                "normalized_feasible_entropy",
            ]
        )
    ][["penalty_condition", "metric", "mean", "ci95_lower", "ci95_upper"]]
    if confirmation.empty:
        confirmation_block = (
            "The validation-selected condition is the conservative repository "
            "default itself. A selected-minus-default contrast is therefore "
            "identically zero and is not assigned a redundant hypothesis test."
        )
    else:
        confirmation_block = confirmation[
            [
                "metric",
                "mean_difference",
                "ci95_lower",
                "ci95_upper",
                "p_value_holm_selected_primary_family",
            ]
        ].to_markdown(index=False, floatfmt=".6f")
    report = f"""# Validation Report — Standard-X Penalty Sensitivity

## Status

- Status: **PASS** for 350 K=8, p=3 standard-X trials.
- Independent problems: 50 across five topology families; validation seeds 0-2 (15 problems), test seeds 3-9 (35 problems).
- Seven frozen penalty conditions span below threshold, the exact zero/feasible tie, just above threshold, the repository's conservative default, and a 10× threshold.
- Every test condition is reported; no test row was used for tuning.
- The repository safe-default condition reproduces the accepted primary metrics with maximum absolute error {reproduction_max:.3e}.
- Prior failed checkpoint attempts retained: {len(prior_failure_hashes)}. They contain no accepted metric because strict JSON serialization rejected undefined diagnostics before checkpoint creation; successful reruns use the unchanged frozen design.

## Validation-only selection

The frozen eligibility/selection rule selected `{selected}`.

{validation.to_markdown(index=False, floatfmt='.6f')}

## Complete held-out test grid

{compact_summary.to_markdown(index=False, floatfmt='.6f')}

## Selected versus conservative default on held-out test networks

{confirmation_block}

## Guardrails

1. Penalty changes the Hamiltonian scale as well as constraint separation. The optimizer seed, 15-step Adam budget, and step size are fixed across penalties, so scale sensitivity is part of the measured behavior.
2. Conditions below the exact threshold deliberately encode an invalid all-zero global ground state; their low feasibility is expected and is not evidence about a correctly encoded constraint.
3. The repository's `max(risk)+1` rule is sufficient but more conservative than the exact zero-state threshold `P > min(risk)` for nonnegative route risks.
4. K=8 is exactly enumerable and classically trivial. This sensitivity study addresses an encoding objection; it cannot support advantage or speedup.
"""
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    artifact_names = [
        *artifacts,
        "trial_design.csv",
        "runtime_benchmark.csv",
        "selected_penalty.json",
        "checkpoint_hashes.json",
        "prior_failure_hashes.json",
        "VALIDATION_REPORT.md",
    ]
    manifest = build_experiment_manifest(
        str(config["experiment_name"]),
        REPOSITORY_ROOT,
        {"bootstrap_seed": int(config["bootstrap_seed"])},
        config,
        CONFIG_PATH,
    )
    manifest["source_sha256"] = verified | {
        "runner": file_sha256(Path(__file__)),
        "penalty_sensitivity": file_sha256(
            REPOSITORY_ROOT / "src" / "penalty_sensitivity.py"
        ),
        "feasibility_qaoa": file_sha256(
            REPOSITORY_ROOT / "src" / "feasibility_qaoa.py"
        ),
    }
    manifest["artifact_sha256"] = {
        name: file_sha256(OUTPUT_DIRECTORY / name) for name in artifact_names
    }
    manifest["checkpoint_sha256"] = checkpoint_hashes
    manifest["prior_failed_attempt_sha256"] = prior_failure_hashes
    manifest["claim_exclusions"] = [
        "quantum_advantage",
        "test_set_tuned_penalty",
        "universal_penalty_optimum",
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
    specs, design = prepare(config, source)
    if args.mode == "prepare":
        print(
            f"Prepared {len(design)} penalty trials: "
            f"{(design['data_split'] == 'validation').sum()} validation and "
            f"{(design['data_split'] == 'test').sum()} test.",
            flush=True,
        )
    if args.mode in {"benchmark", "all"}:
        benchmark(specs, config)
    if args.mode in {"run", "all"}:
        if not (OUTPUT_DIRECTORY / "runtime_benchmark.csv").is_file():
            raise RuntimeError("run the penalty benchmark before trials")
        workers = int(config["default_workers"] if args.workers is None else args.workers)
        if workers <= 0:
            raise ValueError("workers must be positive")
        run_trials(specs, workers=workers)
    if args.mode in {"finalize", "all"}:
        finalize(config, source, verified)


if __name__ == "__main__":
    main()
