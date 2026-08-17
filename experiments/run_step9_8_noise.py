"""Run the targeted Step 9.8 ideal/shot/local-noise comparison."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pennylane import numpy as pnp


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.feasibility_qaoa import (  # noqa: E402
    create_probability_qnode,
    finite_shot_resample,
)
from src.qaoa_diagnostics import summarize_qaoa_distribution  # noqa: E402
from src.qaoa_noise import (  # noqa: E402
    LayerNoiseParameters,
    apply_symmetric_readout_error,
    create_layer_noise_probability_qnode,
    damping_probability,
    effective_error_probability,
)
from src.reproducibility import (  # noqa: E402
    build_experiment_manifest,
    file_sha256,
    write_json,
)
from src.risk_qubo import build_risk_qubo, risk_qubo_to_ising  # noqa: E402
from src.statistical_inference import percentile_bootstrap_interval  # noqa: E402


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "noise_robustness.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step9_8_noise"


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _source_directory(config: dict[str, Any]) -> Path:
    path = (REPOSITORY_ROOT / config["source_directory"]).resolve()
    if REPOSITORY_ROOT.resolve() not in path.parents or not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _verify_sources(source: Path) -> dict[str, str]:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Step 9.2 must be finalized before Step 9.8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified: dict[str, str] = {"manifest.json": file_sha256(manifest_path)}
    for name in ["qaoa_runs.csv", "problem_summary.csv", "validation_checks.json"]:
        path = source / name
        actual = file_sha256(path)
        if manifest["artifact_sha256"].get(name) != actual:
            raise RuntimeError(f"Step 9.2 source hash mismatch: {name}")
        verified[name] = actual
    checks = json.loads((source / "validation_checks.json").read_text())
    if not checks.get("all_checks_pass") or checks.get("trial_count") != 1800:
        raise RuntimeError("Step 9.2 is not complete")
    return verified


def _noise_parameters(config: dict[str, Any], method: str) -> LayerNoiseParameters:
    model = dict(config["noise_model"])
    one_error = float(model["one_qubit_gate_error"])
    two_error = float(model["two_qubit_gate_error"])
    one_duration = float(model["one_qubit_gate_duration_nanoseconds"])
    two_duration = float(model["two_qubit_gate_duration_nanoseconds"])
    t1 = float(model["amplitude_damping_T1_microseconds"])
    cost_one = int(model["cost_single_qubit_interactions_per_wire"])
    cost_two = int(model["cost_two_qubit_interactions_per_wire"])
    cost_duration = cost_one * one_duration + cost_two * two_duration
    if method == "standard_x":
        prep_depolarizing = one_error
        prep_duration = one_duration
        mixer_one = int(model["standard_x_mixer_single_qubit_interactions_per_wire"])
        mixer_two = 0
    elif method == "feasible_xy":
        prep_depolarizing = float(
            model["xy_state_preparation_depolarizing_probability"]
        )
        prep_duration = float(model["xy_state_preparation_duration_nanoseconds"])
        mixer_one = 0
        mixer_two = int(model["xy_ring_mixer_two_qubit_interactions_per_wire"])
    else:
        raise ValueError(method)
    mixer_duration = mixer_one * one_duration + mixer_two * two_duration
    return LayerNoiseParameters(
        preparation_depolarizing=prep_depolarizing,
        preparation_amplitude_damping=damping_probability(prep_duration, t1),
        cost_depolarizing=effective_error_probability(
            one_error, cost_one, two_error, cost_two
        ),
        cost_amplitude_damping=damping_probability(cost_duration, t1),
        mixer_depolarizing=effective_error_probability(
            one_error, mixer_one, two_error, mixer_two
        ),
        mixer_amplitude_damping=damping_probability(mixer_duration, t1),
        readout_bit_flip=float(model["readout_bit_flip_probability"]),
    )


def _probabilities_from_counts(counts: dict[str, int], k: int, shots: int) -> np.ndarray:
    values = np.zeros(2**k, dtype=float)
    for bitstring, count in counts.items():
        values[int(bitstring, 2)] = int(count) / shots
    if not np.isclose(values.sum(), 1.0):
        raise RuntimeError("shot probabilities do not sum to one")
    return values


def _metric_row(
    diagnostics: object,
    source_row: pd.Series,
    *,
    condition: str,
    shots: int,
    shot_seed: int | None,
    evaluation_seconds: float,
) -> dict[str, object]:
    metrics = dict(diagnostics.metrics)
    return {
        "source_run_id": source_row["run_id"],
        "problem_id": source_row["problem_id"],
        "network_id": source_row["network_id"],
        "topology": source_row["topology"],
        "matched_seed_id": int(source_row["matched_seed_id"]),
        "uncertainty": source_row["uncertainty"],
        "K": int(source_row["K"]),
        "depth": int(source_row["depth"]),
        "method": source_row["method"],
        "condition": condition,
        "shots": shots,
        "shot_seed": shot_seed,
        "evaluation_seconds": evaluation_seconds,
        "circuit_depth": int(source_row["circuit_depth"]),
        "gate_count": int(source_row["gate_count"]),
        "two_qubit_gate_count_logical": int(
            source_row["two_qubit_gate_count_logical"]
        ),
        "feasibility_probability": metrics["feasibility_probability"],
        "optimal_state_probability": metrics["optimal_state_probability"],
        "optimal_probability_given_feasible": metrics[
            "optimal_probability_given_feasible"
        ],
        "expected_hamiltonian_energy": metrics["expected_hamiltonian_energy"],
        "ground_state_energy": metrics["ground_state_energy"],
        "energy_regret": metrics["energy_regret"],
        "normalized_feasible_entropy": metrics["normalized_feasible_entropy"],
        "selected_route_index": metrics["selected_route_index"],
        "selected_route_is_optimal": metrics["selected_route_is_optimal"],
        "risk_regret": metrics["risk_regret"],
        "top1_feasible_probability_conditional": metrics[
            "top1_feasible_probability_conditional"
        ],
        "top2_feasible_probability_conditional": metrics[
            "top2_feasible_probability_conditional"
        ],
    }


def _condition_distribution(
    diagnostics: object, source_run_id: str, condition: str
) -> pd.DataFrame:
    table = diagnostics.feasible_distribution.copy()
    table.insert(0, "source_run_id", source_run_id)
    table.insert(1, "condition", condition)
    return table


def _run(config: dict[str, Any], source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(source / "qaoa_runs.csv")
    problems = pd.read_csv(source / "problem_summary.csv").set_index("problem_id")
    selected = runs[
        (runs["K"] == int(config["K"]))
        & (runs["depth"].isin(config["depths"]))
        & (runs["method"].isin(config["methods"]))
        & (runs["topology"].isin(config["topologies"]))
        & (runs["uncertainty"] == config["uncertainty"])
        & (runs["matched_seed_id"].isin(config["matched_seed_ids"]))
    ].sort_values(["topology", "matched_seed_id", "depth", "method"], kind="stable")
    expected_source_runs = (
        len(config["topologies"])
        * len(config["matched_seed_ids"])
        * len(config["depths"])
        * len(config["methods"])
    )
    if len(selected) != expected_source_runs:
        raise RuntimeError(
            f"noise subset expected {expected_source_runs} source runs, found {len(selected)}"
        )
    metric_rows: list[dict[str, object]] = []
    distributions: list[pd.DataFrame] = []
    count_rows: list[dict[str, object]] = []
    reproduction_rows: list[dict[str, object]] = []
    shots = int(config["shots"])
    for source_index, source_row in selected.reset_index(drop=True).iterrows():
        problem = problems.loc[source_row["problem_id"]]
        risk_vector = np.asarray(json.loads(problem["risk_vector_json"]), dtype=float)
        qubo = build_risk_qubo(risk_vector)
        hamiltonian, _, _, _ = risk_qubo_to_ising(qubo)
        gammas = pnp.array(json.loads(source_row["gammas_json"]), requires_grad=False)
        betas = pnp.array(json.loads(source_row["betas_json"]), requires_grad=False)
        ideal_qnode = create_probability_qnode(
            hamiltonian,
            number_of_qubits=int(config["K"]),
            depth=int(source_row["depth"]),
            method=str(source_row["method"]),
        )
        started = time.perf_counter()
        ideal_probabilities = np.asarray(ideal_qnode(gammas, betas), dtype=float)
        ideal_seconds = time.perf_counter() - started
        ideal = summarize_qaoa_distribution(ideal_probabilities, qubo)
        reproduction_rows.append(
            {
                "source_run_id": source_row["run_id"],
                "feasibility_absolute_error": abs(
                    float(ideal.metrics["feasibility_probability"])
                    - float(source_row["feasibility_probability"])
                ),
                "optimal_probability_absolute_error": abs(
                    float(ideal.metrics["optimal_state_probability"])
                    - float(source_row["optimal_state_probability"])
                ),
                "energy_absolute_error": abs(
                    float(ideal.metrics["expected_hamiltonian_energy"])
                    - float(source_row["expected_hamiltonian_energy"])
                ),
            }
        )
        metric_rows.append(
            _metric_row(
                ideal,
                source_row,
                condition="ideal_analytic",
                shots=0,
                shot_seed=None,
                evaluation_seconds=ideal_seconds,
            )
        )
        distributions.append(
            _condition_distribution(ideal, source_row["run_id"], "ideal_analytic")
        )

        shot_seed = int(config["shot_seed_base"]) + source_index
        started = time.perf_counter()
        ideal_counts = finite_shot_resample(
            ideal_probabilities,
            number_of_qubits=int(config["K"]),
            shots=shots,
            seed=shot_seed,
        )
        shot_probabilities = _probabilities_from_counts(
            ideal_counts, int(config["K"]), shots
        )
        shot_diagnostics = summarize_qaoa_distribution(shot_probabilities, qubo)
        shot_seconds = time.perf_counter() - started
        metric_rows.append(
            _metric_row(
                shot_diagnostics,
                source_row,
                condition="finite_shot_noiseless",
                shots=shots,
                shot_seed=shot_seed,
                evaluation_seconds=shot_seconds,
            )
        )
        distributions.append(
            _condition_distribution(
                shot_diagnostics, source_row["run_id"], "finite_shot_noiseless"
            )
        )
        for bitstring_index in range(2 ** int(config["K"])):
            bitstring = format(bitstring_index, f"0{int(config['K'])}b")
            count_rows.append(
                {
                    "source_run_id": source_row["run_id"],
                    "condition": "finite_shot_noiseless",
                    "bitstring": bitstring,
                    "count": int(ideal_counts.get(bitstring, 0)),
                    "shots": shots,
                    "shot_seed": shot_seed,
                }
            )

        noise = _noise_parameters(config, str(source_row["method"]))
        noisy_qnode = create_layer_noise_probability_qnode(
            hamiltonian,
            number_of_qubits=int(config["K"]),
            depth=int(source_row["depth"]),
            method=str(source_row["method"]),
            noise=noise,
        )
        started = time.perf_counter()
        noisy_probabilities = np.asarray(noisy_qnode(gammas, betas), dtype=float)
        noisy_probabilities = apply_symmetric_readout_error(
            noisy_probabilities,
            number_of_qubits=int(config["K"]),
            bit_flip_probability=noise.readout_bit_flip,
        )
        noise_seed = int(config["noise_seed_base"]) + source_index
        noisy_counts = finite_shot_resample(
            noisy_probabilities,
            number_of_qubits=int(config["K"]),
            shots=shots,
            seed=noise_seed,
        )
        noisy_shot_probabilities = _probabilities_from_counts(
            noisy_counts, int(config["K"]), shots
        )
        noisy_diagnostics = summarize_qaoa_distribution(noisy_shot_probabilities, qubo)
        noisy_seconds = time.perf_counter() - started
        metric_rows.append(
            _metric_row(
                noisy_diagnostics,
                source_row,
                condition="noisy_shot",
                shots=shots,
                shot_seed=noise_seed,
                evaluation_seconds=noisy_seconds,
            )
        )
        distributions.append(
            _condition_distribution(
                noisy_diagnostics, source_row["run_id"], "noisy_shot"
            )
        )
        for bitstring_index in range(2 ** int(config["K"])):
            bitstring = format(bitstring_index, f"0{int(config['K'])}b")
            count_rows.append(
                {
                    "source_run_id": source_row["run_id"],
                    "condition": "noisy_shot",
                    "bitstring": bitstring,
                    "count": int(noisy_counts.get(bitstring, 0)),
                    "shots": shots,
                    "shot_seed": noise_seed,
                }
            )
        print(
            f"Noise case {source_index + 1}/{len(selected)}: {source_row['run_id']}",
            flush=True,
        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(distributions, ignore_index=True),
        pd.DataFrame(count_rows),
        pd.DataFrame(reproduction_rows),
    )


def _aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "feasibility_probability",
        "optimal_state_probability",
        "optimal_probability_given_feasible",
        "expected_hamiltonian_energy",
        "energy_regret",
        "normalized_feasible_entropy",
        "risk_regret",
        "selected_route_is_optimal",
        "evaluation_seconds",
    ]
    aggregate = (
        runs.groupby(["depth", "method", "condition"], sort=True)[metrics]
        .agg(["count", "mean", "median", "std"])
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def _paired_effects(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "feasibility_probability",
        "optimal_state_probability",
        "optimal_probability_given_feasible",
        "normalized_feasible_entropy",
        "risk_regret",
    ]
    rows: list[dict[str, object]] = []
    for (depth, method), subset in runs.groupby(["depth", "method"], sort=True):
        for comparison in ["finite_shot_noiseless", "noisy_shot"]:
            for metric in metrics:
                wide = subset.pivot(
                    index="source_run_id", columns="condition", values=metric
                ).dropna()
                differences = (
                    wide[comparison] - wide["ideal_analytic"]
                ).to_numpy(dtype=float)
                estimate, lower, upper = percentile_bootstrap_interval(
                    differences, resamples=20_000, seed=981000 + len(rows)
                )
                rows.append(
                    {
                        "depth": int(depth),
                        "method": method,
                        "contrast": f"{comparison}_minus_ideal_analytic",
                        "metric": metric,
                        "matched_run_count": len(differences),
                        "mean_difference": estimate,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                    }
                )
    return pd.DataFrame(rows)


def _report(runs: pd.DataFrame, aggregate: pd.DataFrame, effects: pd.DataFrame) -> str:
    summary = aggregate[
        [
            "depth",
            "method",
            "condition",
            "feasibility_probability_mean",
            "optimal_state_probability_mean",
            "optimal_probability_given_feasible_mean",
            "normalized_feasible_entropy_mean",
            "risk_regret_mean",
        ]
    ]
    xy_effect = effects[
        (effects["method"] == "feasible_xy")
        & (effects["contrast"] == "noisy_shot_minus_ideal_analytic")
        & (effects["metric"] == "feasibility_probability")
    ]
    return f"""# Validation Report — Step 9.8 Local Noise-Model Simulation

## Status

- Status: **PASS** for the targeted K=4 local density-matrix noise-model study.
- Source ideal QAOA runs: {runs['source_run_id'].nunique()}.
- Condition rows: {len(runs)} (ideal analytic, 4,096-shot noiseless, noisy shot).
- Independent problem instances: {runs['network_id'].nunique()} across five topology families.
- Depths: p=1 and p=3; mixers: standard-X and feasible-XY.
- Angles are inherited from the ideal optimizer and are not reoptimized under noise.

## Results

{summary.to_markdown(index=False, floatfmt='.4f')}

For XY, the noisy-shot minus ideal feasibility effects by depth are:

{xy_effect[['depth', 'mean_difference', 'ci95_lower', 'ci95_upper']].to_markdown(index=False, floatfmt='.4f')}

## Interpretation guardrails

1. This is a phenomenological local channel simulation based on declared gate-error rates, T1, block durations, and readout flips. It is not calibrated to an IBM fake backend and is not a hardware result.
2. Density-matrix evolution remains an exact local classical simulation; shot sampling adds finite-sampling variability.
3. Logical circuit resources come from the inherited ideal circuit. State-preparation and transpilation costs are not hardware resource estimates.
4. K=4 is classically trivial. No speedup, advantage, or deployment claim is supported.
5. The study tests whether the ideal feasibility pattern survives this declared model, not all plausible NISQ noise.
"""


def main() -> None:
    config = _load_config()
    source = _source_directory(config)
    verified = _verify_sources(source)
    runs, distributions, counts, reproduction = _run(config, source)
    if len(runs) != 120 or len(counts) != 40 * 2 * 16:
        raise RuntimeError("unexpected Step 9.8 result count")
    max_error = reproduction[
        [
            "feasibility_absolute_error",
            "optimal_probability_absolute_error",
            "energy_absolute_error",
        ]
    ].to_numpy(dtype=float).max()
    if max_error > 1e-8:
        raise RuntimeError(f"ideal-source reproduction error too large: {max_error}")
    aggregate = _aggregate(runs)
    effects = _paired_effects(runs)
    noise_parameters = pd.DataFrame(
        [
            {"method": method, **_noise_parameters(config, method).__dict__}
            for method in config["methods"]
        ]
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "noise_runs.csv": runs,
        "feasible_route_distributions.csv": distributions,
        "measurement_counts.csv": counts,
        "ideal_reproduction_checks.csv": reproduction,
        "noise_aggregate.csv": aggregate,
        "noise_paired_effects.csv": effects,
        "effective_noise_parameters.csv": noise_parameters,
    }
    for name, table in artifacts.items():
        table.to_csv(OUTPUT_DIRECTORY / name, index=False)
    report = _report(runs, aggregate, effects)
    (OUTPUT_DIRECTORY / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    manifest = build_experiment_manifest(
        experiment_name=str(config["experiment_name"]),
        repository_root=REPOSITORY_ROOT,
        random_seeds={
            "shot_seed_base": int(config["shot_seed_base"]),
            "noise_seed_base": int(config["noise_seed_base"]),
            "effect_bootstrap_seed_base": 981000,
        },
        parameters=config,
        config_path=CONFIG_PATH,
    )
    manifest.update(
        {
            "source_sha256": verified
            | {
                "runner": file_sha256(Path(__file__)),
                "qaoa_noise": file_sha256(REPOSITORY_ROOT / "src" / "qaoa_noise.py"),
                "qaoa_diagnostics": file_sha256(
                    REPOSITORY_ROOT / "src" / "qaoa_diagnostics.py"
                ),
            },
            "artifact_sha256": {
                name: file_sha256(OUTPUT_DIRECTORY / name) for name in artifacts
            }
            | {"VALIDATION_REPORT.md": file_sha256(OUTPUT_DIRECTORY / "VALIDATION_REPORT.md")},
            "claim_exclusions": [
                "hardware_result",
                "backend_specific_behavior",
                "quantum_advantage",
                "noise_generality",
            ],
        }
    )
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    print(f"Step 9.8 complete: {len(runs)} condition rows.", flush=True)


if __name__ == "__main__":
    main()
