"""Build and exhaustively verify generalized K=4/6/8 route-risk QUBOs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.qubo import calculate_ising_energy
from src.reproducibility import build_experiment_manifest, write_json
from src.risk_qubo import (
    brute_force_risk_qubo,
    build_risk_qubo,
    qubo_scaling_record,
    risk_qubo_to_ising,
)


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "qubo_benchmark.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "research" / "step7_qubo"


def run() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_directory = REPOSITORY_ROOT / config["routing_risk_artifacts"]
    scores = pd.read_csv(source_directory / "route_risk_scores.csv")
    scores = scores[
        scores["variant"].isin(config["variants"])
        & scores["K"].isin(config["candidate_counts"])
    ].copy()

    coefficient_rows: list[dict[str, object]] = []
    assignment_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for keys, group in scores.groupby(
        ["network_id", "topology", "K", "variant"], sort=True
    ):
        network_id, topology, k_value, variant = keys
        group = group.sort_values("route_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
        risk_vector = group["risk_score"].to_numpy(dtype=float)
        route_ids = group["route_id"].tolist()
        qubo = build_risk_qubo(
            risk_vector, penalty_margin=float(config["penalty_margin"])
        )
        table = brute_force_risk_qubo(
            qubo, maximum_variables=int(config["maximum_bruteforce_variables"])
        )
        table.insert(0, "network_id", network_id)
        table.insert(1, "topology", topology)
        table.insert(2, "K", int(k_value))
        table.insert(3, "variant", variant)
        table["selected_route_id"] = table["selected_route_index"].map(
            lambda index: route_ids[int(index)] if pd.notna(index) else None
        )
        assignment_tables.append(table)

        _, offset, z_coefficients, zz_coefficients = risk_qubo_to_ising(qubo)
        discrepancies: list[float] = []
        for row in table.itertuples():
            assignment = tuple(int(bit) for bit in row.bitstring)
            ising_energy = calculate_ising_energy(
                assignment, offset, z_coefficients, zz_coefficients
            )
            discrepancies.append(abs(float(row.qubo_cost) - ising_energy))

        best_row = table.loc[table["qubo_cost"].idxmin()]
        exact_index = int(np.argmin(risk_vector))
        scaling = qubo_scaling_record(qubo)
        summary_rows.append(
            {
                "network_id": network_id,
                "topology": topology,
                "K": int(k_value),
                "variant": variant,
                "penalty": qubo.penalty,
                "exact_route_index": exact_index,
                "exact_route_id": route_ids[exact_index],
                "exact_risk_cost": float(risk_vector[exact_index]),
                "qubo_route_id": best_row["selected_route_id"],
                "qubo_bitstring": best_row["bitstring"],
                "qubo_ground_cost": float(best_row["qubo_cost"]),
                "ground_state_valid": bool(best_row["valid"]),
                "matches_direct_argmin": best_row["selected_route_id"]
                == route_ids[exact_index],
                "maximum_ising_discrepancy": max(discrepancies),
                **scaling,
            }
        )

        coefficient_rows.append(
            {
                "network_id": network_id,
                "topology": topology,
                "K": int(k_value),
                "variant": variant,
                "term_type": "constant",
                "first_index": None,
                "second_index": None,
                "coefficient": qubo.constant,
            }
        )
        coefficient_rows.extend(
            {
                "network_id": network_id,
                "topology": topology,
                "K": int(k_value),
                "variant": variant,
                "term_type": "linear",
                "first_index": index,
                "second_index": None,
                "coefficient": coefficient,
            }
            for index, coefficient in enumerate(qubo.linear)
        )
        coefficient_rows.extend(
            {
                "network_id": network_id,
                "topology": topology,
                "K": int(k_value),
                "variant": variant,
                "term_type": "quadratic",
                "first_index": first,
                "second_index": second,
                "coefficient": coefficient,
            }
            for (first, second), coefficient in qubo.quadratic.items()
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    assignments = pd.concat(assignment_tables, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    summary.to_csv(OUTPUT_DIRECTORY / "qubo_verification_summary.csv", index=False)
    assignments.to_csv(OUTPUT_DIRECTORY / "brute_force_assignments.csv", index=False)
    coefficients.to_csv(OUTPUT_DIRECTORY / "qubo_coefficients.csv", index=False)
    summary.drop_duplicates(
        ["K", "search_space_size", "qubo_total_terms", "hamiltonian_nonidentity_terms"]
    ).loc[
        :,
        [
            "K",
            "number_of_qubits",
            "search_space_size",
            "qubo_linear_terms",
            "qubo_quadratic_terms",
            "qubo_total_terms",
            "hamiltonian_z_terms",
            "hamiltonian_zz_terms",
            "hamiltonian_nonidentity_terms",
        ],
    ].sort_values("K").to_csv(OUTPUT_DIRECTORY / "qubo_scaling.csv", index=False)

    manifest = build_experiment_manifest(
        experiment_name=config["experiment_name"],
        repository_root=REPOSITORY_ROOT,
        random_seeds={},
        parameters=config,
        config_path=CONFIG_PATH,
    )
    write_json(OUTPUT_DIRECTORY / "manifest.json", manifest)
    print(
        summary.groupby("K").agg(
            instances=("network_id", "size"),
            all_valid=("ground_state_valid", "all"),
            all_match=("matches_direct_argmin", "all"),
            max_ising_error=("maximum_ising_discrepancy", "max"),
        ).to_string()
    )
    print(f"Enumerated {len(assignments)} total assignments.")
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run()
