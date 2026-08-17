from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "research" / "step9_2_topology_uncertainty_qaoa"


def test_k4_and_k8_share_deadline_and_common_route_objectives() -> None:
    problems = pd.read_csv(RESULTS / "problem_summary.csv")
    routes = pd.read_csv(RESULTS / "candidate_route_problems.csv")

    assert problems.groupby("network_id")["deadline_minutes"].nunique().max() == 1
    k4 = routes[routes["K"] == 4].set_index(
        ["network_id", "uncertainty", "route_index"]
    ).sort_index()
    k8_prefix = routes[(routes["K"] == 8) & (routes["route_index"] < 4)].set_index(
        ["network_id", "uncertainty", "route_index"]
    ).sort_index()
    assert k4.index.equals(k8_prefix.index)

    for column in [
        "prediction_minutes",
        "uncertainty_half_width_minutes",
        "realized_travel_time_minutes",
        "A_expected_time",
        "B_expected_plus_deadline",
        "C_expected_plus_deadline_plus_uncertainty",
    ]:
        np.testing.assert_allclose(
            k4[column].to_numpy(dtype=float),
            k8_prefix[column].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
    assert np.array_equal(k4["ordered_nodes"], k8_prefix["ordered_nodes"])
