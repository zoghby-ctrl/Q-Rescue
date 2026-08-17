from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "research" / "targeted_k16_scaling"


def test_k16_preparation_is_a_matched_extension_of_k8() -> None:
    problems = pd.read_csv(OUTPUT / "problem_summary.csv")
    routes = pd.read_csv(OUTPUT / "candidate_routes.csv")
    prefix = pd.read_csv(OUTPUT / "k8_prefix_reproduction.csv")
    registry = pd.read_csv(OUTPUT / "network_registry.csv")

    assert len(problems) == 18
    assert problems["problem_id"].nunique() == 18
    assert set(problems["K"]) == {16}
    assert len(routes) == 18 * 16
    assert routes.groupby("problem_id").size().eq(16).all()
    assert len(prefix) == 18
    assert prefix["route_nodes_exact_match"].all()
    error_columns = [column for column in prefix if column.endswith("error")]
    assert np.max(np.abs(prefix[error_columns].to_numpy(float))) < 1e-9
    assert registry["K16_complete"].all()
    assert dict(
        registry[["topology", "topology_index"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == {"grid": 0, "irregular_geometric": 2, "sparse_mixed": 3}
