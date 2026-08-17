from __future__ import annotations

import json

import numpy as np

from experiments.run_standard_x_penalty_sensitivity import _atomic_json
from src.penalty_sensitivity import (
    build_unrestricted_penalty_qubo,
    exact_zero_state_penalty_threshold,
    penalty_ground_state_record,
)
from src.risk_qubo import build_risk_qubo


def test_exact_threshold_classifies_zero_state_and_feasible_ground_set() -> None:
    risks = [2.0, 3.0, 5.0]
    threshold = exact_zero_state_penalty_threshold(risks)
    below = penalty_ground_state_record(
        build_unrestricted_penalty_qubo(risks, 0.95 * threshold)
    )
    tied = penalty_ground_state_record(
        build_unrestricted_penalty_qubo(risks, threshold)
    )
    above = penalty_ground_state_record(
        build_unrestricted_penalty_qubo(risks, 1.05 * threshold)
    )

    assert below["zero_state_is_global_ground_state"] is True
    assert below["global_ground_states_any_feasible"] is False
    assert tied["zero_state_is_global_ground_state"] is True
    assert tied["global_ground_states_any_feasible"] is True
    assert above["zero_state_is_global_ground_state"] is False
    assert above["global_ground_states_all_feasible"] is True


def test_unrestricted_builder_matches_safe_builder_above_maximum() -> None:
    risks = [1.0, 2.5, 4.0]
    penalty = 5.0
    unrestricted = build_unrestricted_penalty_qubo(risks, penalty)
    safe = build_risk_qubo(risks, penalty=penalty)

    assert unrestricted == safe
    assert np.isclose(unrestricted.matrix(), safe.matrix()).all()


def test_checkpoint_serializer_preserves_undefined_diagnostics_as_null(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    _atomic_json(
        path,
        {
            "python_nan": float("nan"),
            "numpy_inf": np.float64(np.inf),
            "numpy_integer": np.int64(4),
            "nested": [np.bool_(True), np.float64(0.25)],
        },
    )

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {
        "python_nan": None,
        "numpy_inf": None,
        "numpy_integer": 4,
        "nested": [True, 0.25],
    }
