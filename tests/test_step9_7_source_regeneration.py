from __future__ import annotations

import joblib

from experiments.run_step9_7_adaptive_conformal import (
    _coverage_intervals,
    _load_config,
    _regenerate_step9_observations,
    _source_path,
    _verify_static_reproduction,
)


def test_regenerated_step9_floats_reproduce_accepted_static_intervals() -> None:
    config = _load_config()
    observations, checks = _regenerate_step9_observations(config)
    models = joblib.load(_source_path(config, "conformal_models"))
    intervals, _, _ = _coverage_intervals(config, observations, models)
    reproduction = _verify_static_reproduction(
        intervals,
        _source_path(config, "existing_intervals"),
        float(config["target_alpha"]),
    )

    assert checks["observation_count"] == 9408
    assert checks["observation_id_and_metadata_exact_match"] is True
    assert checks["max_absolute_numeric_csv_roundtrip_delta"] < 1e-12
    assert max(reproduction.values()) < 1e-9
