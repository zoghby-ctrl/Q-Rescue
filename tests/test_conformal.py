"""Independent checks for manual split-conformal regression."""

import numpy as np
import pandas as pd
import pytest

from src.conformal import (
    INTERVAL_COLUMNS,
    SUPPORTED_ALPHAS,
    calibrate_split_conformal,
    calibrate_supported_alphas,
    calibrate_relative_split_conformal,
    calibrate_locally_scaled_split_conformal,
    compute_coverage_metrics,
    evaluate_conformal,
    finite_sample_quantile_level,
    finite_sample_residual_quantile,
    train_residual_scale_model,
)
from src.traffic_model import TARGET_COLUMN, train_traffic_model


def _traffic_frame(number_of_rows: int = 36) -> pd.DataFrame:
    row = np.arange(number_of_rows, dtype=float)
    hour = np.mod(row, 24.0)
    day = np.mod(row, 7.0)
    congestion = 0.1 + np.mod(row, 8.0) / 10.0
    base_time = 8.0 + np.mod(row, 9.0)
    frame = pd.DataFrame(
        {
            "observation_id": [f"obs-{index}" for index in range(number_of_rows)],
            "network_id": [f"network-{index % 2}" for index in range(number_of_rows)],
            "topology": ["grid" if index % 2 == 0 else "radial" for index in range(number_of_rows)],
            "scenario": ["normal" if index % 3 else "incident" for index in range(number_of_rows)],
            "shift_name": ["none"] * number_of_rows,
            "normalized_distance": 0.2 + np.mod(row, 10.0) / 12.0,
            "base_travel_time_min": base_time,
            "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            "day_sin": np.sin(2.0 * np.pi * day / 7.0),
            "day_cos": np.cos(2.0 * np.pi * day / 7.0),
            "congestion": congestion,
            "average_speed_kph": 58.0 - 25.0 * congestion,
            "rain_indicator": (np.mod(row, 6.0) == 0).astype(float),
            "incident_indicator": (np.mod(row, 10.0) == 0).astype(float),
            "road_class_code": np.mod(row, 4.0),
            "intersection_count": 2.0 + np.mod(row, 5.0),
            "historical_delay_minutes": 0.5 + np.mod(row, 7.0) * 0.4,
        }
    )
    frame[TARGET_COLUMN] = (
        base_time
        + 6.0 * congestion
        + 2.0 * frame["rain_indicator"]
        + 4.0 * frame["incident_indicator"]
        + 0.5 * frame["historical_delay_minutes"]
    )
    return frame


def test_finite_sample_higher_quantile_matches_independent_math() -> None:
    """The implemented correction is checked outside the calibration wrapper."""

    scores = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.50
    expected_level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    expected_q = np.quantile(scores, expected_level, method="higher")

    assert np.isclose(finite_sample_quantile_level(len(scores), alpha), 0.60)
    assert finite_sample_residual_quantile(scores, alpha) == expected_q == 4.0


def test_coverage_math_is_inclusive_and_uses_realized_widths() -> None:
    """Coverage includes endpoints and widths come from returned bounds."""

    metrics = compute_coverage_metrics(
        actual=[1.0, 2.0, 3.0, 4.0],
        lower_bound=[0.0, 2.0, 3.1, 0.0],
        upper_bound=[1.0, 2.5, 4.0, 3.9],
        alpha=0.20,
    )

    assert metrics.n_samples == 4
    assert np.isclose(metrics.nominal_coverage, 0.80)
    assert np.isclose(metrics.empirical_coverage, 0.50)
    assert np.isclose(metrics.mean_interval_width, 1.575)
    assert np.isclose(metrics.median_interval_width, 0.95)


def test_intervals_preserve_metadata_clip_bounds_and_evaluate_shift() -> None:
    """Arbitrary shift labels survive prediction and grouped evaluation."""

    frame = _traffic_frame()
    model = train_traffic_model("linear_regression", frame.iloc[:16])
    calibration = frame.iloc[16:26].copy()
    calibration[TARGET_COLUMN] += 30.0
    conformal = calibrate_split_conformal(model, calibration, alpha=0.10)

    shifted = frame.iloc[26:].copy()
    shifted.loc[:, "shift_name"] = "higher_congestion"
    shifted.loc[:, "congestion"] += 0.15
    intervals = conformal.predict(shifted)

    assert list(intervals.columns[-4:]) == list(INTERVAL_COLUMNS)
    for metadata_column in (
        "observation_id",
        "network_id",
        "topology",
        "scenario",
        "shift_name",
        TARGET_COLUMN,
    ):
        assert metadata_column in intervals.columns
    assert (intervals["lower_bound"] >= 0.0).all()
    assert (intervals["upper_bound"] >= intervals["lower_bound"]).all()
    assert (intervals["interval_half_width"] == conformal.calibration_quantile).all()
    assert (intervals["lower_bound"] == 0.0).any()

    evaluation = evaluate_conformal(conformal, shifted)
    assert evaluation.overall.shape == (1, 6)
    assert evaluation.by_network.shape == (2, 7)
    assert evaluation.by_topology.shape == (2, 7)
    assert len(evaluation.intervals) == len(shifted)

    calibrated_levels = calibrate_supported_alphas(model, calibration)
    assert tuple(calibrated_levels) == SUPPORTED_ALPHAS


def test_invalid_alpha_empty_calibration_and_missing_target_fail() -> None:
    """Invalid calibration inputs fail before an interval can be reported."""

    frame = _traffic_frame()
    model = train_traffic_model("linear_regression", frame.iloc[:16])
    calibration = frame.iloc[16:26]

    for alpha in (0.0, 1.0, -0.1, np.nan):
        with pytest.raises(ValueError, match="alpha"):
            calibrate_split_conformal(model, calibration, alpha=alpha)

    with pytest.raises(ValueError, match="at least one row"):
        calibrate_split_conformal(model, calibration.iloc[0:0], alpha=0.10)
    with pytest.raises(ValueError, match="required target"):
        calibrate_split_conformal(
            model,
            calibration.drop(columns=TARGET_COLUMN),
            alpha=0.10,
        )


def test_relative_split_conformal_has_route_specific_valid_widths() -> None:
    frame = _traffic_frame()
    model = train_traffic_model("linear_regression", frame.iloc[:16])
    calibration = frame.iloc[16:28]
    relative = calibrate_relative_split_conformal(
        model, calibration, alpha=0.10, minimum_scale_minutes=1.0
    )
    evaluation_frame = frame.iloc[28:].copy()
    intervals = relative.predict(evaluation_frame)

    expected_scales = np.maximum(model.predict(evaluation_frame), 1.0)
    expected_widths = relative.calibration_quantile * expected_scales
    np.testing.assert_allclose(intervals["interval_half_width"], expected_widths)
    assert intervals["interval_half_width"].nunique() > 1
    assert relative.calibration_quantile >= 0.0
    evaluation = evaluate_conformal(relative, evaluation_frame)
    assert len(evaluation.intervals) == len(evaluation_frame)


def test_locally_scaled_conformal_uses_separate_feature_dependent_scale() -> None:
    frame = _traffic_frame(48)
    frame[TARGET_COLUMN] += 0.15 * np.square(np.arange(48) % 6)
    model = train_traffic_model("linear_regression", frame.iloc[:16])
    scale_model = train_residual_scale_model(
        model,
        frame.iloc[16:40],
        random_state=9,
        n_estimators=25,
        minimum_scale_minutes=0.1,
    )
    local = calibrate_locally_scaled_split_conformal(
        model, scale_model, frame.iloc[40:44], alpha=0.10
    )
    evaluation_frame = frame.iloc[44:]
    intervals = local.predict(evaluation_frame)
    expected = local.calibration_quantile * scale_model.predict_scale(evaluation_frame)
    np.testing.assert_allclose(intervals["interval_half_width"], expected)
    assert (scale_model.predict_scale(evaluation_frame) >= 0.1).all()
    assert len(evaluate_conformal(local, evaluation_frame).intervals) == 4
