"""Tests for deterministic, network-independent travel-time models."""

import numpy as np
import pandas as pd

from src.traffic_model import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    MODEL_NAMES,
    TARGET_COLUMN,
    compute_regression_metrics,
    create_linear_regression_model,
    evaluate_predictions,
    train_all_traffic_models,
    train_and_select_models,
)


def _make_traffic_frame(number_of_rows: int = 48) -> pd.DataFrame:
    row = np.arange(number_of_rows, dtype=float)
    hour = np.mod(row, 24.0)
    day = np.mod(row, 7.0)
    congestion = 0.15 + 0.7 * np.mod(row, 9.0) / 8.0
    rain = (np.mod(row, 5.0) == 0).astype(float)
    incident = (np.mod(row, 11.0) == 0).astype(float)
    base_time = 7.0 + np.mod(row, 13.0) * 0.8
    historical_delay = 1.0 + np.mod(row * 3.0, 10.0) * 0.35

    frame = pd.DataFrame(
        {
            "observation_id": [f"obs-{index:03d}" for index in range(number_of_rows)],
            "network_id": [f"network-{index % 2}" for index in range(number_of_rows)],
            "topology": ["grid" if index % 2 == 0 else "radial" for index in range(number_of_rows)],
            "scenario": ["normal" if index % 3 else "rush_hour" for index in range(number_of_rows)],
            "normalized_distance": 0.1 + np.mod(row, 10.0) / 10.0,
            "base_travel_time_min": base_time,
            "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            "day_sin": np.sin(2.0 * np.pi * day / 7.0),
            "day_cos": np.cos(2.0 * np.pi * day / 7.0),
            "congestion": congestion,
            "average_speed_kph": 55.0 - 24.0 * congestion,
            "rain_indicator": rain,
            "incident_indicator": incident,
            "road_class_code": np.mod(row, 4.0),
            "intersection_count": 2.0 + np.mod(row, 6.0),
            "historical_delay_minutes": historical_delay,
        }
    )
    frame[TARGET_COLUMN] = (
        base_time
        + 7.0 * congestion
        + 2.5 * rain
        + 5.0 * incident
        + 0.6 * historical_delay
        + 0.08 * frame["intersection_count"]
    )
    return frame


def test_feature_contract_excludes_identity_and_target_columns() -> None:
    """Network identity and evaluation labels must never enter X."""

    assert len(FEATURE_COLUMNS) == 13
    assert not set(IDENTITY_COLUMNS).intersection(FEATURE_COLUMNS)
    assert TARGET_COLUMN not in FEATURE_COLUMNS

    linear_model = create_linear_regression_model()
    assert list(linear_model.named_steps) == ["scaler", "regressor"]


def test_all_models_are_deterministic_and_ignore_identity_metadata() -> None:
    """Changing IDs/topology labels alone cannot change fitted predictions."""

    frame = _make_traffic_frame()
    training = frame.iloc[:32].copy()
    validation = frame.iloc[32:].copy()
    relabeled_training = training.copy()
    relabeled_validation = validation.copy()
    relabeled_training.loc[:, "observation_id"] = "changed-training-id"
    relabeled_training.loc[:, "network_id"] = "unseen-training-network"
    relabeled_training.loc[:, "topology"] = "changed-topology"
    relabeled_training.loc[:, "scenario"] = "changed-scenario"
    relabeled_validation.loc[:, "observation_id"] = "changed-validation-id"
    relabeled_validation.loc[:, "network_id"] = "unseen-validation-network"
    relabeled_validation.loc[:, "topology"] = "other-topology"
    relabeled_validation.loc[:, "scenario"] = "other-scenario"

    first_models = train_all_traffic_models(
        training, random_state=17, random_forest_trees=30
    )
    second_models = train_all_traffic_models(
        relabeled_training, random_state=17, random_forest_trees=30
    )

    assert tuple(first_models) == MODEL_NAMES
    for model_name in MODEL_NAMES:
        first_predictions = first_models[model_name].predict(validation)
        second_predictions = second_models[model_name].predict(relabeled_validation)
        assert first_predictions.shape == (len(validation),)
        np.testing.assert_allclose(first_predictions, second_predictions, rtol=0, atol=0)


def test_metric_values_and_evaluation_table_shapes() -> None:
    """Overall and grouped tables expose auditable sklearn metrics."""

    frame = _make_traffic_frame(6)
    frame[TARGET_COLUMN] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    predicted = np.asarray([1.0, 4.0, 2.0, 4.0, 7.0, 5.0])

    metrics = compute_regression_metrics(frame[TARGET_COLUMN], predicted)
    assert metrics.n_samples == 6
    assert np.isclose(metrics.mae, 1.0)
    assert np.isclose(metrics.rmse, np.sqrt(10.0 / 6.0))
    assert np.isclose(metrics.r2, 1.0 - 10.0 / 17.5)

    evaluation = evaluate_predictions(frame, predicted)
    assert evaluation.overall.shape == (1, 4)
    assert evaluation.by_network.shape == (2, 5)
    assert evaluation.by_topology.shape == (2, 5)
    assert evaluation.predictions.shape == (6, 9)
    assert evaluation.by_network["n_samples"].sum() == 6
    assert evaluation.by_topology["n_samples"].sum() == 6


def test_selection_uses_the_smallest_validation_rmse() -> None:
    """Model selection ranks all candidates using held-out validation rows."""

    frame = _make_traffic_frame()
    result = train_and_select_models(
        frame.iloc[:32],
        frame.iloc[32:],
        random_state=23,
        random_forest_trees=30,
    )

    assert result.ranking.shape == (3, 6)
    assert result.ranking["rank"].tolist() == [1, 2, 3]
    assert result.best_model_name == result.ranking.loc[0, "model_name"]
    assert result.best_model.name == result.best_model_name
    assert result.ranking["validation_rmse"].is_monotonic_increasing
    assert set(result.evaluations) == set(MODEL_NAMES)
    for evaluation in result.evaluations.values():
        assert len(evaluation.predictions) == 16
