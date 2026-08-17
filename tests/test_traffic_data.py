"""Tests for generic deterministic synthetic traffic observations."""

import networkx as nx
import numpy as np
import pandas as pd

from src.traffic_data import (
    IDENTITY_METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SPLIT_NAMES,
    TARGET_COLUMN,
    TRAFFIC_DATA_COLUMNS,
    generate_historical_trip_observations,
    prepare_model_data,
    split_traffic_observations,
)
from src.traffic_scenarios import SCENARIO_PRESETS, TrafficScenario


def _test_networks() -> tuple[dict[str, nx.DiGraph], dict[str, str]]:
    """Return two small canonical graphs with different physical structure."""

    grid = nx.DiGraph()
    grid.add_edge(
        "g0",
        "g1",
        distance_km=1.0,
        base_travel_time_min=2.0,
        speed_kph=30.0,
        road_class="residential",
        reliability=0.96,
    )
    grid.add_edge(
        "g1",
        "g2",
        distance_km=1.5,
        base_travel_time_min=2.5,
        road_class="secondary",
        reliability=0.90,
    )
    grid.add_edge(
        "g1",
        "g3",
        distance_km=0.8,
        base_travel_time_min=1.8,
        road_class="service",
    )

    radial = nx.DiGraph()
    radial.add_edge(
        "hub",
        "r1",
        distance_km=2.0,
        base_travel_time_min=3.0,
        road_class="primary",
        reliability=0.92,
    )
    radial.add_edge(
        "hub",
        "r2",
        distance_km=2.4,
        base_travel_time_min=3.8,
        road_class="tertiary",
        reliability=0.88,
    )
    radial.add_edge(
        "r1",
        "r2",
        distance_km=1.2,
        base_travel_time_min=2.4,
        road_class="residential",
    )

    networks = {"network_grid_01": grid, "network_radial_01": radial}
    topology_labels = {
        "network_grid_01": "grid",
        "network_radial_01": "radial",
    }
    return networks, topology_labels


def _traffic_data(random_seed: int = 31415) -> pd.DataFrame:
    networks, topology_labels = _test_networks()
    return generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=("normal", "rush", "rain", "shift"),
        observations_per_network_per_scenario=36,
        random_seed=random_seed,
    )


def test_traffic_generation_is_reproducible_and_mapping_order_independent() -> None:
    """The same seed and graphs must produce the same complete table."""

    networks, topology_labels = _test_networks()
    first = generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=("normal", "rush", "prediction_noise"),
        observations_per_network_per_scenario=24,
        random_seed=91,
    )
    reversed_networks = dict(reversed(list(networks.items())))
    second = generate_historical_trip_observations(
        reversed_networks,
        topology_labels,
        scenarios=("normal", "rush", "prediction_noise"),
        observations_per_network_per_scenario=24,
        random_seed=91,
    )

    pd.testing.assert_frame_equal(first, second)
    changed_seed = generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=("normal", "rush", "prediction_noise"),
        observations_per_network_per_scenario=24,
        random_seed=92,
    )
    assert not first.loc[:, MODEL_FEATURE_COLUMNS].equals(
        changed_seed.loc[:, MODEL_FEATURE_COLUMNS]
    )


def test_generated_schema_is_numeric_finite_and_location_independent() -> None:
    """Requested model columns are numeric and identities remain metadata."""

    traffic_data = _traffic_data()

    assert tuple(traffic_data.columns) == TRAFFIC_DATA_COLUMNS
    assert traffic_data["observation_id"].is_unique
    assert set(traffic_data["network_id"]) == {
        "network_grid_01",
        "network_radial_01",
    }
    assert set(traffic_data["topology"]) == {"grid", "radial"}
    assert set(traffic_data["scenario"]) == {"normal", "rush", "rain", "shift"}
    assert not set(IDENTITY_METADATA_COLUMNS).intersection(MODEL_FEATURE_COLUMNS)
    assert all(
        pd.api.types.is_numeric_dtype(traffic_data[column])
        for column in (*MODEL_FEATURE_COLUMNS, TARGET_COLUMN)
    )
    assert np.isfinite(
        traffic_data.loc[:, (*MODEL_FEATURE_COLUMNS, TARGET_COLUMN)].to_numpy(dtype=float)
    ).all()
    assert (traffic_data[TARGET_COLUMN] > 0.0).all()
    assert traffic_data["normalized_distance"].between(0.0, 1.0).all()
    for cyclic_column in ("hour_sin", "hour_cos", "day_sin", "day_cos"):
        assert traffic_data[cyclic_column].between(-1.0, 1.0).all()
    assert (traffic_data["route_edge_count"] > 1).mean() > 0.50
    assert traffic_data["route_reliability"].between(0.0, 1.0).all()

    features, target = prepare_model_data(traffic_data)
    assert tuple(features.columns) == MODEL_FEATURE_COLUMNS
    assert target.name == TARGET_COLUMN
    assert "network_id" not in features and "topology" not in features


def test_train_calibration_validation_test_splits_are_disjoint_and_deterministic() -> None:
    """Every ID occurs once and every network is represented in every split."""

    traffic_data = _traffic_data()
    first = split_traffic_observations(traffic_data, random_seed=808)
    second = split_traffic_observations(
        traffic_data.sample(frac=1.0, random_state=7),
        random_seed=808,
    )

    assert tuple(first) == SPLIT_NAMES
    for split_name in SPLIT_NAMES:
        pd.testing.assert_frame_equal(first[split_name], second[split_name])
        assert set(first[split_name]["network_id"]) == {
            "network_grid_01",
            "network_radial_01",
        }

    id_sets = {
        split_name: set(split["observation_id"])
        for split_name, split in first.items()
    }
    for first_index, first_name in enumerate(SPLIT_NAMES):
        for second_name in SPLIT_NAMES[first_index + 1 :]:
            assert id_sets[first_name].isdisjoint(id_sets[second_name])
    assert set().union(*id_sets.values()) == set(traffic_data["observation_id"])


def test_presets_cover_requested_events_and_closed_edges_have_no_targets() -> None:
    """All requested presets exist and fully closed observations are omitted."""

    assert {
        "normal",
        "rush",
        "rain",
        "congestion",
        "accident",
        "closure",
        "prediction_noise",
        "shift",
    }.issubset(SCENARIO_PRESETS)

    networks, topology_labels = _test_networks()
    always_closed = TrafficScenario(name="always_closed", closure_probability=1.0)
    traffic_data = generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=("normal", always_closed),
        observations_per_network_per_scenario=24,
        random_seed=123,
    )

    assert set(traffic_data["scenario"]) == {"normal"}
    assert np.isfinite(traffic_data[TARGET_COLUMN]).all()


def test_historical_delay_is_independent_of_current_outcome_noise() -> None:
    """Changing only target noise must not change the pre-trip lag feature."""

    networks, topology_labels = _test_networks()
    no_noise = TrafficScenario(name="lag_test", prediction_noise_fraction=0.0)
    high_noise = TrafficScenario(name="lag_test", prediction_noise_fraction=0.50)
    stable = generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=(no_noise,),
        observations_per_network_per_scenario=30,
        random_seed=456,
    )
    noisy = generate_historical_trip_observations(
        networks,
        topology_labels,
        scenarios=(high_noise,),
        observations_per_network_per_scenario=30,
        random_seed=456,
    )

    pd.testing.assert_series_equal(
        stable["historical_delay_minutes"],
        noisy["historical_delay_minutes"],
    )
    assert not stable[TARGET_COLUMN].equals(noisy[TARGET_COLUMN])
