"""Deterministic synthetic historical trip observations and data splits."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Final

import networkx as nx
import numpy as np
import pandas as pd

from src.traffic_scenarios import TrafficScenario, resolve_scenarios


IDENTITY_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "observation_id",
    "network_id",
    "topology",
    "scenario",
)
ROUTE_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "origin",
    "destination",
    "route_nodes",
    "route_edge_count",
    "route_reliability",
    "route_distance_km",
    "distance_normalization_km",
)
TEMPORAL_METADATA_COLUMNS: Final[tuple[str, ...]] = ("hour_of_day", "day_of_week")
MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "normalized_distance",
    "base_travel_time_min",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "congestion",
    "average_speed_kph",
    "rain_indicator",
    "incident_indicator",
    "road_class_code",
    "intersection_count",
    "historical_delay_minutes",
)
TARGET_COLUMN: Final[str] = "travel_time_minutes"
TRAFFIC_DATA_COLUMNS: Final[tuple[str, ...]] = (
    *IDENTITY_METADATA_COLUMNS,
    *ROUTE_METADATA_COLUMNS,
    *TEMPORAL_METADATA_COLUMNS,
    *MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
SPLIT_NAMES: Final[tuple[str, ...]] = (
    "train",
    "calibration",
    "validation",
    "test",
)
DEFAULT_SPLIT_FRACTIONS: Final[dict[str, float]] = {
    "train": 0.50,
    "calibration": 0.20,
    "validation": 0.15,
    "test": 0.15,
}


ROAD_CLASS_CODES: Final[dict[str, int]] = {
    "unknown": 0,
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "residential": 6,
    "service": 7,
    "unclassified": 8,
    "other": 9,
    "local": 10,
    "collector": 11,
    "arterial": 12,
    "radial": 13,
    "ring": 14,
}


def _as_positive_float(value: object, field_name: str, edge: tuple[object, object]) -> float:
    """Convert a required edge value and reject non-finite/non-positive data."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Edge {edge!r} has a non-numeric {field_name}.") from error
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"Edge {edge!r} must have a positive finite {field_name}.")
    return numeric_value


def _validate_networks(
    networks: Mapping[str, nx.DiGraph],
    topology_labels: Mapping[str, str],
) -> None:
    """Validate only the canonical fields needed by the traffic generator."""

    if not networks:
        raise ValueError("At least one road network is required.")
    if set(networks) != set(topology_labels):
        raise ValueError("topology_labels must contain exactly one entry per network_id.")

    for network_id, graph in networks.items():
        if not isinstance(network_id, str) or not network_id.strip():
            raise ValueError("Every network_id must be a non-empty string.")
        if not isinstance(graph, nx.DiGraph) or not graph.is_directed():
            raise TypeError(f"Network {network_id!r} must be a NetworkX directed graph.")
        if graph.number_of_edges() == 0:
            raise ValueError(f"Network {network_id!r} has no edges.")
        if not str(topology_labels[network_id]).strip():
            raise ValueError(f"Network {network_id!r} needs a non-empty topology label.")

        for source, target, attributes in graph.edges(data=True):
            edge = (source, target)
            if "distance_km" not in attributes or "base_travel_time_min" not in attributes:
                raise ValueError(
                    f"Edge {edge!r} in {network_id!r} must define distance_km "
                    "and base_travel_time_min."
                )
            _as_positive_float(attributes["distance_km"], "distance_km", edge)
            _as_positive_float(
                attributes["base_travel_time_min"],
                "base_travel_time_min",
                edge,
            )
            if "speed_kph" in attributes and attributes["speed_kph"] is not None:
                _as_positive_float(attributes["speed_kph"], "speed_kph", edge)


def _road_class_code(value: object) -> int:
    """Map common road classes consistently across every generated network."""

    if isinstance(value, (list, tuple)):
        value = value[0] if value else "unknown"
    label = str(value if value is not None else "unknown").strip().lower()
    return ROAD_CLASS_CODES.get(label, ROAD_CLASS_CODES["other"])


def encode_road_class(value: object) -> int:
    """Public, network-independent road-class encoding used at inference."""

    return _road_class_code(value)


def _candidate_trip_paths(
    graph: nx.DiGraph,
    *,
    maximum_paths_per_od: int = 2,
    maximum_catalog_size: int = 512,
) -> list[tuple[object, ...]]:
    """Return weighted shortest simple paths for reachable OD pairs.

    The catalog is capped so this synthetic-data helper remains practical on
    moderate graphs.  Paths are deterministic because nodes and OD pairs are
    visited in a stable order.
    """

    stable_nodes = sorted(
        graph.nodes,
        key=lambda node: f"{type(node).__name__}:{node!r}",
    )
    paths: list[tuple[object, ...]] = []
    for origin in stable_nodes:
        for destination in stable_nodes:
            if origin == destination or not nx.has_path(graph, origin, destination):
                continue
            simple_paths = nx.shortest_simple_paths(
                graph,
                origin,
                destination,
                weight="base_travel_time_min",
            )
            for path_index, path in enumerate(simple_paths):
                if path_index >= maximum_paths_per_od:
                    break
                paths.append(tuple(path))
                if len(paths) >= maximum_catalog_size:
                    return paths
    return paths


def _route_attributes(
    graph: nx.DiGraph,
    path: Sequence[object],
) -> dict[str, object]:
    """Aggregate static edge attributes for one complete candidate trip."""

    edge_attributes = [
        graph.edges[source, target]
        for source, target in zip(path[:-1], path[1:], strict=True)
    ]
    distances = np.asarray(
        [
            _as_positive_float(attributes["distance_km"], "distance_km", edge)
            for edge, attributes in zip(
                zip(path[:-1], path[1:], strict=True),
                edge_attributes,
                strict=True,
            )
        ],
        dtype=float,
    )
    base_times = np.asarray(
        [
            _as_positive_float(
                attributes["base_travel_time_min"],
                "base_travel_time_min",
                edge,
            )
            for edge, attributes in zip(
                zip(path[:-1], path[1:], strict=True),
                edge_attributes,
                strict=True,
            )
        ],
        dtype=float,
    )
    reliabilities = np.asarray(
        [
            float(np.clip(float(attributes.get("reliability", 0.90)), 0.0, 1.0))
            for attributes in edge_attributes
        ],
        dtype=float,
    )
    road_class_codes = np.asarray(
        [_road_class_code(attributes.get("road_class", "unknown")) for attributes in edge_attributes],
        dtype=float,
    )

    return {
        "path": tuple(path),
        "edge_attributes": edge_attributes,
        "base_times": base_times,
        "reliabilities": reliabilities,
        "total_distance_km": float(distances.sum()),
        "total_base_time_min": float(base_times.sum()),
        "route_reliability": float(np.average(reliabilities, weights=base_times)),
        "road_class_code": float(np.average(road_class_codes, weights=base_times)),
        # Each internal node is an intersection traversed during the trip.  This
        # avoids double-counting a shared node once for each incident edge.
        "intersection_count": max(0, len(path) - 2),
    }


def generate_historical_trip_observations(
    networks: Mapping[str, nx.DiGraph],
    topology_labels: Mapping[str, str],
    *,
    scenarios: Sequence[str | TrafficScenario] | None = None,
    observations_per_network_per_scenario: int = 48,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Create generic historical OD-trip observations for multiple networks.

    Network identity and topology are retained strictly as metadata.  Neither
    appears in :data:`MODEL_FEATURE_COLUMNS`, and no city/network-specific
    coefficient is used to create the target.  Weighted shortest simple paths
    provide reasonable routes; their edge attributes and sampled dynamic
    conditions are aggregated to a total trip travel-time target.
    """

    _validate_networks(networks, topology_labels)
    resolved_scenarios = resolve_scenarios(scenarios)
    if observations_per_network_per_scenario <= 0:
        raise ValueError("observations_per_network_per_scenario must be positive.")

    route_catalogs: dict[str, list[dict[str, object]]] = {}
    for network_id in sorted(networks):
        graph = networks[network_id]
        paths = _candidate_trip_paths(graph)
        if not paths:
            raise ValueError(f"Network {network_id!r} has no reachable origin-destination pair.")
        route_catalogs[network_id] = [
            _route_attributes(graph, path)
            for path in paths
        ]
    maximum_distance = max(
        float(route["total_distance_km"])
        for catalog in route_catalogs.values()
        for route in catalog
    )
    rng = np.random.default_rng(random_seed)
    lagged_delay_by_route: dict[tuple[str, tuple[object, ...]], float] = {}
    rows: list[dict[str, object]] = []

    for network_id in sorted(networks):
        route_catalog = route_catalogs[network_id]
        multi_edge_routes = [route for route in route_catalog if len(route["path"]) > 2]
        for scenario in resolved_scenarios:
            for _ in range(observations_per_network_per_scenario):
                # Prefer multi-edge trips while retaining a small, realistic
                # possibility of adjacent-node journeys.
                preferred_catalog = (
                    multi_edge_routes
                    if multi_edge_routes and rng.random() < 0.85
                    else route_catalog
                )
                route = preferred_catalog[int(rng.integers(0, len(preferred_catalog)))]
                path = route["path"]
                edge_attributes = route["edge_attributes"]

                # If any segment is closed, the whole candidate trip is
                # unavailable and no invalid target is emitted.
                if any(
                    rng.random() < scenario.closure_probability
                    for _ in edge_attributes
                ):
                    continue

                route_key = (network_id, path)
                # Read the lag before sampling the current outcome.  The state
                # is updated below from noise-free latent delay, never from this
                # row's travel_time_minutes or observation noise.
                historical_delay = lagged_delay_by_route.get(route_key, 0.0)

                base_times = route["base_times"]
                reliabilities = route["reliabilities"]
                edge_congestion = np.clip(
                    rng.normal(
                        scenario.congestion_mean,
                        scenario.congestion_std,
                        size=len(edge_attributes),
                    ),
                    0.0,
                    1.0,
                )
                rain_indicator = int(rng.random() < scenario.rain_probability)
                edge_incidents = [
                    int(
                        rng.random()
                        < min(
                            1.0,
                            scenario.incident_probability
                            + 0.10 * (1.0 - reliability),
                        )
                    )
                    for reliability in reliabilities
                ]
                incident_indicator = int(any(edge_incidents))
                hour_of_day = int(rng.choice(scenario.hours))
                day_of_week = int(rng.integers(0, 7))
                hour_angle = 2.0 * np.pi * hour_of_day / 24.0
                day_angle = 2.0 * np.pi * day_of_week / 7.0

                edge_delay_multipliers = 1.0 + (
                    scenario.congestion_delay_weight * edge_congestion
                    + scenario.rain_delay_fraction * rain_indicator
                    + scenario.incident_delay_fraction * np.asarray(edge_incidents)
                    + scenario.shift_delay_fraction
                )
                latent_edge_times = base_times * edge_delay_multipliers
                latent_trip_time = float(latent_edge_times.sum())
                base_time = float(route["total_base_time_min"])
                distance_km = float(route["total_distance_km"])
                systematic_delay = max(0.0, latent_trip_time - base_time)
                congestion = float(np.average(edge_congestion, weights=base_times))
                # A pre-trip speed estimate may come from sensors or a traffic
                # service, but the completed trip time is not yet known.  Do
                # not calculate this feature as distance / latent_trip_time:
                # that would algebraically leak the outcome into the model.
                free_flow_route_speed = 60.0 * distance_km / base_time
                observable_speed_multiplier = 1.0 + (
                    0.72 * congestion
                    + 0.10 * rain_indicator
                    + 0.28 * incident_indicator
                )
                sensor_noise = rng.normal(0.0, 0.04 * free_flow_route_speed)
                average_speed_kph = max(
                    1.0,
                    free_flow_route_speed / observable_speed_multiplier
                    + sensor_noise,
                )

                observation_noise = rng.normal(
                    0.0,
                    base_time * scenario.prediction_noise_fraction,
                )
                travel_time = max(
                    0.05,
                    latent_trip_time + 0.15 * historical_delay + observation_noise,
                )
                lagged_delay_by_route[route_key] = (
                    0.65 * historical_delay + 0.35 * systematic_delay
                )

                rows.append(
                    {
                        "observation_id": f"obs_{len(rows):09d}",
                        "network_id": network_id,
                        "topology": str(topology_labels[network_id]),
                        "scenario": scenario.name,
                        "origin": str(path[0]),
                        "destination": str(path[-1]),
                        "route_nodes": " -> ".join(str(node) for node in path),
                        "route_edge_count": len(path) - 1,
                        "route_reliability": route["route_reliability"],
                        "route_distance_km": distance_km,
                        "distance_normalization_km": maximum_distance,
                        "normalized_distance": distance_km / maximum_distance,
                        "base_travel_time_min": base_time,
                        "hour_of_day": hour_of_day,
                        "day_of_week": day_of_week,
                        "hour_sin": float(np.sin(hour_angle)),
                        "hour_cos": float(np.cos(hour_angle)),
                        "day_sin": float(np.sin(day_angle)),
                        "day_cos": float(np.cos(day_angle)),
                        "congestion": congestion,
                        "average_speed_kph": average_speed_kph,
                        "rain_indicator": rain_indicator,
                        "incident_indicator": incident_indicator,
                        "road_class_code": route["road_class_code"],
                        "intersection_count": route["intersection_count"],
                        "historical_delay_minutes": historical_delay,
                        TARGET_COLUMN: travel_time,
                    }
                )

    traffic_data = pd.DataFrame(rows, columns=TRAFFIC_DATA_COLUMNS)
    if not traffic_data.empty:
        numeric_columns = (*MODEL_FEATURE_COLUMNS, TARGET_COLUMN)
        numeric_values = traffic_data.loc[:, numeric_columns].to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            raise RuntimeError("Generated traffic data unexpectedly contains non-finite values.")
    return traffic_data


def _validated_split_fractions(
    split_fractions: Mapping[str, float] | None,
) -> np.ndarray:
    """Return fractions in canonical split order after strict validation."""

    fractions = DEFAULT_SPLIT_FRACTIONS if split_fractions is None else split_fractions
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"split_fractions must define exactly: {', '.join(SPLIT_NAMES)}.")
    values = np.asarray([float(fractions[name]) for name in SPLIT_NAMES], dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Every split fraction must be positive and finite.")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError("Split fractions must sum to 1.0.")
    return values


def _allocate_split_counts(number_of_rows: int, fractions: np.ndarray) -> np.ndarray:
    """Allocate rows near requested ratios while keeping all four sets nonempty."""

    raw_counts = number_of_rows * fractions
    counts = np.floor(raw_counts).astype(int)
    remaining = number_of_rows - int(counts.sum())
    fractional_order = np.argsort(-(raw_counts - counts), kind="stable")
    for index in fractional_order[:remaining]:
        counts[index] += 1

    if number_of_rows >= len(SPLIT_NAMES):
        for empty_index in np.flatnonzero(counts == 0):
            donor_index = int(np.argmax(counts))
            if counts[donor_index] <= 1:
                raise RuntimeError("Could not create four nonempty split allocations.")
            counts[donor_index] -= 1
            counts[empty_index] += 1
    return counts


def _network_split_seed(random_seed: int, network_id: str) -> int:
    """Derive a stable per-network seed independent of mapping or row order."""

    digest = hashlib.sha256(f"{random_seed}:{network_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def split_traffic_observations(
    traffic_data: pd.DataFrame,
    *,
    split_fractions: Mapping[str, float] | None = None,
    random_seed: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Create deterministic, network-stratified train/calibration/validation/test sets.

    Observation IDs are the split unit.  An ID can occur in exactly one output,
    and every network is represented in every set.  The latter requirement
    needs at least four observations per network and is checked explicitly.
    """

    missing_columns = set(TRAFFIC_DATA_COLUMNS) - set(traffic_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Traffic data is missing required columns: {missing}.")
    if traffic_data.empty:
        raise ValueError("Cannot split an empty traffic dataset.")
    if traffic_data["observation_id"].isna().any():
        raise ValueError("observation_id cannot be missing.")
    if traffic_data["observation_id"].duplicated().any():
        raise ValueError("observation_id must be unique before splitting.")

    fractions = _validated_split_fractions(split_fractions)
    assignments: dict[str, list[pd.DataFrame]] = {name: [] for name in SPLIT_NAMES}

    for network_id, network_rows in traffic_data.groupby("network_id", sort=True):
        if len(network_rows) < len(SPLIT_NAMES):
            raise ValueError(
                f"Network {network_id!r} has {len(network_rows)} observations; "
                "at least four are required for per-network split coverage."
            )

        ordered_rows = network_rows.sort_values("observation_id", kind="stable")
        rng = np.random.default_rng(_network_split_seed(random_seed, str(network_id)))
        shuffled_positions = rng.permutation(len(ordered_rows))
        counts = _allocate_split_counts(len(ordered_rows), fractions)
        start = 0
        for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
            selected_positions = shuffled_positions[start : start + count]
            assignments[split_name].append(ordered_rows.iloc[selected_positions].copy())
            start += int(count)

    splits: dict[str, pd.DataFrame] = {}
    for split_name in SPLIT_NAMES:
        combined = pd.concat(assignments[split_name], ignore_index=True)
        splits[split_name] = combined.sort_values(
            ["network_id", "observation_id"], kind="stable"
        ).reset_index(drop=True)

    validate_split_separation(splits, expected_observation_ids=traffic_data["observation_id"])
    return splits


def validate_split_separation(
    splits: Mapping[str, pd.DataFrame],
    *,
    expected_observation_ids: Sequence[object] | pd.Series | None = None,
) -> None:
    """Raise if split names, ID uniqueness, disjointness, or coverage is wrong."""

    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"Splits must contain exactly: {', '.join(SPLIT_NAMES)}.")

    seen_ids: set[object] = set()
    for split_name in SPLIT_NAMES:
        split = splits[split_name]
        if "observation_id" not in split:
            raise ValueError(f"Split {split_name!r} has no observation_id column.")
        split_ids = set(split["observation_id"])
        if len(split_ids) != len(split):
            raise ValueError(f"Split {split_name!r} contains duplicate observation IDs.")
        overlap = seen_ids.intersection(split_ids)
        if overlap:
            raise ValueError(f"Observation IDs occur in multiple splits: {sorted(overlap)!r}.")
        seen_ids.update(split_ids)

    if expected_observation_ids is not None and seen_ids != set(expected_observation_ids):
        raise ValueError("Split IDs do not exactly cover the source traffic dataset.")

    all_networks = set().union(*(set(split["network_id"]) for split in splits.values()))
    for split_name, split in splits.items():
        missing_networks = all_networks - set(split["network_id"])
        if missing_networks:
            raise ValueError(
                f"Split {split_name!r} is missing networks: {sorted(missing_networks)!r}."
            )


def prepare_model_data(traffic_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric model features and target without identity metadata."""

    required_columns = set(MODEL_FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = required_columns - set(traffic_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Traffic data is missing model columns: {missing}.")

    features = traffic_data.loc[:, MODEL_FEATURE_COLUMNS].copy()
    target = traffic_data.loc[:, TARGET_COLUMN].copy()
    if any(not pd.api.types.is_numeric_dtype(features[column]) for column in features):
        raise TypeError("Every model feature must be numeric.")
    if not pd.api.types.is_numeric_dtype(target):
        raise TypeError("The travel-time target must be numeric.")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Model features and target cannot contain missing values.")
    return features, target


# A readable alternate name for callers that think in terms of a complete
# synthetic traffic table rather than individual historical observations.
generate_synthetic_traffic_data = generate_historical_trip_observations
