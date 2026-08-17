"""Generic, location-independent traffic scenario definitions.

The presets describe distributions used to create controlled synthetic traffic
observations.  They are data, not branches in the routing algorithms, so the
same generator can be used with every canonical road-network graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from collections.abc import Sequence


ALL_HOURS: Final[tuple[int, ...]] = tuple(range(24))


@dataclass(frozen=True, slots=True)
class TrafficScenario:
    """Parameters for one controlled synthetic traffic distribution.

    Delay fractions are relative to an edge's base travel time.  For example,
    ``rain_delay_fraction=0.20`` adds 20 percent when rain is sampled.
    ``prediction_noise_fraction`` is the standard deviation of zero-mean
    observation noise as a fraction of base travel time.  A closure removes
    that edge-time pair from the historical dataset rather than assigning an
    infinite or otherwise invalid regression target.
    """

    name: str
    hours: tuple[int, ...] = ALL_HOURS
    congestion_mean: float = 0.20
    congestion_std: float = 0.08
    rain_probability: float = 0.05
    incident_probability: float = 0.01
    closure_probability: float = 0.0
    congestion_delay_weight: float = 0.90
    rain_delay_fraction: float = 0.20
    incident_delay_fraction: float = 0.70
    prediction_noise_fraction: float = 0.03
    shift_delay_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid distributions when a preset or custom case is built."""

        if not self.name.strip():
            raise ValueError("A traffic scenario must have a non-empty name.")
        if not self.hours or any(not 0 <= hour <= 23 for hour in self.hours):
            raise ValueError("Scenario hours must be integers from 0 through 23.")

        probabilities = {
            "rain_probability": self.rain_probability,
            "incident_probability": self.incident_probability,
            "closure_probability": self.closure_probability,
        }
        for parameter_name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{parameter_name} must lie in [0, 1].")

        nonnegative_values = {
            "congestion_std": self.congestion_std,
            "congestion_delay_weight": self.congestion_delay_weight,
            "rain_delay_fraction": self.rain_delay_fraction,
            "incident_delay_fraction": self.incident_delay_fraction,
            "prediction_noise_fraction": self.prediction_noise_fraction,
            "shift_delay_fraction": self.shift_delay_fraction,
        }
        for parameter_name, value in nonnegative_values.items():
            if value < 0.0:
                raise ValueError(f"{parameter_name} must be non-negative.")
        if not 0.0 <= self.congestion_mean <= 1.0:
            raise ValueError("congestion_mean must lie in [0, 1].")


SCENARIO_PRESETS: Final[dict[str, TrafficScenario]] = {
    "normal": TrafficScenario(
        name="normal",
        congestion_mean=0.18,
        congestion_std=0.07,
        rain_probability=0.04,
        incident_probability=0.01,
    ),
    "rush": TrafficScenario(
        name="rush",
        hours=(7, 8, 9, 16, 17, 18, 19),
        congestion_mean=0.68,
        congestion_std=0.12,
        rain_probability=0.05,
        incident_probability=0.03,
    ),
    "rain": TrafficScenario(
        name="rain",
        congestion_mean=0.35,
        congestion_std=0.12,
        rain_probability=0.90,
        incident_probability=0.02,
        rain_delay_fraction=0.30,
        prediction_noise_fraction=0.05,
    ),
    "congestion": TrafficScenario(
        name="congestion",
        hours=tuple(range(6, 22)),
        congestion_mean=0.82,
        congestion_std=0.10,
        rain_probability=0.05,
        incident_probability=0.03,
        congestion_delay_weight=1.10,
    ),
    "accident": TrafficScenario(
        name="accident",
        congestion_mean=0.52,
        congestion_std=0.16,
        rain_probability=0.08,
        incident_probability=0.85,
        incident_delay_fraction=1.25,
        prediction_noise_fraction=0.08,
    ),
    "closure": TrafficScenario(
        name="closure",
        congestion_mean=0.45,
        congestion_std=0.15,
        rain_probability=0.08,
        incident_probability=0.12,
        closure_probability=0.30,
        incident_delay_fraction=0.90,
    ),
    "prediction_noise": TrafficScenario(
        name="prediction_noise",
        congestion_mean=0.35,
        congestion_std=0.18,
        rain_probability=0.12,
        incident_probability=0.05,
        prediction_noise_fraction=0.25,
    ),
    "shift": TrafficScenario(
        name="shift",
        hours=(6, 7, 8, 9, 15, 16, 17, 18, 19, 20),
        congestion_mean=0.76,
        congestion_std=0.17,
        rain_probability=0.40,
        incident_probability=0.16,
        congestion_delay_weight=1.15,
        rain_delay_fraction=0.38,
        incident_delay_fraction=1.10,
        prediction_noise_fraction=0.14,
        shift_delay_fraction=0.22,
    ),
}


# Distribution-shift and closure experiments are opt-in.  Keeping them out of
# the default historical sample prevents accidental contamination of a later
# in-distribution training benchmark.
DEFAULT_HISTORICAL_SCENARIOS: Final[tuple[str, ...]] = (
    "normal",
    "rush",
    "rain",
    "congestion",
    "accident",
    "prediction_noise",
)


def get_scenario(name: str) -> TrafficScenario:
    """Return a named preset with a clear error for unknown names."""

    try:
        return SCENARIO_PRESETS[name]
    except KeyError as error:
        available = ", ".join(sorted(SCENARIO_PRESETS))
        raise ValueError(f"Unknown scenario {name!r}. Available presets: {available}.") from error


def resolve_scenarios(
    scenarios: Sequence[str | TrafficScenario] | None = None,
) -> tuple[TrafficScenario, ...]:
    """Resolve preset names and custom definitions into unique scenarios."""

    requested = DEFAULT_HISTORICAL_SCENARIOS if scenarios is None else scenarios
    resolved = tuple(
        get_scenario(item) if isinstance(item, str) else item
        for item in requested
    )
    if not resolved:
        raise ValueError("At least one traffic scenario is required.")
    if any(not isinstance(item, TrafficScenario) for item in resolved):
        raise TypeError("Scenarios must be preset names or TrafficScenario objects.")

    scenario_names = [scenario.name for scenario in resolved]
    if len(scenario_names) != len(set(scenario_names)):
        raise ValueError("Scenario names must be unique within one generated dataset.")
    return resolved
