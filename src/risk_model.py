"""Small deterministic risk-score functions for the prototype."""

import pandas as pd


def calculate_risk_score(
    expected_time_minutes: float,
    uncertainty_minutes: float,
    uncertainty_weight: float = 1.0,
) -> float:
    """Return expected time plus weighted uncertainty, measured in minutes."""

    if uncertainty_weight < 0:
        raise ValueError("uncertainty_weight must be non-negative")
    return expected_time_minutes + uncertainty_weight * uncertainty_minutes


def add_risk_scores(
    route_data: pd.DataFrame,
    uncertainty_weight: float = 1.0,
) -> pd.DataFrame:
    """Return a copy of the route table with a risk_score column."""

    scored_routes = route_data.copy()
    scored_routes["risk_score"] = scored_routes.apply(
        lambda row: calculate_risk_score(
            float(row["expected_time_minutes"]),
            float(row["uncertainty_minutes"]),
            uncertainty_weight,
        ),
        axis=1,
    )
    return scored_routes


def choose_expected_time_route(route_data: pd.DataFrame) -> str:
    """Return the name of the route with the lowest expected travel time."""

    best_row_index = route_data["expected_time_minutes"].idxmin()
    return str(route_data.loc[best_row_index, "route"])


def choose_risk_aware_route(scored_routes: pd.DataFrame) -> str:
    """Return the name of the route with the lowest risk score."""

    best_row_index = scored_routes["risk_score"].idxmin()
    return str(scored_routes.loc[best_row_index, "route"])

