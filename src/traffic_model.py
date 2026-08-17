"""Deterministic travel-time models and evaluation tables for Step 4.

Network, topology, scenario, and observation identifiers are deliberately
excluded from :data:`FEATURE_COLUMNS`.  They are retained only as metadata for
traceable predictions and group-level evaluation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS: tuple[str, ...] = (
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
TARGET_COLUMN = "travel_time_minutes"
IDENTITY_COLUMNS: tuple[str, ...] = (
    "observation_id",
    "network_id",
    "topology",
    "scenario",
)
EVALUATION_GROUP_COLUMNS: tuple[str, str] = ("network_id", "topology")
MODEL_NAMES: tuple[str, ...] = (
    "linear_regression",
    "random_forest",
    "hist_gradient_boosting",
)
DEFAULT_RANDOM_SEED = 2026


TrafficEstimator = (
    Pipeline | RandomForestRegressor | HistGradientBoostingRegressor
)


@dataclass(frozen=True)
class RegressionMetrics:
    """Scalar regression metrics for one evaluation population."""

    n_samples: int
    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> dict[str, int | float]:
        """Return a table-ready representation with stable column names."""

        return {
            "n_samples": self.n_samples,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
        }


@dataclass
class FittedTrafficModel:
    """A fitted estimator together with its explicit data contract."""

    name: str
    estimator: TrafficEstimator
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    target_column: str = TARGET_COLUMN

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict travel time using feature columns only."""

        return predict_travel_times(self, data)


@dataclass(frozen=True)
class RegressionEvaluation:
    """Overall, grouped, and row-level validation results."""

    overall: pd.DataFrame
    by_network: pd.DataFrame
    by_topology: pd.DataFrame
    predictions: pd.DataFrame


@dataclass(frozen=True)
class ModelSelectionResult:
    """Fitted candidates ranked by validation RMSE."""

    best_model_name: str
    best_model: FittedTrafficModel
    ranking: pd.DataFrame
    evaluations: Mapping[str, RegressionEvaluation]


def create_linear_regression_model() -> Pipeline:
    """Create scaled ordinary least squares with no identity features."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def create_random_forest_model(
    *,
    random_state: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 200,
) -> RandomForestRegressor:
    """Create a reproducible random-forest travel-time regressor."""

    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )


def create_hist_gradient_boosting_model(
    *,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> HistGradientBoostingRegressor:
    """Create a reproducible histogram gradient-boosting regressor."""

    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=5,
        random_state=random_state,
    )


def create_model_catalog(
    *,
    random_state: int = DEFAULT_RANDOM_SEED,
    random_forest_trees: int = 200,
) -> dict[str, TrafficEstimator]:
    """Construct the three Step 4 model candidates in stable order."""

    return {
        "linear_regression": create_linear_regression_model(),
        "random_forest": create_random_forest_model(
            random_state=random_state,
            n_estimators=random_forest_trees,
        ),
        "hist_gradient_boosting": create_hist_gradient_boosting_model(
            random_state=random_state
        ),
    }


def _validate_numeric_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{context} must be a pandas DataFrame")
    if data.empty:
        raise ValueError(f"{context} must contain at least one row")

    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")

    non_numeric = [column for column in columns if not is_numeric_dtype(data[column])]
    if non_numeric:
        raise TypeError(f"{context} columns must be numeric: {non_numeric}")

    values = data.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains missing or non-finite numeric values")


def _feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    _validate_numeric_columns(data, FEATURE_COLUMNS, context="traffic data")
    return data.loc[:, FEATURE_COLUMNS]


def train_traffic_model(
    model_name: str,
    training_data: pd.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_SEED,
    random_forest_trees: int = 200,
) -> FittedTrafficModel:
    """Construct and fit one named model on the explicit feature list."""

    catalog = create_model_catalog(
        random_state=random_state,
        random_forest_trees=random_forest_trees,
    )
    if model_name not in catalog:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of {tuple(catalog)}"
        )

    _validate_numeric_columns(
        training_data,
        (*FEATURE_COLUMNS, TARGET_COLUMN),
        context="training data",
    )
    estimator = catalog[model_name]
    estimator.fit(
        training_data.loc[:, FEATURE_COLUMNS],
        training_data.loc[:, TARGET_COLUMN],
    )
    return FittedTrafficModel(name=model_name, estimator=estimator)


def train_all_traffic_models(
    training_data: pd.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_SEED,
    random_forest_trees: int = 200,
) -> dict[str, FittedTrafficModel]:
    """Fit every Step 4 candidate with the same training rows and seed."""

    return {
        model_name: train_traffic_model(
            model_name,
            training_data,
            random_state=random_state,
            random_forest_trees=random_forest_trees,
        )
        for model_name in MODEL_NAMES
    }


def predict_travel_times(
    model: FittedTrafficModel,
    data: pd.DataFrame,
) -> np.ndarray:
    """Return one travel-time prediction per input row."""

    features = _feature_matrix(data)
    predictions = np.asarray(model.estimator.predict(features), dtype=float)
    if predictions.shape != (len(data),):
        raise ValueError("Estimator predictions must be one-dimensional")
    if not np.isfinite(predictions).all():
        raise ValueError("Estimator produced missing or non-finite predictions")
    return predictions


def compute_regression_metrics(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
) -> RegressionMetrics:
    """Compute MAE, RMSE, and R2 with explicit single-row behavior."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("actual and predicted values must be one-dimensional")
    if len(actual_values) == 0:
        raise ValueError("at least one prediction is required")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted values must have the same shape")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("actual and predicted values must be finite")

    r2 = float("nan")
    if len(actual_values) >= 2:
        r2 = float(r2_score(actual_values, predicted_values))
    return RegressionMetrics(
        n_samples=len(actual_values),
        mae=float(mean_absolute_error(actual_values, predicted_values)),
        rmse=float(root_mean_squared_error(actual_values, predicted_values)),
        r2=r2,
    )


def _group_metric_table(
    data: pd.DataFrame,
    predicted: np.ndarray,
    group_column: str,
) -> pd.DataFrame:
    grouped_values = pd.DataFrame(
        {
            group_column: data[group_column].to_numpy(),
            "actual": data[TARGET_COLUMN].to_numpy(dtype=float),
            "predicted": predicted,
        }
    )
    rows: list[dict[str, object]] = []
    for group_value, group in grouped_values.groupby(
        group_column, sort=True, dropna=False
    ):
        metrics = compute_regression_metrics(group["actual"], group["predicted"])
        rows.append({group_column: group_value, **metrics.as_dict()})
    return pd.DataFrame(
        rows,
        columns=[group_column, "n_samples", "mae", "rmse", "r2"],
    )


def evaluate_predictions(
    data: pd.DataFrame,
    predicted: Sequence[float] | np.ndarray | pd.Series,
) -> RegressionEvaluation:
    """Build overall, per-network, per-topology, and row-level tables."""

    _validate_numeric_columns(data, (TARGET_COLUMN,), context="evaluation data")
    missing_groups = [
        column for column in EVALUATION_GROUP_COLUMNS if column not in data.columns
    ]
    if missing_groups:
        raise ValueError(
            f"evaluation data is missing grouping columns: {missing_groups}"
        )

    predicted_values = np.asarray(predicted, dtype=float)
    metrics = compute_regression_metrics(data[TARGET_COLUMN], predicted_values)
    overall = pd.DataFrame([metrics.as_dict()])

    trace_columns = [
        column for column in IDENTITY_COLUMNS if column in data.columns
    ]
    prediction_table = data.loc[:, trace_columns].reset_index(drop=True).copy()
    prediction_table["actual_travel_time_minutes"] = data[
        TARGET_COLUMN
    ].to_numpy(dtype=float)
    prediction_table["predicted_travel_time_minutes"] = predicted_values
    prediction_table["residual_minutes"] = (
        prediction_table["actual_travel_time_minutes"]
        - prediction_table["predicted_travel_time_minutes"]
    )
    prediction_table["absolute_error_minutes"] = prediction_table[
        "residual_minutes"
    ].abs()
    prediction_table["squared_error_minutes2"] = prediction_table[
        "residual_minutes"
    ].pow(2)

    return RegressionEvaluation(
        overall=overall,
        by_network=_group_metric_table(data, predicted_values, "network_id"),
        by_topology=_group_metric_table(data, predicted_values, "topology"),
        predictions=prediction_table,
    )


def evaluate_traffic_model(
    model: FittedTrafficModel,
    data: pd.DataFrame,
) -> RegressionEvaluation:
    """Predict and evaluate one fitted model on a labeled data frame."""

    return evaluate_predictions(data, model.predict(data))


def select_best_by_validation_rmse(
    fitted_models: Mapping[str, FittedTrafficModel],
    validation_data: pd.DataFrame,
) -> ModelSelectionResult:
    """Evaluate candidates and select the lowest validation RMSE."""

    if not fitted_models:
        raise ValueError("at least one fitted model is required")

    evaluations: dict[str, RegressionEvaluation] = {}
    ranking_rows: list[dict[str, object]] = []
    for model_name, model in fitted_models.items():
        evaluation = evaluate_traffic_model(model, validation_data)
        evaluations[model_name] = evaluation
        metrics = evaluation.overall.iloc[0]
        ranking_rows.append(
            {
                "model_name": model_name,
                "n_validation_samples": int(metrics["n_samples"]),
                "validation_mae": float(metrics["mae"]),
                "validation_rmse": float(metrics["rmse"]),
                "validation_r2": float(metrics["r2"]),
            }
        )

    ranking = (
        pd.DataFrame(ranking_rows)
        .sort_values(
            ["validation_rmse", "model_name"],
            ascending=[True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    best_model_name = str(ranking.loc[0, "model_name"])
    return ModelSelectionResult(
        best_model_name=best_model_name,
        best_model=fitted_models[best_model_name],
        ranking=ranking,
        evaluations=evaluations,
    )


def train_and_select_models(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_SEED,
    random_forest_trees: int = 200,
) -> ModelSelectionResult:
    """Fit all candidates and select one using validation data only."""

    fitted_models = train_all_traffic_models(
        training_data,
        random_state=random_state,
        random_forest_trees=random_forest_trees,
    )
    return select_best_by_validation_rmse(fitted_models, validation_data)
