"""Understandable split-conformal intervals for travel-time prediction.

Absolute calibration residuals are the nonconformity scores.  Because travel
time cannot be negative, the raw lower endpoint ``prediction - q`` is clipped
at zero.  Consequently, a clipped interval can be narrower than ``2 * q``;
``interval_half_width`` still records the calibrated, pre-clipping value q.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor

from src.traffic_model import FEATURE_COLUMNS, TARGET_COLUMN, FittedTrafficModel


SUPPORTED_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)
INTERVAL_COLUMNS: tuple[str, ...] = (
    "prediction",
    "lower_bound",
    "upper_bound",
    "interval_half_width",
)
GROUP_COLUMNS: tuple[str, str] = ("network_id", "topology")


@dataclass(frozen=True)
class SplitConformalRegressor:
    """A fitted point model plus one calibrated residual quantile."""

    base_model: FittedTrafficModel
    alpha: float
    calibration_quantile: float
    quantile_level: float
    n_calibration: int

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return metadata, optional actuals, and conformal intervals."""

        return predict_intervals(self, data)


@dataclass(frozen=True)
class RelativeSplitConformalRegressor:
    """Split conformal using absolute residual divided by a fixed point scale.

    The scale ``max(prediction, minimum_scale_minutes)`` is fixed before seeing
    calibration outcomes.  The calibrated quantile is dimensionless, and each
    interval half-width is that ratio times the route's point-prediction scale.
    """

    base_model: FittedTrafficModel
    alpha: float
    calibration_quantile: float
    quantile_level: float
    n_calibration: int
    minimum_scale_minutes: float = 1.0

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        return predict_relative_intervals(self, data)


@dataclass(frozen=True)
class FittedResidualScaleModel:
    """Identity-free model of absolute point-prediction error in minutes."""

    estimator: RandomForestRegressor
    minimum_scale_minutes: float

    def predict_scale(self, data: pd.DataFrame) -> np.ndarray:
        predictions = np.asarray(
            self.estimator.predict(data.loc[:, FEATURE_COLUMNS]), dtype=float
        )
        return np.maximum(predictions, self.minimum_scale_minutes)


@dataclass(frozen=True)
class LocallyScaledSplitConformalRegressor:
    """Split conformal normalized by a pre-fitted feature-dependent scale."""

    base_model: FittedTrafficModel
    scale_model: FittedResidualScaleModel
    alpha: float
    calibration_quantile: float
    quantile_level: float
    n_calibration: int

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        return predict_locally_scaled_intervals(self, data)


@dataclass(frozen=True)
class CoverageMetrics:
    """Coverage and interval-width statistics for one population."""

    n_samples: int
    alpha: float
    nominal_coverage: float
    empirical_coverage: float
    mean_interval_width: float
    median_interval_width: float

    def as_dict(self) -> dict[str, int | float]:
        """Return stable names suitable for a result table."""

        return {
            "n_samples": self.n_samples,
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "empirical_coverage": self.empirical_coverage,
            "mean_interval_width": self.mean_interval_width,
            "median_interval_width": self.median_interval_width,
        }


@dataclass(frozen=True)
class ConformalEvaluation:
    """Intervals and their overall and group-level coverage tables."""

    overall: pd.DataFrame
    by_network: pd.DataFrame
    by_topology: pd.DataFrame
    intervals: pd.DataFrame


def _validated_alpha(alpha: float) -> float:
    try:
        value = float(alpha)
    except (TypeError, ValueError) as error:
        raise TypeError("alpha must be a number strictly between 0 and 1") from error
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    return value


def finite_sample_quantile_level(n_calibration: int, alpha: float) -> float:
    """Return min(1, ceil((n+1)(1-alpha))/n) for split conformal."""

    alpha_value = _validated_alpha(alpha)
    if n_calibration <= 0:
        raise ValueError("calibration data must contain at least one row")
    numerator = np.ceil((n_calibration + 1) * (1.0 - alpha_value))
    return float(min(1.0, numerator / n_calibration))


def finite_sample_residual_quantile(
    absolute_residuals: Sequence[float] | np.ndarray | pd.Series,
    alpha: float,
) -> float:
    """Compute the finite-sample-corrected empirical ``higher`` quantile."""

    scores = np.asarray(absolute_residuals, dtype=float)
    if scores.ndim != 1:
        raise ValueError("absolute residuals must be one-dimensional")
    if len(scores) == 0:
        raise ValueError("calibration data must contain at least one row")
    if not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise ValueError("absolute residuals must be finite and nonnegative")
    level = finite_sample_quantile_level(len(scores), alpha)
    return float(np.quantile(scores, level, method="higher"))


def _actual_values(data: pd.DataFrame, *, context: str) -> np.ndarray:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{context} must be a pandas DataFrame")
    if data.empty:
        raise ValueError(f"{context} must contain at least one row")
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"{context} is missing required target {TARGET_COLUMN!r}")
    if not is_numeric_dtype(data[TARGET_COLUMN]):
        raise TypeError(f"{TARGET_COLUMN!r} must be numeric")
    values = data[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{TARGET_COLUMN!r} must contain only finite values")
    return values


def calibrate_split_conformal(
    base_model: FittedTrafficModel,
    calibration_data: pd.DataFrame,
    *,
    alpha: float = 0.10,
) -> SplitConformalRegressor:
    """Calibrate q from labeled rows that were not used to train the model."""

    alpha_value = _validated_alpha(alpha)
    actual = _actual_values(calibration_data, context="calibration data")
    point_predictions = base_model.predict(calibration_data)
    scores = np.abs(actual - point_predictions)
    level = finite_sample_quantile_level(len(scores), alpha_value)
    quantile = finite_sample_residual_quantile(scores, alpha_value)
    return SplitConformalRegressor(
        base_model=base_model,
        alpha=alpha_value,
        calibration_quantile=quantile,
        quantile_level=level,
        n_calibration=len(scores),
    )


def calibrate_supported_alphas(
    base_model: FittedTrafficModel,
    calibration_data: pd.DataFrame,
    alphas: Sequence[float] = SUPPORTED_ALPHAS,
) -> dict[float, SplitConformalRegressor]:
    """Calibrate the standard 0.05, 0.10, and 0.20 levels, or supplied levels."""

    return {
        float(alpha): calibrate_split_conformal(
            base_model, calibration_data, alpha=alpha
        )
        for alpha in alphas
    }


def calibrate_relative_split_conformal(
    base_model: FittedTrafficModel,
    calibration_data: pd.DataFrame,
    *,
    alpha: float = 0.10,
    minimum_scale_minutes: float = 1.0,
) -> RelativeSplitConformalRegressor:
    """Calibrate dimensionless residual ratios on a held-out split."""

    alpha_value = _validated_alpha(alpha)
    scale_floor = float(minimum_scale_minutes)
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("minimum_scale_minutes must be finite and positive")
    actual = _actual_values(calibration_data, context="calibration data")
    predictions = base_model.predict(calibration_data)
    scales = np.maximum(np.abs(predictions), scale_floor)
    scores = np.abs(actual - predictions) / scales
    level = finite_sample_quantile_level(len(scores), alpha_value)
    quantile = finite_sample_residual_quantile(scores, alpha_value)
    return RelativeSplitConformalRegressor(
        base_model=base_model,
        alpha=alpha_value,
        calibration_quantile=quantile,
        quantile_level=level,
        n_calibration=len(scores),
        minimum_scale_minutes=scale_floor,
    )


def calibrate_relative_supported_alphas(
    base_model: FittedTrafficModel,
    calibration_data: pd.DataFrame,
    alphas: Sequence[float] = SUPPORTED_ALPHAS,
    *,
    minimum_scale_minutes: float = 1.0,
) -> dict[float, RelativeSplitConformalRegressor]:
    """Calibrate relative-width split conformal at several alpha values."""

    return {
        float(alpha): calibrate_relative_split_conformal(
            base_model,
            calibration_data,
            alpha=alpha,
            minimum_scale_minutes=minimum_scale_minutes,
        )
        for alpha in alphas
    }


def train_residual_scale_model(
    base_model: FittedTrafficModel,
    scale_training_data: pd.DataFrame,
    *,
    random_state: int = 2026,
    n_estimators: int = 200,
    minimum_scale_minutes: float = 0.25,
) -> FittedResidualScaleModel:
    """Fit error scale on labeled rows disjoint from conformal calibration.

    In the research pipeline these rows are the Step 4 validation split.  The
    network/topology metadata are excluded because the estimator receives only
    :data:`FEATURE_COLUMNS`.
    """

    actual = _actual_values(scale_training_data, context="scale training data")
    minimum_scale = float(minimum_scale_minutes)
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale_minutes must be finite and positive")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    absolute_residuals = np.abs(actual - base_model.predict(scale_training_data))
    estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=1,
    )
    estimator.fit(scale_training_data.loc[:, FEATURE_COLUMNS], absolute_residuals)
    return FittedResidualScaleModel(estimator, minimum_scale)


def calibrate_locally_scaled_split_conformal(
    base_model: FittedTrafficModel,
    scale_model: FittedResidualScaleModel,
    calibration_data: pd.DataFrame,
    *,
    alpha: float = 0.10,
) -> LocallyScaledSplitConformalRegressor:
    """Calibrate absolute-error/estimated-scale nonconformity scores."""

    alpha_value = _validated_alpha(alpha)
    actual = _actual_values(calibration_data, context="calibration data")
    predictions = base_model.predict(calibration_data)
    scales = scale_model.predict_scale(calibration_data)
    scores = np.abs(actual - predictions) / scales
    level = finite_sample_quantile_level(len(scores), alpha_value)
    quantile = finite_sample_residual_quantile(scores, alpha_value)
    return LocallyScaledSplitConformalRegressor(
        base_model=base_model,
        scale_model=scale_model,
        alpha=alpha_value,
        calibration_quantile=quantile,
        quantile_level=level,
        n_calibration=len(scores),
    )


def calibrate_locally_scaled_supported_alphas(
    base_model: FittedTrafficModel,
    scale_model: FittedResidualScaleModel,
    calibration_data: pd.DataFrame,
    alphas: Sequence[float] = SUPPORTED_ALPHAS,
) -> dict[float, LocallyScaledSplitConformalRegressor]:
    return {
        float(alpha): calibrate_locally_scaled_split_conformal(
            base_model, scale_model, calibration_data, alpha=alpha
        )
        for alpha in alphas
    }


def predict_intervals(
    conformal_model: SplitConformalRegressor,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Predict symmetric intervals, clipping their lower endpoint at zero.

    Any non-feature input columns are retained as metadata.  This includes
    ``travel_time_minutes`` when actual outcomes are available on evaluation or
    distribution-shift frames.
    """

    predictions = conformal_model.base_model.predict(data)
    q = conformal_model.calibration_quantile
    lower = np.maximum(0.0, predictions - q)
    upper = np.maximum(lower, predictions + q)

    metadata_columns = [
        column
        for column in data.columns
        if column not in FEATURE_COLUMNS and column not in INTERVAL_COLUMNS
    ]
    result = data.loc[:, metadata_columns].reset_index(drop=True).copy()
    result["prediction"] = predictions
    result["lower_bound"] = lower
    result["upper_bound"] = upper
    result["interval_half_width"] = q
    return result


def predict_relative_intervals(
    conformal_model: RelativeSplitConformalRegressor,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Return intervals whose valid split-conformal width scales by prediction."""

    predictions = conformal_model.base_model.predict(data)
    scales = np.maximum(
        np.abs(predictions), conformal_model.minimum_scale_minutes
    )
    half_widths = conformal_model.calibration_quantile * scales
    lower = np.maximum(0.0, predictions - half_widths)
    upper = np.maximum(lower, predictions + half_widths)
    metadata_columns = [
        column
        for column in data.columns
        if column not in FEATURE_COLUMNS and column not in INTERVAL_COLUMNS
    ]
    result = data.loc[:, metadata_columns].reset_index(drop=True).copy()
    result["prediction"] = predictions
    result["lower_bound"] = lower
    result["upper_bound"] = upper
    result["interval_half_width"] = half_widths
    return result


def predict_locally_scaled_intervals(
    conformal_model: LocallyScaledSplitConformalRegressor,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Return feature-dependent intervals using a separately trained scale."""

    predictions = conformal_model.base_model.predict(data)
    scales = conformal_model.scale_model.predict_scale(data)
    half_widths = conformal_model.calibration_quantile * scales
    lower = np.maximum(0.0, predictions - half_widths)
    upper = np.maximum(lower, predictions + half_widths)
    metadata_columns = [
        column
        for column in data.columns
        if column not in FEATURE_COLUMNS and column not in INTERVAL_COLUMNS
    ]
    result = data.loc[:, metadata_columns].reset_index(drop=True).copy()
    result["prediction"] = predictions
    result["lower_bound"] = lower
    result["upper_bound"] = upper
    result["interval_half_width"] = half_widths
    return result


def compute_coverage_metrics(
    actual: Sequence[float] | np.ndarray | pd.Series,
    lower_bound: Sequence[float] | np.ndarray | pd.Series,
    upper_bound: Sequence[float] | np.ndarray | pd.Series,
    *,
    alpha: float,
) -> CoverageMetrics:
    """Compute inclusive empirical coverage and realized interval widths."""

    alpha_value = _validated_alpha(alpha)
    actual_values = np.asarray(actual, dtype=float)
    lower_values = np.asarray(lower_bound, dtype=float)
    upper_values = np.asarray(upper_bound, dtype=float)
    if any(values.ndim != 1 for values in (actual_values, lower_values, upper_values)):
        raise ValueError("actual and interval bounds must be one-dimensional")
    if len(actual_values) == 0:
        raise ValueError("at least one interval is required")
    if actual_values.shape != lower_values.shape or actual_values.shape != upper_values.shape:
        raise ValueError("actual and interval bounds must have the same shape")
    if not all(
        np.isfinite(values).all()
        for values in (actual_values, lower_values, upper_values)
    ):
        raise ValueError("actual and interval bounds must be finite")
    if np.any(lower_values > upper_values):
        raise ValueError("lower bounds cannot exceed upper bounds")

    covered = (actual_values >= lower_values) & (actual_values <= upper_values)
    widths = upper_values - lower_values
    return CoverageMetrics(
        n_samples=len(actual_values),
        alpha=alpha_value,
        nominal_coverage=1.0 - alpha_value,
        empirical_coverage=float(np.mean(covered)),
        mean_interval_width=float(np.mean(widths)),
        median_interval_width=float(np.median(widths)),
    )


def _group_coverage_table(
    intervals: pd.DataFrame,
    *,
    group_column: str,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_value, group in intervals.groupby(group_column, sort=True, dropna=False):
        metrics = compute_coverage_metrics(
            group[TARGET_COLUMN],
            group["lower_bound"],
            group["upper_bound"],
            alpha=alpha,
        )
        rows.append({group_column: group_value, **metrics.as_dict()})
    return pd.DataFrame(
        rows,
        columns=[
            group_column,
            "n_samples",
            "alpha",
            "nominal_coverage",
            "empirical_coverage",
            "mean_interval_width",
            "median_interval_width",
        ],
    )


def evaluate_conformal(
    conformal_model: (
        SplitConformalRegressor
        | RelativeSplitConformalRegressor
        | LocallyScaledSplitConformalRegressor
    ),
    evaluation_data: pd.DataFrame,
) -> ConformalEvaluation:
    """Evaluate any labeled ID, OOD, or shifted frame without special cases."""

    _actual_values(evaluation_data, context="evaluation data")
    missing_groups = [
        column for column in GROUP_COLUMNS if column not in evaluation_data.columns
    ]
    if missing_groups:
        raise ValueError(
            f"evaluation data is missing grouping columns: {missing_groups}"
        )

    intervals = conformal_model.predict(evaluation_data)
    metrics = compute_coverage_metrics(
        intervals[TARGET_COLUMN],
        intervals["lower_bound"],
        intervals["upper_bound"],
        alpha=conformal_model.alpha,
    )
    return ConformalEvaluation(
        overall=pd.DataFrame([metrics.as_dict()]),
        by_network=_group_coverage_table(
            intervals, group_column="network_id", alpha=conformal_model.alpha
        ),
        by_topology=_group_coverage_table(
            intervals, group_column="topology", alpha=conformal_model.alpha
        ),
        intervals=intervals,
    )
