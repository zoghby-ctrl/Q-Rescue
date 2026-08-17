# Step 4 validation report

## Overall assessment: Share with caveats

The final Step 4 run is internally consistent and suitable as a controlled
synthetic baseline for Step 5. It is not evidence of real-world accuracy or
cross-network generalization because every topology appears in training and
the traffic outcomes are generated rather than observed.

## Methodology review

- Question: can the same identity-free feature contract predict complete trip
  travel time across five synthetic topology families under six traffic
  scenarios?
- Grain: one complete origin-destination trip per row.
- Splits: 50% train, 20% calibration, 15% validation, 15% test, stratified by
  network with observation IDs disjoint across all four sets.
- Model selection: lowest validation RMSE; the test set was not used to choose
  the model.
- Models: linear regression, random forest, and histogram gradient boosting.
- Metrics: MAE, RMSE, and R-squared overall and separately by network/topology.

## Data-quality checks

- 4,800 rows and 4,800 unique observation IDs; zero duplicates.
- Zero missing values; every model/target value is finite.
- Five networks, five topology labels, and six scenarios are present.
- Every split contains all five networks.
- All six pairwise split-overlap counts are zero, and split IDs exactly cover
  the source table.
- 98.65% of rows are multi-edge trips; paths range from 1 to 13 edges.
- `network_id`, `topology`, `scenario`, and `observation_id` are metadata only
  and do not occur in the 13-column model feature contract.

## Issue found and fixed

An initial discarded run computed `average_speed_kph` directly as route
distance divided by the latent travel time used to construct the outcome. That
was a high-severity target-leakage risk and produced artificially optimistic
test RMSE near 0.83 minutes. The generator now creates a noisy pre-trip sensor
estimate from free-flow speed and observable congestion/rain/incident state.
The benchmark was regenerated from scratch; the accepted best-model test RMSE
is 1.713 minutes. No number from the discarded run is used as a result.

## Calculation spot-checks

- Best validation model: histogram gradient boosting, validation RMSE 1.678
  minutes (from the saved selection table).
- Accepted test metrics independently recomputed from row-level predictions:
  MAE 0.904292, RMSE 1.713140, R-squared 0.940970 over 720 rows.
- These independently recomputed values exactly match
  `model_metrics_overall.csv`.

## Required caveats

- This is controlled simulated traffic, not live or historical city traffic.
- The generator and model share observable variables by design; good ID scores
  do not prove OOD transfer.
- Each topology currently has one network instance in this Step 4 run.
- Random row splitting does not test unseen topology or temporal shift; those
  are reserved for Steps 9 and 9.1.
- R-squared should not be compared across topology groups without considering
  their different target variance.

## Decision

Proceed to split-conformal calibration using the untouched calibration split.
Do not train or tune on calibration or test rows.
