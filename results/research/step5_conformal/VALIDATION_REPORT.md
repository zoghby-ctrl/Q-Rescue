# Step 5 validation report

## Overall assessment: Share with caveats

The manual split-conformal calculations are correct and traceable. An absolute
residual baseline, a prespecified prediction-scaled comparator, and a locally
scaled comparator are reported. Coverage is near but below nominal on the held-out ID test and fails
materially under the controlled traffic shift. This is a useful negative
result, not evidence that coverage is preserved out of distribution.

## Methodology review

- The point model was trained on Step 4 training rows and selected using only
  validation RMSE.
- Absolute residuals on the untouched 960-row calibration set are the
  nonconformity scores.
- For n calibration scores and error rate alpha, the finite-sample level is
  `min(1, ceil((n+1)(1-alpha))/n)`, evaluated with the empirical `higher`
  quantile.
- The interval is `[max(0, prediction-q), prediction+q]` because negative
  travel time is impossible. The saved half-width remains q; realized width
  can be smaller when the lower endpoint is clipped.
- The relative comparator uses score `|y-prediction|/max(prediction, 1 min)`.
  Its calibrated q is dimensionless and its route-specific half-width is
  `q*max(prediction, 1 min)`. The scaling rule is fixed before calibration.
- The local comparator fits an identity-free random-forest residual-scale model
  on the separate validation split, then calibrates
  `|y-prediction|/predicted_scale` on the untouched calibration split. Thus the
  scale learner never sees calibration or test outcomes.
- ID evaluation uses the untouched 720-row Step 4 test set. OOD-traffic
  evaluation uses 720 new trips from the preregistered `shift` scenario on the
  same five network structures.

## Calculation spot-checks

Coverage and mean width were independently recomputed from every row of
`intervals.csv`; they exactly match the aggregate tables.

| alpha | nominal | ID coverage | OOD-traffic coverage | q (min) |
|---:|---:|---:|---:|---:|
| 0.05 | 95% | 94.31% | 71.94% | 3.115 |
| 0.10 | 90% | 87.78% | 52.50% | 2.043 |
| 0.20 | 80% | 76.25% | 32.36% | 1.177 |

The relative-width comparator achieved ID/shift coverage of 94.44%/73.33%,
87.92%/45.56%, and 77.78%/25.42% at the same three alpha values. At alpha 0.10
its mean ID width was 3.926 minutes versus 4.082 for the absolute method. It is
retained because route-specific widths make the uncertainty term operational,
not because it solved distribution shift—it did not.

The local-scale comparator achieved ID/shift coverage of 94.58%/58.61%,
89.72%/45.83%, and 78.61%/32.64%. At alpha 0.10 its mean ID width was 3.643
minutes, the narrowest of the three methods while reaching 89.72% ID coverage.
This fixed-run observation motivates later repeated-seed comparison; it is not
yet a general superiority claim. Its feature-dependent widths are used by the
route-risk ablation because they can distinguish candidates with similar means.

At alpha 0.10, ID coverage ranges from 76.39% on sparse mixed to
97.22% on radial. Under traffic shift it ranges from 22.22% on sparse mixed to
94.44% on radial. These are conditional diagnostic rates; the global split
conformal procedure guarantees neither topology-conditional nor shifted
coverage.

## Issues and interpretation

1. Medium: both methods' ID rates are below their nominal levels in this fixed
   experiment. Finite-sample conformal validity is a repeated-sample marginal
   statement under exchangeability, not a promise that every realized test set
   exactly reaches nominal coverage.
2. High: traffic shift breaks exchangeability and causes severe undercoverage.
   These intervals must not be described as calibrated for the shifted domain.
3. Medium: topology groups have different residual scales. A single global q
   does not provide conditional coverage; group-conditional/Mondrian or adaptive
   methods are future comparators, not silently substituted here.

## Required caveats

- Coverage is empirical on controlled simulated trips.
- Interval width is not a probability.
- The interval concerns route travel time, not patient outcome or clinical risk.
- Global split conformal is not OOD-safe and is not topology-conditionally
  calibrated.
- Prediction scaling yields heterogeneous interval widths but does not make
  OOD coverage valid.
- The local scale model uses validation residuals; it must remain frozen before
  the calibration split is opened.
- No novelty is claimed from using conformal prediction.

## Decision

Proceed to Step 6 using the intervals exactly as measured. Preserve the OOD
failure as a benchmark result and never imply that nominal coverage survived
the shift.
