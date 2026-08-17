# Validation Report — Step 9.7 Adaptive Conformal Baseline

## Status

- Status: **PASS** for the frozen ACI-style sequential baseline.
- Methods: absolute split conformal, locally scaled split conformal, and adaptive local ACI.
- Target coverage: 0.90; adaptation rate gamma: 0.005, fixed before evaluation.
- Point model, residual-scale model, and calibration-score pool remain frozen.
- Original pre-CSV Step 9 floating-point observations are deterministically regenerated from hash-verified sources; saved static intervals reproduce within tolerance.
- Each current interval is constructed before its current outcome updates the state.
- Maximum static-artifact reproduction error: 7.105e-15.

## Coverage

| evaluation_group   |   absolute_split |   adaptive_local_aci |   local_scaled_split |
|:-------------------|-----------------:|---------------------:|---------------------:|
| OOD-combined       |           0.5283 |               0.8460 |               0.5037 |
| OOD-topology       |           0.8646 |               0.8847 |               0.9033 |
| OOD-traffic        |           0.6347 |               0.8430 |               0.4747 |
| TEST-ID            |           0.9129 |               0.8884 |               0.9122 |

## Route-decision scope

The controlled route objective holds the point-prediction deadline term fixed
and changes only interval half-width between methods. Adaptive route feedback
uses only the selected adaptive route after the decision; unselected candidate
outcomes are not used to update future decisions. Across the four evaluation
groups, adaptive decisions cover 64
controlled route events. Detailed violation, width, route-change, and realized
regret metrics are saved in CSV form.

## Limitations

1. This is one ACI-style baseline, not an adaptive-conformal benchmark suite.
2. Test streams are controlled synthetic sequences in stable recorded order; order sensitivity is intrinsic to online adaptation.
3. The update changes nominal alpha but does not repair a misspecified point or scale model.
4. Route evaluation has four events per network (16 per evaluation group), so network-cluster intervals are exploratory.
5. Coverage under arbitrary distribution shift is not guaranteed or claimed.
