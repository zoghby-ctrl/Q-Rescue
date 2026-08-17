# Validation Report — Standard-X Penalty Sensitivity

## Status

- Status: **PASS** for 350 K=8, p=3 standard-X trials.
- Independent problems: 50 across five topology families; validation seeds 0-2 (15 problems), test seeds 3-9 (35 problems).
- Seven frozen penalty conditions span below threshold, the exact zero/feasible tie, just above threshold, the repository's conservative default, and a 10× threshold.
- Every test condition is reported; no test row was used for tuning.
- The repository safe-default condition reproduces the accepted primary metrics with maximum absolute error 5.684e-14.
- Prior failed checkpoint attempts retained: 42. They contain no accepted metric because strict JSON serialization rejected undefined diagnostics before checkpoint creation; successful reruns use the unchanged frozen design.

## Validation-only selection

The frozen eligibility/selection rule selected `validated_safe_default`.

| penalty_condition      |   validation_problem_count | global_ground_states_all_feasible_every_problem   |   validation_mean_optimal_state_probability |   validation_mean_feasibility_probability |   median_ratio_to_exact_threshold | selected_by_frozen_validation_rule   |
|:-----------------------|---------------------------:|:--------------------------------------------------|--------------------------------------------:|------------------------------------------:|----------------------------------:|:-------------------------------------|
| half_threshold         |                         15 | False                                             |                                    0.012342 |                                  0.101234 |                          0.500000 | False                                |
| near_above_threshold   |                         15 | True                                              |                                    0.008543 |                                  0.082174 |                          1.050000 | False                                |
| near_below_threshold   |                         15 | False                                             |                                    0.019148 |                                  0.092217 |                          0.950000 | False                                |
| threshold_tie          |                         15 | False                                             |                                    0.009878 |                                  0.078566 |                          1.000000 | False                                |
| twice_threshold        |                         15 | True                                              |                                    0.005925 |                                  0.046041 |                          2.000000 | False                                |
| validated_safe_default |                         15 | True                                              |                                    0.009051 |                                  0.056399 |                          1.836304 | True                                 |
| very_large             |                         15 | True                                              |                                    0.004798 |                                  0.046002 |                         10.000000 | False                                |

## Complete held-out test grid

| penalty_condition      | metric                             |     mean |   ci95_lower |   ci95_upper |
|:-----------------------|:-----------------------------------|---------:|-------------:|-------------:|
| half_threshold         | feasibility_probability            | 0.075480 |     0.047103 |     0.109917 |
| half_threshold         | optimal_state_probability          | 0.009577 |     0.005654 |     0.014398 |
| half_threshold         | optimal_probability_given_feasible | 0.129806 |     0.106431 |     0.155242 |
| half_threshold         | normalized_feasible_entropy        | 0.887870 |     0.840307 |     0.930270 |
| near_above_threshold   | feasibility_probability            | 0.060295 |     0.037720 |     0.091246 |
| near_above_threshold   | optimal_state_probability          | 0.006821 |     0.003835 |     0.011209 |
| near_above_threshold   | optimal_probability_given_feasible | 0.117737 |     0.099584 |     0.136563 |
| near_above_threshold   | normalized_feasible_entropy        | 0.907847 |     0.876097 |     0.936932 |
| near_below_threshold   | feasibility_probability            | 0.091599 |     0.059543 |     0.130517 |
| near_below_threshold   | optimal_state_probability          | 0.010784 |     0.006264 |     0.016563 |
| near_below_threshold   | optimal_probability_given_feasible | 0.105133 |     0.083722 |     0.127834 |
| near_below_threshold   | normalized_feasible_entropy        | 0.893050 |     0.849395 |     0.934289 |
| threshold_tie          | feasibility_probability            | 0.075816 |     0.047288 |     0.110446 |
| threshold_tie          | optimal_state_probability          | 0.007820 |     0.004947 |     0.011665 |
| threshold_tie          | optimal_probability_given_feasible | 0.117850 |     0.091616 |     0.147143 |
| threshold_tie          | normalized_feasible_entropy        | 0.877109 |     0.842764 |     0.909674 |
| twice_threshold        | feasibility_probability            | 0.068122 |     0.041999 |     0.099226 |
| twice_threshold        | optimal_state_probability          | 0.011040 |     0.004225 |     0.021713 |
| twice_threshold        | optimal_probability_given_feasible | 0.119243 |     0.093517 |     0.154296 |
| twice_threshold        | normalized_feasible_entropy        | 0.884830 |     0.833262 |     0.928869 |
| validated_safe_default | feasibility_probability            | 0.051334 |     0.035789 |     0.071047 |
| validated_safe_default | optimal_state_probability          | 0.006646 |     0.004393 |     0.009458 |
| validated_safe_default | optimal_probability_given_feasible | 0.130082 |     0.100301 |     0.164839 |
| validated_safe_default | normalized_feasible_entropy        | 0.880025 |     0.841654 |     0.915455 |
| very_large             | feasibility_probability            | 0.036374 |     0.028288 |     0.045510 |
| very_large             | optimal_state_probability          | 0.003929 |     0.002830 |     0.005141 |
| very_large             | optimal_probability_given_feasible | 0.117128 |     0.090590 |     0.145939 |
| very_large             | normalized_feasible_entropy        | 0.886040 |     0.825505 |     0.936242 |

## Selected versus conservative default on held-out test networks

The validation-selected condition is the conservative repository default itself. A selected-minus-default contrast is therefore identically zero and is not assigned a redundant hypothesis test.

## Guardrails

1. Penalty changes the Hamiltonian scale as well as constraint separation. The optimizer seed, 15-step Adam budget, and step size are fixed across penalties, so scale sensitivity is part of the measured behavior.
2. Conditions below the exact threshold deliberately encode an invalid all-zero global ground state; their low feasibility is expected and is not evidence about a correctly encoded constraint.
3. The repository's `max(risk)+1` rule is sufficient but more conservative than the exact zero-state threshold `P > min(risk)` for nonnegative route risks.
4. K=8 is exactly enumerable and classically trivial. This sensitivity study addresses an encoding objection; it cannot support advantage or speedup.
