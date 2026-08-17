# Validation Report — Targeted K=16 Scaling

## Status

- Status: **PASS** for 72 targeted ideal-statevector K=16 trials.
- Independent networks: 9 (three topology families × three matched seeds).
- Uncertainty: low/high; depths: p=1/3; mixers: standard-X/feasible-XY.
- Every K=16 problem exactly reproduces its accepted K=8 first-route prefix, risk scores, realized outcomes, and deadline to 1e-9.
- Optimizer: the same 15-step Adam budget as the primary K=4/8 study.

## K=16 feasible-XY summary

|    depth |   feasibility_probability_mean |   optimal_state_probability_mean |   optimal_probability_given_feasible_mean |   normalized_feasible_entropy_mean |   risk_regret_mean |   wall_time_seconds_mean |
|---------:|-------------------------------:|---------------------------------:|------------------------------------------:|-----------------------------------:|-------------------:|-------------------------:|
| 1.000000 |                       1.000000 |                         0.104401 |                                  0.104401 |                           0.910302 |           2.053400 |                61.736384 |
| 3.000000 |                       1.000000 |                         0.122348 |                                  0.122348 |                           0.863407 |           2.105875 |               121.745936 |

## Matched K=16 minus K=8 effects

| metric                      |   mean_network_difference |   ci95_lower |   ci95_upper |   independent_network_count |
|:----------------------------|--------------------------:|-------------:|-------------:|----------------------------:|
| optimal_state_probability   |                 -0.141319 |    -0.163381 |    -0.118350 |                           9 |
| normalized_feasible_entropy |                  0.053340 |     0.031273 |     0.075969 |                           9 |

## Guardrails

1. K=16 has only 65,536 computational states and is classically trivial; exact candidate-risk minimization is immediate.
2. This is an indicative scaling trend across nine controlled synthetic networks, not a claim about useful quantum scale.
3. The statevector simulator and exact enumeration are classical computations. Runtime is reported only for reproducibility, never as a speed comparison.
4. The fixed optimizer budget may under-optimize the larger circuit, but it is held equal to the primary study instead of being tuned for a favorable K=16 result.
