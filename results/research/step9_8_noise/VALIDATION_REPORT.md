# Validation Report — Step 9.8 Local Noise-Model Simulation

## Status

- Status: **PASS** for the targeted K=4 local density-matrix noise-model study.
- Source ideal QAOA runs: 40.
- Condition rows: 120 (ideal analytic, 4,096-shot noiseless, noisy shot).
- Independent problem instances: 10 across five topology families.
- Depths: p=1 and p=3; mixers: standard-X and feasible-XY.
- Angles are inherited from the ideal optimizer and are not reoptimized under noise.

## Results

|   depth | method      | condition             |   feasibility_probability_mean |   optimal_state_probability_mean |   optimal_probability_given_feasible_mean |   normalized_feasible_entropy_mean |   risk_regret_mean |
|--------:|:------------|:----------------------|-------------------------------:|---------------------------------:|------------------------------------------:|-----------------------------------:|-------------------:|
|       1 | feasible_xy | finite_shot_noiseless |                         1.0000 |                           0.3953 |                                    0.3953 |                             0.7617 |             0.3502 |
|       1 | feasible_xy | ideal_analytic        |                         1.0000 |                           0.3942 |                                    0.3942 |                             0.7614 |             0.3502 |
|       1 | feasible_xy | noisy_shot            |                         0.7506 |                           0.2876 |                                    0.3830 |                             0.8087 |             0.3502 |
|       1 | standard_x  | finite_shot_noiseless |                         0.3292 |                           0.0749 |                                    0.2211 |                             0.9546 |             4.9912 |
|       1 | standard_x  | ideal_analytic        |                         0.3320 |                           0.0762 |                                    0.2176 |                             0.9570 |             3.8702 |
|       1 | standard_x  | noisy_shot            |                         0.3335 |                           0.0761 |                                    0.2214 |                             0.9751 |             4.0022 |
|       3 | feasible_xy | finite_shot_noiseless |                         1.0000 |                           0.4805 |                                    0.4805 |                             0.6368 |             0.3135 |
|       3 | feasible_xy | ideal_analytic        |                         1.0000 |                           0.4815 |                                    0.4815 |                             0.6404 |             0.3135 |
|       3 | feasible_xy | noisy_shot            |                         0.5708 |                           0.2404 |                                    0.4206 |                             0.8173 |             0.3135 |
|       3 | standard_x  | finite_shot_noiseless |                         0.3031 |                           0.0588 |                                    0.1832 |                             0.8866 |             7.3081 |
|       3 | standard_x  | ideal_analytic        |                         0.3015 |                           0.0585 |                                    0.1833 |                             0.8911 |             6.5293 |
|       3 | standard_x  | noisy_shot            |                         0.3032 |                           0.0631 |                                    0.2029 |                             0.9378 |             6.5293 |

For XY, the noisy-shot minus ideal feasibility effects by depth are:

|   depth |   mean_difference |   ci95_lower |   ci95_upper |
|--------:|------------------:|-------------:|-------------:|
|  1.0000 |           -0.2494 |      -0.2526 |      -0.2460 |
|  3.0000 |           -0.4292 |      -0.4332 |      -0.4255 |

## Interpretation guardrails

1. This is a phenomenological local channel simulation based on declared gate-error rates, T1, block durations, and readout flips. It is not calibrated to an IBM fake backend and is not a hardware result.
2. Density-matrix evolution remains an exact local classical simulation; shot sampling adds finite-sampling variability.
3. Logical circuit resources come from the inherited ideal circuit. State-preparation and transpilation costs are not hardware resource estimates.
4. K=4 is classically trivial. No speedup, advantage, or deployment claim is supported.
5. The study tests whether the ideal feasibility pattern survives this declared model, not all plausible NISQ noise.
