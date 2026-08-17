# Step 9.4 Robustness and Ablation Validation Report

## Outcome

The package reconciles inherited raw artifacts with targeted missing cells. Static
deadline results are explicitly proxies; dynamic results are realized outcomes in a
controlled simulator, not real traffic. Ideal statevector QAOA probabilities are not
hardware results. No runtime or speed comparison is made.

All validation checks passed: **True**.

## Isolated static risk removals

Deadlines are held fixed within each network/K instance. `without_conformal_calibration`
uses degenerate point intervals and therefore removes both calibrated width and the
conservative conformal upper-bound proxy; it is not conformal inference.

| condition | decision_count | selection_change_count_from_full | deadline_proxy_violation_rate | mean_selected_deadline_margin_minutes |
| --- | --- | --- | --- | --- |
| full | 15 | 0 | 0.0000 | 2.0150 |
| without_conformal_calibration | 15 | 0 | 0.0000 | 4.2983 |
| without_deadline_penalty | 15 | 0 | 0.0000 | 2.0150 |
| without_reroute_penalty | 15 | 3 | 0.0000 | 2.0729 |
| without_uncertainty | 15 | 0 | 0.0000 | 2.0150 |

The inherited A-E table is retained separately because several legacy variants remove
multiple terms at once and are not a one-term causal isolation.

## Alpha and coverage robustness

Alpha selection effects use the same candidate states and keep the original alpha=0.10
deadline fixed so interval width is the changed factor.

| alpha | decision_count | selection_change_count_from_alpha_0_10 | deadline_proxy_violation_rate | mean_selected_interval_half_width |
| --- | --- | --- | --- | --- |
| 0.0500 | 15.0000 | 0.0000 | 0.1333 | 2.9920 |
| 0.1000 | 15.0000 | 0.0000 | 0.0000 | 2.2834 |
| 0.2000 | 15.0000 | 0.0000 | 0.0000 | 1.5732 |

Observed coverage failure regions use the strict descriptive rule that the nominal
coverage lies above the empirical Wilson 95% upper bound. Source populations are never
pooled.

| source_scope | evaluation_set | method | alpha | nominal_coverage | empirical_coverage | wilson_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| step5_original_synthetic_networks | test_ood_traffic | absolute | 0.0500 | 0.9500 | 0.7194 | 0.7510 |
| step5_original_synthetic_networks | test_id | absolute | 0.1000 | 0.9000 | 0.8778 | 0.8997 |
| step5_original_synthetic_networks | test_ood_traffic | absolute | 0.1000 | 0.9000 | 0.5250 | 0.5612 |
| step5_original_synthetic_networks | test_id | absolute | 0.2000 | 0.8000 | 0.7625 | 0.7921 |
| step5_original_synthetic_networks | test_ood_traffic | absolute | 0.2000 | 0.8000 | 0.3236 | 0.3586 |
| step5_original_synthetic_networks | test_ood_traffic | relative_scaled | 0.0500 | 0.9500 | 0.7333 | 0.7643 |
| step5_original_synthetic_networks | test_ood_traffic | relative_scaled | 0.1000 | 0.9000 | 0.4556 | 0.4921 |
| step5_original_synthetic_networks | test_ood_traffic | relative_scaled | 0.2000 | 0.8000 | 0.2542 | 0.2872 |
| step5_original_synthetic_networks | test_ood_traffic | local_scaled | 0.0500 | 0.9500 | 0.5861 | 0.6215 |
| step5_original_synthetic_networks | test_ood_traffic | local_scaled | 0.1000 | 0.9000 | 0.4583 | 0.4949 |
| step5_original_synthetic_networks | test_ood_traffic | local_scaled | 0.2000 | 0.8000 | 0.3264 | 0.3615 |
| step8_5_real_road_topology_controlled_simulated_traffic | real_topology_controlled_simulated_traffic | local_scaled | 0.1000 | 0.9000 | 0.5443 | 0.5934 |

## Prediction noise

For the frozen Step 4 best model, the held-out `prediction_noise` scenario has MAE
**1.5138 min** and RMSE
**2.4444 min**. In the separate repeated-topology
scope, alpha=0.10 local intervals cover **0.7925**
of prediction-noise rows (n=53). These scopes are
reported separately and are not pooled.

## Risk-weight failure regions

The sweep is one-at-a-time around the baseline (1, 2, 1); it is not a full factorial
interaction study. A row is retained here when selections changed or at least one
selected route violated the static deadline proxy.

| sweep_parameter | sweep_value | selection_change_count_from_baseline_weights | deadline_proxy_violation_rate | mean_selected_deadline_margin_minutes |
| --- | --- | --- | --- | --- |
| lambda_reroute | 0.0000 | 3 | 0.0000 | 2.0729 |

## QUBO penalty strength

Unsafe penalties are evaluated directly rather than passed through the safe Step 7
constructor. Ground-state ties count as a failure of guaranteed validity whenever any
invalid assignment shares the minimum.

| K | penalty_setting | instance_count | invalid_ground_state_instance_count | mean_ground_state_valid_fraction | zero_state_ground_instance_count |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.25x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 4 | 0.5x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 4 | 0.75x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 4 | 0.99x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 4 | 0x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 4 | 1x_minimum_valid_cost | 5 | 5 | 0.5000 | 5 |
| 6 | 0.25x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 6 | 0.5x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 6 | 0.75x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 6 | 0.99x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 6 | 0x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 6 | 1x_minimum_valid_cost | 5 | 5 | 0.5000 | 5 |
| 8 | 0.25x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 8 | 0.5x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 8 | 0.75x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 8 | 0.99x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 8 | 0x_minimum_valid_cost | 5 | 5 | 0.0000 | 5 |
| 8 | 1x_minimum_valid_cost | 5 | 5 | 0.5000 | 5 |

The `safe_maximum_cost_plus_margin` reference preserves valid exact ground states for
all 15 network/K instances. The sweep also shows
where the sufficient Step 7 rule is conservative versus where low penalties actually
fail.

## QAOA method, K, and p

Inherited Step 7.5 contains 106 runs. Eight targeted runs add the missing p=3 cells for
K=6/8 on the same single grid scaling instance and seeds. K=4 remains a separate
five-topology scope. Valid and optimal probabilities are analytic ideal-simulator
probabilities.

| K | depth | method | run_count | mean_valid_probability | mean_optimal_probability | most_probable_valid_optimum_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 1 | feasible_xy | 2 | 1.0000 | 0.2068 | 0.0000 |
| 6 | 1 | standard_x | 2 | 0.1073 | 0.0210 | 1.0000 |
| 6 | 2 | feasible_xy | 2 | 1.0000 | 0.3307 | 0.5000 |
| 6 | 2 | standard_x | 2 | 0.1911 | 0.0252 | 0.0000 |
| 6 | 3 | feasible_xy | 2 | 1.0000 | 0.3142 | 0.5000 |
| 6 | 3 | standard_x | 2 | 0.0363 | 0.0013 | 0.0000 |
| 8 | 1 | feasible_xy | 2 | 1.0000 | 0.2194 | 0.0000 |
| 8 | 1 | standard_x | 2 | 0.0265 | 0.0074 | 0.5000 |
| 8 | 2 | feasible_xy | 2 | 1.0000 | 0.2678 | 1.0000 |
| 8 | 2 | standard_x | 2 | 0.0414 | 0.0058 | 0.5000 |
| 8 | 3 | feasible_xy | 2 | 1.0000 | 0.2936 | 1.0000 |
| 8 | 3 | standard_x | 2 | 0.0229 | 0.0020 | 0.0000 |

Feasible-XY preserved one-hot validity across all inherited and targeted runs:
**True**. Standard-X feasibility leakage and
most-probable selection misses are listed in `qaoa_failure_regions.csv`.

Warm start is explicitly **not implemented and not run**. A fabricated comparison from
cold-start angles would be invalid.

## Realized controlled dynamic outcomes

The targeted matrix contains 175 paired classical
simulations: five ablations by five synthetic networks by seven event rotations. Each
condition uses the same per-case random seed; paths may diverge after a selection change.

| condition | run_count | arrival_rate | realized_deadline_violation_rate | mean_reroutes | initial_selection_change_rate_from_full |
| --- | --- | --- | --- | --- | --- |
| full | 35 | 1.0000 | 0.0286 | 0.4571 | 0.0000 |
| without_conformal_calibration | 35 | 1.0000 | 0.0286 | 0.4286 | 0.1429 |
| without_deadline_penalty | 35 | 1.0000 | 0.0286 | 0.3714 | 0.0000 |
| without_reroute_penalty | 35 | 1.0000 | 0.0286 | 0.8571 | 0.0000 |
| without_uncertainty | 35 | 1.0000 | 0.0286 | 0.4000 | 0.0571 |

| condition | paired_case_count | initial_selection_change_count | realized_deadline_improvement_count | realized_deadline_worsening_count | mean_reroute_delta_condition_minus_full |
| --- | --- | --- | --- | --- | --- |
| without_conformal_calibration | 35 | 5 | 0 | 0 | -0.0286 |
| without_deadline_penalty | 35 | 0 | 0 | 0 | -0.0857 |
| without_reroute_penalty | 35 | 0 | 0 | 0 | 0.4000 |
| without_uncertainty | 35 | 2 | 0 | 0 | -0.0571 |

Inherited Step 8 policy outcomes are retained only as context; they do not isolate a
single risk term. Static deadline-proxy rates must not be compared numerically with
realized dynamic violation rates as if they shared a population.

## K=4/6/8 and classical context

Static risk, penalty-ground-state, and inherited classical-baseline tables cover
K=4/6/8. Classical baseline runtime columns were intentionally excluded, and this
report makes no speed or quantum-advantage claim.

## Reproducibility and claim boundaries

- `ablation_index.csv` maps every result/report to raw sources or targeted runs and SHA-256 hashes.
- `manifest.json` records configuration, environment, source hashes, and output hashes.
- `validation_checks.json` records machine-readable completeness and reconciliation checks.
- Source scopes remain distinct in every aggregate; there is no pooled ID/OOD, synthetic/real-topology, static/dynamic, or K4-topology/K6-8-scaling metric.
- No warm start, hardware execution, real-traffic inference, runtime comparison, speed claim, or quantum-advantage claim is present.
