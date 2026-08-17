# Validation Report — Step 9 Cross-Network/OOD Benchmark

## Status and separation

- Status: **PASS** for this bounded synthetic cross-network experiment.
- Base network partitions: 4 train, 4 validation, 4 calibration, 4 TEST-ID, and 4 held-out-topology networks.
- Known families: grid, radial, sparse_mixed, and dense_mixed, with one distinct network per family in each known-family partition.
- Held-out family: `irregular_geometric`; all four instances occur only in OOD-topology/OOD-combined evaluation.
- Every pair of train/validation/calibration/TEST-ID/OOD-topology base partitions has zero network-ID overlap. OOD-traffic intentionally shifts the TEST-ID networks, while OOD-combined intentionally shifts the held-out-topology networks.
- Observation IDs are partition-prefixed and globally unique.
- Shift scenario generation occurs only after the base partition audit and training-scale freeze.

## Frozen preprocessing

- The sole distance normalization is the raw training-set route-distance maximum: **6.918167 km**.
- The same divisor is applied without clipping to every validation, calibration, ID, topology-OOD, traffic-OOD, and combined-OOD row.
- 145 rows have normalized distance above 1.0. This is expected OOD behavior, not an error or a batch-derived rescaling.
- Network identity, topology, scenario, and observation IDs are metadata and are excluded from model features.

## Point prediction

- All three point models were trained only on train networks and ranked only on validation networks.
- Selected model: `hist_gradient_boosting`.

| evaluation_group | n_samples | mae | rmse | r2 |
|---|---|---|---|---|
| TEST-ID | 1344 | 1.1606 | 2.0755 | 0.8922 |
| OOD-topology | 1344 | 1.4990 | 2.6016 | 0.8622 |
| OOD-traffic | 1344 | 3.2740 | 4.6137 | 0.6745 |
| OOD-combined | 1344 | 3.9733 | 5.4153 | 0.6005 |

## Conformal prediction

- Residual scale fitting uses validation networks only.
- Absolute, relative, and locally scaled conformal scores use calibration networks only.
- The table below reports every requested method, alpha, and held-out group; no result was filtered for favorability.

| evaluation_group | method | configured_alpha | nominal_coverage | empirical_coverage | mean_interval_width |
|---|---|---|---|---|---|
| OOD-combined | absolute | 0.0500 | 0.9500 | 0.7827 | 11.4462 |
| OOD-combined | absolute | 0.1000 | 0.9000 | 0.5283 | 6.1879 |
| OOD-combined | absolute | 0.2000 | 0.8000 | 0.2954 | 3.5422 |
| OOD-combined | local | 0.0500 | 0.9500 | 0.6146 | 9.9814 |
| OOD-combined | local | 0.1000 | 0.9000 | 0.5037 | 8.0619 |
| OOD-combined | local | 0.2000 | 0.8000 | 0.3512 | 5.8003 |
| OOD-combined | relative | 0.0500 | 0.9500 | 0.7649 | 10.2891 |
| OOD-combined | relative | 0.1000 | 0.9000 | 0.5655 | 7.6623 |
| OOD-combined | relative | 0.2000 | 0.8000 | 0.3296 | 5.0146 |
| OOD-topology | absolute | 0.0500 | 0.9500 | 0.9464 | 11.3480 |
| OOD-topology | absolute | 0.1000 | 0.9000 | 0.8646 | 6.1762 |
| OOD-topology | absolute | 0.2000 | 0.8000 | 0.7589 | 3.5413 |
| OOD-topology | local | 0.0500 | 0.9500 | 0.9487 | 7.9161 |
| OOD-topology | local | 0.1000 | 0.9000 | 0.9033 | 6.3938 |
| OOD-topology | local | 0.2000 | 0.8000 | 0.7999 | 4.6001 |
| OOD-topology | relative | 0.0500 | 0.9500 | 0.9561 | 8.7394 |
| OOD-topology | relative | 0.1000 | 0.9000 | 0.8981 | 6.5082 |
| OOD-topology | relative | 0.2000 | 0.8000 | 0.8028 | 4.2593 |
| OOD-traffic | absolute | 0.0500 | 0.9500 | 0.8318 | 11.2908 |
| OOD-traffic | absolute | 0.1000 | 0.9000 | 0.6347 | 6.1858 |
| OOD-traffic | absolute | 0.2000 | 0.8000 | 0.4085 | 3.5414 |
| OOD-traffic | local | 0.0500 | 0.9500 | 0.5737 | 7.6874 |
| OOD-traffic | local | 0.1000 | 0.9000 | 0.4747 | 6.2090 |
| OOD-traffic | local | 0.2000 | 0.8000 | 0.3400 | 4.4672 |
| OOD-traffic | relative | 0.0500 | 0.9500 | 0.7589 | 8.2889 |
| OOD-traffic | relative | 0.1000 | 0.9000 | 0.5469 | 6.1727 |
| OOD-traffic | relative | 0.2000 | 0.8000 | 0.3251 | 4.0398 |
| TEST-ID | absolute | 0.0500 | 0.9500 | 0.9717 | 11.0817 |
| TEST-ID | absolute | 0.1000 | 0.9000 | 0.9129 | 6.1570 |
| TEST-ID | absolute | 0.2000 | 0.8000 | 0.8132 | 3.5415 |
| TEST-ID | local | 0.0500 | 0.9500 | 0.9539 | 6.1860 |
| TEST-ID | local | 0.1000 | 0.9000 | 0.9122 | 4.9965 |
| TEST-ID | local | 0.2000 | 0.8000 | 0.8132 | 3.5948 |
| TEST-ID | relative | 0.0500 | 0.9500 | 0.9635 | 7.1231 |
| TEST-ID | relative | 0.1000 | 0.9000 | 0.9234 | 5.3046 |
| TEST-ID | relative | 0.2000 | 0.8000 | 0.8289 | 3.4716 |

## Route decisions with simulated outcomes

- Every group has 4 networks × 4 pre-outcome events and exactly K=4 routes per event.
- Deadline rule fixed before prediction and outcome: `1.40 ×` the fastest candidate's free-flow time.
- Expected-time and calibrated-risk choices use no actual-outcome column. Regret and violations are computed only after selection against the shared event simulation.
- Risk selection uses `local` conformal intervals at alpha=0.10 and risk variant D.

| evaluation_group | selection_strategy | event_count | actual_deadline_violation_rate | mean_actual_deadline_margin_minutes | mean_actual_regret_minutes | actual_fastest_selection_rate |
|---|---|---|---|---|---|---|
| OOD-combined | calibrated_risk | 16 | 1.0000 | -14.4595 | 0.7257 | 0.3125 |
| OOD-combined | expected_time | 16 | 1.0000 | -14.7067 | 0.9729 | 0.3750 |
| OOD-topology | calibrated_risk | 16 | 0.7500 | -4.9193 | 0.3487 | 0.3750 |
| OOD-topology | expected_time | 16 | 0.7500 | -5.1333 | 0.5627 | 0.2500 |
| OOD-traffic | calibrated_risk | 16 | 1.0000 | -11.2364 | 0.6444 | 0.6875 |
| OOD-traffic | expected_time | 16 | 1.0000 | -11.9635 | 1.3715 | 0.6250 |
| TEST-ID | calibrated_risk | 16 | 0.8125 | -4.3946 | 0.5645 | 0.5625 |
| TEST-ID | expected_time | 16 | 0.8125 | -4.4613 | 0.6313 | 0.5625 |

## Compact real QAOA subset

- Both standard-X and feasible-XY circuits were optimized and evaluated on an ideal statevector simulator, then resampled with 2000 deterministic shots.
- The reproducible subset contains one event per evaluation-group/topology cell: 10 QUBOs and 20 circuit runs.
- The exact optimum is the modeled calibrated-risk QUBO optimum, not necessarily the actual fastest route.

| evaluation_group | method | run_count | mean_analytic_valid_probability | mean_analytic_optimal_probability | mean_shot_valid_probability | mean_shot_optimal_probability | most_probable_valid_optimum_rate |
|---|---|---|---|---|---|---|---|
| OOD-combined | feasible_xy | 1 | 1.0000 | 0.4158 | 1.0000 | 0.4095 | 1.0000 |
| OOD-combined | standard_x | 1 | 0.3240 | 0.0796 | 0.3220 | 0.0715 | 0.0000 |
| OOD-topology | feasible_xy | 1 | 1.0000 | 0.6951 | 1.0000 | 0.7025 | 1.0000 |
| OOD-topology | standard_x | 1 | 0.1965 | 0.0489 | 0.2100 | 0.0530 | 0.0000 |
| OOD-traffic | feasible_xy | 4 | 1.0000 | 0.4736 | 1.0000 | 0.4807 | 0.5000 |
| OOD-traffic | standard_x | 4 | 0.2607 | 0.0597 | 0.2574 | 0.0588 | 0.5000 |
| TEST-ID | feasible_xy | 4 | 1.0000 | 0.4893 | 1.0000 | 0.4831 | 0.7500 |
| TEST-ID | standard_x | 4 | 0.2101 | 0.0605 | 0.2119 | 0.0644 | 0.5000 |

## Scientific limitations

1. All networks and traffic are synthetic; this is not evidence of real-world operational or medical effectiveness.
2. TEST-ID means unseen network instances from known synthetic families, not observations from a previously seen network.
3. OOD-topology isolates one held-out synthetic family; it does not establish universal topology generalization.
4. OOD-traffic uses one configured shift preset, and OOD-combined composes that preset with one held-out family.
5. Route outcomes are controlled simulations. The deadline rule is illustrative and is not a clinical or dispatch policy.
6. QAOA uses K=4, depth 1, a small reproducible subset, one seeded optimization per QUBO/method, and an ideal simulator. No hardware or speedup claim is supported.
7. Group/topology QAOA cells contain only the explicit run counts in the CSV; do not infer population-level advantages.

## Reproduction

Run `python experiments/run_step9_cross_network.py` from the repository root. The runner refuses to overwrite an existing result directory. `manifest.json` records the effective configuration, package versions, seeds, source hashes, and artifact hashes.
