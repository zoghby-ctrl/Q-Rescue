# Validation Report — Steps 8.25 and 8.75

## Scope and validation status

- Status: **PASS** for this bounded repeated-synthetic benchmark.
- Networks: 15 total; exactly 3–3 instances per family across all five configured families.
- Candidate routing: K=4; 15/15 networks produced all K loopless routes.
- Evaluation scope: held-out observations from known synthetic networks. This is ID evaluation, not unseen-network generalization.
- Traffic: controlled simulation only; no live or real-city traffic is represented.
- Traffic variation: every network receives the same four scenario regimes, spanning incident probability 0.01–0.05, congestion standard deviation 0.07–0.18, and prediction-noise fraction 0.03–0.25. Variation is within every topology, not a topology-specific confounder.
- Connectivity: all 15 generated graphs have one weak and one strong component; largest weak/strong component fractions are 1.0. This benchmark therefore does not study disconnected networks.
- QAOA: ideal noiseless simulator, one representative per topology, one fixed depth/seed/iteration budget per method.

## Point-model result

- Validation-selected model: `linear_regression`.
- Held-out test rows: 255.
- Test MAE: 0.7570 minutes.
- Test RMSE: 1.0653 minutes.
- Test R²: 0.9566.

| topology | n_samples | mae | test_rmse | r2 |
|---|---|---|---|---|
| dense_mixed | 51 | 0.7068 | 1.0998 | 0.9246 |
| grid | 51 | 0.4919 | 0.5854 | 0.9227 |
| irregular_geometric | 51 | 0.7824 | 1.0380 | 0.9402 |
| radial | 51 | 0.6860 | 0.8426 | 0.7296 |
| sparse_mixed | 51 | 1.1181 | 1.5281 | 0.9093 |

## Conformal result

- Method: locally scaled split conformal.
- Alpha: 0.10; nominal coverage: 0.9000.
- Empirical test coverage: 0.9255.
- Mean interval width: 3.1951 minutes.

| topology | n_samples | empirical_coverage | mean_interval_width |
|---|---|---|---|
| dense_mixed | 51 | 0.9020 | 3.0610 |
| grid | 51 | 0.9804 | 2.5448 |
| irregular_geometric | 51 | 0.9216 | 3.3168 |
| radial | 51 | 0.9216 | 2.6684 |
| sparse_mixed | 51 | 0.9020 | 4.3848 |

## Risk-routing result

- Variant D changed the expected-time route in 3/15 network instances.
- Variant D conservative upper-bound deadline violations: 0/15 selected routes.
- Mean selected-route deadline margin: 0.8476 minutes.
- These are model-based conservative decisions, not observed emergency outcomes.

## Compact QAOA comparison

- `feasible_xy`: n=5, mean valid=1.0000, mean optimal=0.3533, most-probable-valid optimum rate=0.6000.
- `standard_x`: n=5, mean valid=0.1515, mean optimal=0.0368, most-probable-valid optimum rate=0.2000.

Wall-clock timing is intentionally not used to claim a quantum speedup.

## Scientific limitations

1. Three instances per family are enough for a repeated smoke benchmark, not universal topology claims.
2. Training and test rows come from the same 15 network instances; unseen-network evaluation belongs to Step 9.
3. Correlations use only five QAOA representative networks and are explicitly exploratory, with no significance claim.
   Correlations against feasible-XY valid probability are undefined because validity is structurally 1.0; the CSV stores NaN plus `near_constant_qaoa_outcome` rather than correlating floating-point noise.
4. The traffic generator is controlled synthetic data, even when graph structure is road-like.
5. QAOA uses an ideal simulator, K=4, depth 1, one optimization seed, and 12 iterations.
6. No hardware, noise, runtime advantage, medical effectiveness, or real-world deployment claim is supported.

## Reproduction

Run `python experiments/run_steps8_25_8_75_topology.py` from the repository root. The runner refuses to overwrite an existing output directory. Exact seeds, parameters, package versions, config hash, source hashes, and artifact hashes are stored in `manifest.json`.
