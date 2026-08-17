# Validation Report — Step 9.2 Feasibility Without Utility

## Status

- Primary factorial status: **PASS** for completeness and matched-input checks.
- New ideal-statevector QAOA trials: 1800.
- Matched routing problems: 300.
- Topologies: 5; uncertainty regimes: 3.
- K values: [4, 8]; depths: [1, 3, 5].
- Mixers: standard Pauli-X and one-hot feasibility-preserving XY.
- Independent matched seed IDs per topology × uncertainty × K × depth × mixer cell: 10.
- Measured cumulative optimization wall time (sum across workers): 9891.64 seconds. This is simulator accounting, not a speed benchmark.

## Matched design checks

- Every problem/depth pair contains exactly one standard-X and one feasible-XY trial.
- The two mixers share the same graph, traffic state, candidate order, uncertainty state, deadline, risk vector, exact optimum, problem hash, and optimization seed.
- Deadlines are computed from point predictions once and held fixed across low/medium/high uncertainty.
- Objective A is expected time; B adds point-deadline exceedance; C adds uncertainty while preserving B's deadline term. D is excluded from the static factorial because a reroute state does not exist there; inherited Step 8 dynamic evidence retains the reroute term.

## Descriptive primary results

- Feasible-XY mean P_feasible: 1.000000.
- Standard-X mean P_feasible: 0.179564.
- Feasible-XY mean P_opt: 0.424894.
- Standard-X mean P_opt: 0.038180.
- Feasible-XY mean conditional feasible entropy: 0.706888.
- Feasible-XY mean selected-route risk regret: 0.548661 minute-equivalent units.
- Inherited dynamic QAOA runs linked to decision-level P_opt: 35 (separate population; not pooled with the factorial).

The complete topology × uncertainty × K × depth × mixer aggregate is stored in
`qaoa_factorial_aggregate.csv`; paired X/XY differences are in
`qaoa_mixer_pairs.csv`. Statistical confidence intervals and association tests
are produced in Step 9.6 and must be consulted before inferential claims.

## Scientific limitations

1. All 1,800 primary trials are ideal statevector simulations with a fixed 15-step Adam budget. Convergence diagnostics are saved; deeper circuits may remain under-optimized.
2. The low/medium/high regimes use the existing base-time-scaled stochastic residual mechanism. They isolate predictive residual dispersion while leaving point predictions and deadlines fixed, but they remain controlled synthetic uncertainty.
3. The operational outcome attached to each primary problem is a matched one-decision route realization, not a full edge-by-edge rerouting trajectory. Full dynamic metrics come from the separately identified Step 8 population.
4. Exact optimality refers to the encoded candidate-risk objective, not the realized fastest route or a patient outcome.
5. K=4/8 remain classically trivial. Exact enumeration is the correctness reference and no quantum advantage, speedup, or operational superiority is claimed.

## Reproduction

Run `python experiments/run_step9_2_topology_uncertainty_qaoa.py all`. Each
completed trial is atomically stored under `checkpoints/trials/`; rerunning the
command resumes missing cells. Source/config/environment hashes and a hash of
every raw trial checkpoint are recorded in `manifest.json` and
`checkpoint_hashes.json`.
