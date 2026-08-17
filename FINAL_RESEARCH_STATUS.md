# Final Research Status

## Exact step reached

Q-Rescue has reached the **final scientific freeze after Steps 9.2 and
9.6–9.8**. The 1,800-trial primary interaction study, cluster-aware inference,
post-hoc topology diagnostics, adaptive-conformal baseline, local noise-model
study, targeted K=16 trend, and validation/test-separated standard-X penalty
control are complete. Step 10 physical hardware has not started.

The full freeze ledger is [FINAL_SCIENCE_FREEZE.md](FINAL_SCIENCE_FREEZE.md),
and the machine-verifiable snapshot is
[SCIENCE_FREEZE_MANIFEST.json](SCIENCE_FREEZE_MANIFEST.json).

## Central result

Feasible-XY keeps ideal probability entirely inside the one-hot route sector,
but validity does not imply exact-optimum concentration or operational utility.
Across 1,800 matched trials, mean feasible mass is 1.000 for XY versus 0.180
for X, while XY mean exact-optimum mass is only 0.425. K=8 versus K=4 lowers XY
optimum mass and raises feasible entropy. Depth p=5 improves concentration over
p=1 without a resolved deadline or realized-regret effect.

Entropy is strongly negatively associated with exact-optimum mass and modestly
positively associated with encoded regret; its association with the binary
deadline outcome is weak and its interval includes zero. The supported
mechanism is therefore qualified: diffuse valid mass explains encoded
concentration failure, not the entire operational pipeline.

## Other validated findings

- Static local conformal coverage is near target for ID/topology shift but
  collapses under traffic and combined shift.
- One fixed ACI-style baseline partly recovers shifted coverage at a large
  interval-width cost, remains below target, and does not change the controlled
  route choices relative to local split conformal.
- On four cached OpenStreetMap graphs with simulated traffic, engineering replay
  succeeds but statistical transfer fails.
- The inherited QAOA-assisted dynamic policy has more deadline violations and
  longer arrival time than expected-time routing in an exploratory five-network
  comparison.
- Targeted K=16 retains ideal XY feasible mass but further lowers exact-optimum
  concentration; K=16 remains classically trivial.
- The declared local noise model reduces XY feasible and optimum mass,
  especially at p=3. It is not hardware evidence.
- Objective B (expected time plus point deadline) changes none of 300 primary
  choices. Adding uncertainty changes 18% without a resolved mean operational
  gain.
- The standard-X penalty grid is tuned on validation networks only and fully
  reported on disjoint test networks. The rule retains the conservative default;
  every valid held-out condition still has mean feasible mass below 0.069 and
  mean exact-optimum mass below 0.012. Scale sensitivity remains a limitation.
- Exact candidate-risk selection is the correctness reference and is expected
  to dominate these small K problems.

## Completed publication artifacts

- [Step 9.2 Findings](STEP_9_2_FINDINGS.md)
- [Statistical validation](results/research/step9_6_statistics/VALIDATION_REPORT.md)
- [Adaptive-conformal validation](results/research/step9_7_adaptive_conformal/VALIDATION_REPORT.md)
- [Noise validation](results/research/step9_8_noise/VALIDATION_REPORT.md)
- [K=16 validation](results/research/targeted_k16_scaling/VALIDATION_REPORT.md)
- [Penalty validation](results/research/standard_x_penalty_sensitivity/VALIDATION_REPORT.md)
- [Generated manuscript](paper/MANUSCRIPT_DRAFT.md)
- [Claims ledger](paper/CLAIMS.md)
- [Figure index](FIGURE_INDEX.md) and [table index](TABLE_INDEX.md)
- [Literature master matrix](paper/evidence/LITERATURE_MASTER_MATRIX.md)

## Claims we must not make

No quantum advantage, speedup, QAOA superiority over classical routing,
hardware behavior, observed real-traffic validity, emergency/clinical benefit,
city-scale deployment readiness, arbitrary-shift conformal guarantee, universal
topology mechanism, full VRP/dispatch solution, mixer invention, or exhaustive
novelty claim is supported.

## Remaining weaknesses

Traffic is predominantly synthetic; real graphs still use simulated traffic;
K is small and exactly enumerable; X/XY are not resource-matched; optimizer
budgets are short; dynamic inference has five networks; noise is local and
phenomenological; only one adaptive conformal method is tested; no hardware,
observed emergency data, native compilation, container/lockfile, Git history,
or independent replication is available.

## Readiness assessment

The package is ready for supervisor scientific review and is defensible for a
venue accepting a small-scale empirical diagnostic/simulation paper. It is not
ready to claim operational or quantum performance. Before journal submission,
select a venue, convert to its template, export verified citations into the
bibliography, archive a versioned artifact snapshot, and conduct an independent
number/claim audit.

## Hardware recommendation

Step 10 is conditionally worthwhile only as a small, predeclared K=4
native-compilation/noise feasibility study on one target backend, after a
backend-calibrated fake-backend replay. It should not be framed as an advantage
experiment. For many venues, observed-traffic validation or external
replication is the higher-value next step.
