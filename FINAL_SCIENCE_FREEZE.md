# Final Science Freeze

Freeze date: 2026-08-13 (Africa/Cairo)

## 1. Final research question

When one-of-K emergency-route selection is solved with shallow QAOA under
predictive uncertainty, why can a feasibility-preserving ansatz remain entirely
inside the valid one-hot subspace without concentrating enough probability on
the exact risk-optimal route or improving downstream operational outcomes?

The final paper studies the interfaces among constraint validity, exact-optimum
concentration, feasible-state entropy, encoded regret, predictive calibration,
network/candidate structure, objective alignment, and realized utility. It is
an empirical diagnostic study. Exact candidate minimization is the ground truth.

## 2. Final supported contributions

1. **Matched feasibility–utility protocol.** A complete 1,800-trial factorial
   compares standard-X and one-hot XY on identical graph, traffic, route, risk,
   deadline, uncertainty, exact-optimum, and optimizer-seed inputs across five
   topology families, three uncertainty regimes, K=4/8, p=1/3/5, and ten
   matched seeds per cell.
2. **Metric separation.** The study jointly saves feasible mass, exact-optimum
   mass, optimum mass conditional on feasibility, full feasible-route
   distributions, Shannon/normalized entropy, energy and risk regret, route
   rank, top-one/top-two concentration, convergence, circuit resources, and
   realized outcomes.
3. **Cluster-aware inference.** Network instances, not repeated rows, are the
   primary independent units. Major matched effects have network-bootstrap
   intervals, paired sign-flip tests, explicit family definitions, and Holm
   correction. Post-hoc mechanism analysis is separated from confirmatory
   contrasts.
4. **Mechanism evidence with a boundary.** XY eliminates ideal invalid mass,
   but shallow fixed-budget probability remains diffuse, especially as K grows.
   Entropy explains encoded concentration and regret much better than it
   explains the binary deadline event.
5. **Reviewer controls.** Objective A/B/C isolation, a matched-prefix K=16
   trend, and a validation/test-separated standard-X penalty study address
   objective confounding, scale, and penalty-tuning objections without searching
   a broad p=8 grid.
6. **Calibration and noise diagnostics.** One fixed ACI-style baseline tests
   whether adaptive updating repairs static shift failure; a finite-shot/local-
   noise study tests whether the ideal XY invariant survives a declared
   phenomenological channel model.
7. **Negative-result preservation and traceability.** Adverse dynamic,
   calibration, transfer, noise, and execution outcomes remain in the accepted
   record. Configs, sources, raw checkpoints, summaries, and publication
   artifacts are hashed.

## 3. Final unsupported claims

The frozen corpus does **not** support:

- quantum advantage, quantum speedup, useful quantum scale, or classical
  intractability;
- QAOA superiority over exact risk selection, Dijkstra, or classical routing;
- using `P_feasible` as a proxy for `P_opt`, regret, or operational utility;
- equal hardware resources for X and XY, backend behavior, or physical-hardware
  validation;
- observed real-traffic accuracy, emergency-service benefit, clinical benefit,
  city-wide generalization, or deployment readiness;
- conformal validity under arbitrary shift or optimality of the selected
  adaptive method;
- a full shortest-path, multi-vehicle, VRP, dispatch, or signal-control solver;
- invention of XY/constraint-preserving mixers;
- exhaustive novelty, absence of prior joint work, or “first ever” status.

## 4. Complete experiment inventory

| Block | Exact population | Status |
|---|---:|---|
| Step 4 prediction | 4,800 synthetic trips | Accepted |
| Step 5 calibration | 960 calibration, 720 ID, 720 shifted rows | Accepted |
| Steps 6/6.5 risk/candidates | 90 routes, 15 decisions, K=4/6/8 | Accepted |
| Step 7 exact QUBO | 30 QUBOs, 3,360 enumerated states | Accepted |
| Step 7.5 historical QAOA | 106 runs | Accepted historical |
| Step 8 dynamic | 35 scenarios × four policies = 140 trajectories | Accepted |
| Steps 8.25/8.75 | 15 networks, 1,680 trips | Accepted |
| Step 8.5 real topology | Four cached OSM graphs, 384 simulated trips | Accepted negative |
| Step 9 cross-network/OOD | 20 networks, 9,408 trips, 64 route events | Accepted |
| Step 9.1 LONO | 20 folds, 4,320 held-out rows | Accepted |
| **Step 9.2 primary** | **1,800 QAOA trials, 300 problems, 50 networks** | **Accepted** |
| Step 9.3 classical | 15 decisions, 90 selections | Accepted |
| Step 9.4 robustness | static/penalty/QAOA plus 175 paired dynamic rows | Accepted |
| Step 9.6 inference | 50 primary network clusters | Accepted |
| Step 9.6b mechanisms | 100 topology summaries, 140 graph associations, 60 within-network associations | Accepted exploratory |
| **Step 9.7 adaptive** | **16,128 intervals, 192 method/event decisions** | **Accepted** |
| **Step 9.8 noise** | **40 source runs, 120 conditions, 1,280 count rows** | **Accepted** |
| Targeted K=16 | 72 trials, 18 problems, nine networks | Accepted indicative |
| Standard-X penalty | 350 trials; 15 validation and 35 test networks | Accepted reviewer control |

The new publication-freeze quantum corpus comprises 1,800 primary
optimizations, 72 K=16 optimizations, 350 penalty optimizations, and 120
ideal/shot/noise condition evaluations. Of the noise rows, 40 reproduce the
accepted analytic source and 80 are new finite-shot/noisy evaluations; these
categories are reported separately rather than collapsed into an inflated run
count.

## 5. Exact sample sizes, factors, seeds, and runtime

- Primary factors: 5 topologies × 3 uncertainties × 2 K values × 3 depths × 2
  mixers × 10 matched seeds = 1,800 trials. Independent network IDs: 50;
  problem IDs: 300; mixer pairs: 900.
- Primary outputs: 1,800 run rows, 10,800 feasible-route distribution rows,
  and 28,800 optimizer-history rows. All 1,800 atomic checkpoints passed.
- Primary QAOA: K={4,8}; p={1,3,5}; standard-X/feasible-XY; 15 Adam updates,
  step size 0.05. Cumulative optimizer wall time summed across workers is
  9,891.64 seconds; unique checkpoint timestamps span 64.31 minutes. These are
  reproducibility measurements, not speed comparisons.
- Targeted K=16: three topology families × low/high uncertainty × p={1,3} ×
  two mixers × three matched seeds = 72 trials on nine networks. The complex
  statevector is 1 MiB per trial. Cumulative trial wall time is 6,533.28
  seconds; checkpoint span is 2,158.17 seconds. The fixed-budget benchmark
  preflight estimated 2,131 seconds with three workers.
- Penalty study: K=8, p=3, standard-X, medium uncertainty, five topologies,
  seeds 0–2 validation and 3–9 test, seven frozen conditions, and 15 Adam
  updates: 350 trials. Its benchmark estimate is 981 seconds at three workers;
  measured cumulative trial wall time is 2,901.64 seconds and checkpoint span
  is 960.59 seconds (16.01 minutes).
- Noise study: K=4, p={1,3}, both mixers, ten matched problems, 4,096 shots,
  and analytic/noiseless-shot/noisy-shot conditions. Cumulative condition
  evaluation time is 3.90 seconds, excluding inherited ideal optimization.
- The broad optional p=8 sweep was not run. The Step 9.2 decision gate found
  p=5 sufficient to test the declared depth mechanism without resolved utility.

## 6. Statistical tests and assumptions

The prespecified inference config uses 20,000 percentile bootstrap resamples
and 100,000 paired sign-flip draws. Primary comparisons average repeated cells
within each of 50 networks before inference. Major metric families use Holm
family-wise adjustment; effect sizes and 95% intervals are primary, p-values
secondary. Wilson intervals accompany descriptive rates.

A naive row-independent multifactor ANOVA was rejected because probabilities
are bounded, XY feasibility is structurally constant at one, and repeated
topology/uncertainty/K/p/mixer cells share networks. Prespecified marginal
within-network contrasts are used instead. Dynamic inference has only five
network clusters and remains exploratory. The Step 9.6b topology/route analysis
is post-hoc, correlational, and noncausal.

## 7. Core numerical results

### Primary mixer and concentration effects

- Feasible-XY mean `P_feasible` = 1.000000; standard-X = 0.179564.
- Matched XY−X `P_feasible` = +0.820436, 95% CI [0.813327, 0.827442].
- Feasible-XY mean `P_opt` = 0.424894; standard-X = 0.038180.
- Matched XY−X `P_opt` = +0.386714 [0.354827, 0.417694].
- Matched XY−X `P_opt_given_feasible` = +0.244580
  [0.215743, 0.272530].
- Matched XY−X risk regret = −6.890432 minute-equivalent units
  [−7.898585, −5.924442].
- Matched XY−X one-decision violation = −0.327778
  [−0.413333, −0.236667]. This is a mixer contrast, not a comparison with the
  exact classical optimum.

### K, depth, uncertainty, and entropy

- XY K=8 `P_opt` means are 0.2276, 0.2763, and 0.3087 at p=1,3,5.
- XY K8−K4 `P_opt` = −0.308081 [−0.351904, −0.264147]; normalized entropy =
  +0.206264 [0.174629, 0.238829].
- XY p5−p1 `P_opt` = +0.127614 [0.101001, 0.155079]; normalized entropy =
  −0.173749 [−0.203966, −0.144351]. Its violation effect is −0.023333
  [−0.070000, 0.020000], and realized-regret effect is −0.239225 minutes
  [−0.636490, 0.115822].
- XY high−low uncertainty `P_opt` = +0.003805 [−0.015956, 0.024031]; entropy =
  −0.002133 [−0.015686, 0.011602]. Controlled uncertainty is not the primary
  encoded concentration stressor in this design.
- Mean within-network XY entropy association: `P_opt` rho = −0.731930
  [−0.801157, −0.652508]; risk regret rho = +0.269532
  [0.172498, 0.364096]; binary violation rho = +0.101983
  [−0.004924, 0.204963]; realized regret rho = +0.149325
  [0.064262, 0.236530].

### Targeted K=16

- XY stays at `P_feasible=1`. Mean `P_opt` is 0.104401 at p=1 and 0.122348 at
  p=3; normalized entropy is 0.910302 and 0.863407.
- Matched K16−K8 XY `P_opt` = −0.141319
  [−0.163381, −0.118350]; entropy = +0.053340
  [0.031273, 0.075969].

### Standard-X penalty control

- The frozen validation rule selects the existing conservative
  `max(risk)+1` default, with median penalty/threshold ratio 1.836 on validation.
- On 35 held-out networks, the selected/default condition has mean
  `P_feasible` 0.051334 [0.035789, 0.071047] and mean `P_opt` 0.006646
  [0.004393, 0.009458].
- Across all algebraically valid held-out conditions, the largest descriptive
  mean `P_feasible` is 0.068122 and the largest mean `P_opt` is 0.011040 (both
  at twice threshold). The 10× condition decreases them to 0.036374 and
  0.003929. Penalty tuning therefore does not explain away the standard-X
  concentration failure under the fixed optimizer protocol.

### Objective and dynamic results

- Objective B versus expected-time A changes 0/300 routes.
- Adding uncertainty in C changes 18% of routes, but realized time changes by
  +0.089321 minutes [−0.293332, 0.430981] and violation by +0.010000
  [−0.020000, 0.046667].
- Inherited QAOA−expected-time dynamic violation = +0.142857
  [0.000000, 0.371429]; arrival time = +1.681931 minutes
  [0.020418, 4.631830]. Only five clusters make these exploratory.

### Calibration and noise

- Static local 90% coverage: TEST-ID 0.9122; topology shift 0.9033; traffic
  shift 0.4747; combined shift 0.5037.
- Adaptive coverage: TEST-ID 0.8884; topology shift 0.8847; traffic shift
  0.8430; combined shift 0.8460. Traffic/combined mean widths rise to 14.909
  and 17.957 minutes. Adaptive and local methods select identical routes in all
  64 controlled events.
- XY noisy−ideal `P_feasible`: −0.249439
  [−0.252643, −0.246019] at p=1 and −0.429151
  [−0.433228, −0.425488] at p=3.
- XY noisy−ideal `P_opt`: −0.106662
  [−0.139696, −0.073009] at p=1 and −0.241100
  [−0.337997, −0.150140] at p=3. The model is phenomenological, not hardware.

## 8. Negative, null, and adverse results

- Perfect ideal feasible mass does not imply perfect or even majority
  exact-optimum mass in many cells.
- Entropy supports the encoded concentration mechanism but does not directly
  explain the binary operational deadline event.
- Greater tested depth improves concentration without a resolved operational
  gain; a p=8 fishing expedition is therefore unwarranted.
- High versus low controlled uncertainty does not materially change encoded XY
  concentration, although realized uncertainty still changes outcomes.
- The point-deadline term changes no primary static selections. The uncertainty
  term changes choices without a resolved realized benefit.
- The inherited QAOA-assisted dynamic policy is worse in direction than
  expected-time rerouting on deadline violations and arrival time.
- Static conformal calibration collapses under traffic/combined shift. Adaptive
  updating partly recovers coverage only with much wider intervals and no local-
  baseline route change in this event corpus.
- Real-topology engineering transfer succeeds while frozen-model statistical
  transfer fails (`R²=-0.1866`, local coverage 0.5443).
- The local noise model destroys exact XY subspace invariance and further
  reduces optimum mass.
- K=16 strengthens the adverse concentration trend and remains exactly,
  classically tractable.
- Standard-X penalty behavior is scale-sensitive; the held-out sweep is a
  robustness control, not a tuned victory for either mixer.

## 9. Reproducibility status

- Every final runner stores effective config, seeds, timestamps, environment,
  source hashes, raw results, summaries, validation report, and artifact hashes.
- Step 9.2 and both targeted QAOA studies use atomic per-trial checkpoints and
  resume missing IDs only. Finalization refuses duplicates/incomplete sets.
- The initial Step 9.2 duplicate-worker execution incident and the penalty JSON
  serialization failure are documented. Neither produced duplicated/substituted
  accepted observations; failure records are retained and hashed.
- Step 9.7 regenerates pre-CSV source floats from accepted hash-matching code and
  reproduces static intervals to 7.105e−15 before adaptive evaluation.
- Figures, tables, and manuscript numbers are rebuilt from accepted artifacts;
  their builders and outputs have separate manifests.
- The final verifier checks all final result manifests/checkpoints, publication
  manifests, local Markdown links, the 33-paper evidence row count, and the full
  pytest suite before writing `SCIENCE_FREEZE_MANIFEST.json`.
- The frozen full-suite expectation is 158/158 passing tests; the verifier
  refuses to write the manifest if the count changes or any test fails.
- Remaining provenance gap: the workspace is not a Git worktree and lacks a
  fully hashed lockfile/container/CI workflow. SHA-256 source/artifact snapshots
  mitigate but do not erase this limitation.

## 10. Remaining limitations

The largest scientific weaknesses are synthetic traffic, only four real road
graphs with simulated outcomes, small and exactly enumerable candidate sets,
fixed short optimizer budgets, structurally unequal X/XY resource costs, no
native compilation or backend-calibrated noise, only five dynamic network
clusters, one adaptive method, post-hoc topology mechanisms, no observed
emergency-service data, and no independent replication. The 33-paper review is
structured and primary-source-verified but not a registered systematic review.

## 11. Is Step 10 hardware still worthwhile?

**Conditionally, yes—but it is not required to rescue the current result.** A
small predeclared K=4 study can answer whether feasible mass and concentration
survive native compilation, state preparation, connectivity overhead, device
noise, calibration drift, shots, and mitigation. It should first use a current
backend-calibrated fake backend; then compile X/XY to the same physical target,
retain failed jobs, and report raw/mitigated distributions and full wall time.

Step 10 is not worthwhile if framed as a speedup or routing-improvement test:
the problems are classically trivial, the local noise result is adverse, and
the current corpus offers no basis for advantage. Observed-traffic validation
or independent replication may yield more scientific value for many venues.

## 12. Supervisor and submission readiness

The manuscript is **ready for supervisor scientific review**. Its central
claim, negative findings, statistical scope, literature boundary, figures,
tables, and limitations are internally consistent and artifact-traceable.

Submission is scientifically defensible **only as a small-scale empirical
diagnostic/simulation study** with the current guarded title and claims. It is
not yet a push-button journal submission: a target venue must be selected,
Markdown converted to the venue template, citations exported to a bibliography,
the artifact snapshot archived in version control or a DOI-bearing repository,
and an independent number/wording review completed. A venue demanding observed
traffic, hardware, or external replication would require additional work rather
than stronger prose.
