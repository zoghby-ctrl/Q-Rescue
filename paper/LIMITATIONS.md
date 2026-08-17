# Limitations

## Data, uncertainty, and transfer

- Most traffic observations come from controlled synthetic generators. The
  four cached OpenStreetMap graphs provide real topology but simulated traffic,
  one origin–destination choice per region, and only four independent regions.
- The primary low/medium/high regimes control residual dispersion while holding
  point predictions and deadlines fixed. They are useful interventions, not a
  population model of real emergency-traffic uncertainty.
- Static split-conformal guarantees rely on exchangeability. The shifted
  undercoverage is evidence of assumption failure, not a contradiction of that
  theory. The clipped discrete-score ACI-style baseline is order-dependent and
  does not inherit the source method's theorem unchanged.
- Leave-one-network-out folds share training networks and are not independent
  replications. The 50 primary networks are independent experimental units
  within the generator, not a representative sample of cities.

## Optimization and quantum scope

- The QUBO chooses one precomputed simple path. It is not graph-wide route
  construction, multi-vehicle routing, fleet dispatch, or a full VRP.
- K=4, K=8, and the targeted K=16 extension are exactly enumerable and
  classically trivial. Exact candidate minimization is the correctness
  reference and is expected to dominate this workflow.
- Feasible-XY starts in and preserves the one-hot sector, so its feasibility
  advantage is structural. The comparison does not equalize state-preparation,
  connectivity, native-gate, compilation, or hardware resource costs.
- The primary optimizer uses only 15 Adam updates. Increasing p changes both
  expressivity and optimization difficulty; failure to converge is part of the
  measured fixed-budget behavior, not proof of an ansatz limit.
- Step 9.8 is a phenomenological local density-matrix model, not an IBM fake
  backend, calibrated device model, or hardware result. Angles are not
  reoptimized under noise. K=8 noise, native compilation, mitigation, queue
  time, and physical-device drift remain unmeasured.
- Penalty sensitivity changes Hamiltonian scale as well as energetic validity.
  No penalty selected on these validation networks is a universal optimum.
  Warm starts and a targeted p=8 comparison were not run; p=5 already improved
  concentration without a resolved operational gain.

## Operational interpretation

- The primary factorial attaches one matched route realization to each static
  problem. It is not a full trajectory. The inherited dynamic study has only
  five networks and 35 event rotations per policy.
- Static deadline proxies, one-decision violations, and trajectory-level
  deadline violations are different estimands. Realization noise and objective
  mismatch interrupt any simple mediation from entropy through P_opt to a
  binary deadline outcome.
- No vehicle interaction, dispatch capacity, triage priority, signal control,
  clinical outcome, or stochastic emergency-service process is modeled.
- Event latency and simulator wall time cannot be interpreted as quantum
  runtime or evidence of speedup.

## Statistics and provenance

- Confirmatory inference treats each primary network instance as the cluster
  and uses prespecified matched contrasts, percentile cluster bootstrap, paired
  sign-flip tests, and Holm family-wise correction. This avoids row-level
  pseudoreplication but does not turn the finite generator into a population
  sample.
- Topology/property and six-problem within-network correlations are explicitly
  post-hoc. Topology class determines several graph properties, so associations
  are descriptive rather than causal.
- Dynamic inference has only five clusters; even intervals excluding zero are
  exploratory. Adaptive-route comparisons have four networks per evaluation
  group.
- The directory is not a Git worktree. Config, source, artifact, and checkpoint
  hashes provide strong snapshot traceability but cannot replace commit history,
  a locked transitive environment, container, CI, or independent replication.
- The 33-paper literature matrix is evidence-traceable but not an exhaustive
  systematic review. Corpus-scoped comparisons must not become exclusivity or
  “first ever” claims.
