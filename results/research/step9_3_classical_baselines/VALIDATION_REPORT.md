# Step 9.3 classical-baseline validation report

## Overall assessment: Ready to share with stated caveats

This benchmark reused 90 full-objective route rows from the validated
Steps 6/6.5 artifact. They form 15 candidate decisions across 5
synthetic topology families and K=4, 6, and 8. The source snapshot, instance
selections, raw runtime repetitions, aggregates, tie checks, manifest, and
machine-readable validation checks are saved beside this report.

## Methodology

Four methods select only from the same candidate table: candidate shortest
distance, shortest predicted travel time, exact full-risk argmin, and minimum
calibrated upper bound. Exact risk argmin is the tractable ground truth. Risk regret is
`selected risk - exact minimum risk` in minute-equivalent units.

Two NetworkX Dijkstra methods are deliberately separate unrestricted-graph
baselines. One minimizes `distance_km` and reports both path kilometers and its
additive base travel time; the other minimizes additive
`base_travel_time_min`. Risk, regret, and deadline proxies are reported only
when a graph-wide path exactly matches a saved candidate. The expected-time
Dijkstra path matched in 15/15 cases.
The graph-wide distance path matched in
15/15 cases.
Both use deterministic polynomial Dijkstra tie handling; tied paths are counted
without enumerating every path.

The deadline metric is conservative `upper_bound > deadline`, not an observed
late arrival. Runtime is the median of repeated local Python calls and is
reported only as an implementation diagnostic.

## Results

| Method | Comparable | Exact-optimal | Mean regret | Deadline proxy violations | Median runtime (us) |
|---|---:|---:|---:|---:|---:|
| Shortest-distance candidate | 15/15 | 6/15 | 7.800 | 7/15 | 712.80 |
| Unrestricted Dijkstra (distance) | 15/15 | 6/15 | 7.800 | 7/15 | 928.80 |
| Shortest expected-time candidate | 15/15 | 12/15 | 0.242 | 0/15 | 704.80 |
| Unrestricted Dijkstra (base expected time) | 15/15 | 9/15 | 7.309 | 6/15 | 819.55 |
| Exact risk argmin | 15/15 | 15/15 | 0.000 | 0/15 | 679.20 |
| Minimum-upper-bound heuristic | 15/15 | 12/15 | 0.242 | 0/15 | 764.90 |

## Stable tie handling

There were 6 method/decision rows with more than one minimum.
Every one of the 90 selections was unchanged after candidate
row order or graph insertion order was reversed. Candidate `x_N` ties use
numeric decision-variable order; graph-path ties use canonical node order.

## Calculation spot-checks

- Exact-risk maximum absolute regret: 0.000e+00.
- Minimum observed regret across all comparable rows: 0.000e+00.
- Raw runtime rows: 4500; expected 4500.
- Source decision sizes equal their stated K: True.
- Saved aggregate reconciliation: True.

## Issues and limitations

1. The corpus contains one controlled instance per topology and K, so topology
   differences are descriptive rather than statistically generalizable.
2. Predicted candidate time is a route-level model output; unrestricted
   Dijkstra uses additive base edge time. No route-model-to-edge prediction
   integration is implemented here, so their scopes and objectives differ.
3. The conservative deadline proxy is not a realized emergency outcome.
4. Runtime varies by machine and process load and must not be extrapolated into
   scaling or cross-platform performance claims.
5. Full-risk scores inherit the documented synthetic states, conformal model,
   deadline definition, and rerouting weight from Steps 4-6.5.
6. These synthetic inputs are simple directed graphs. This benchmark does not
   validate distance semantics for later GraphML imports where collapsing
   parallel road edges under another weight could remove a distance-optimal edge.

The graph-wide distance baseline's parallel-edge projection restriction was not
triggered: all five benchmark networks are native synthetic simple graphs with
zero collapsed parallel edges. No projected GraphML distance result is included.
Candidate shortest distance and graph-wide shortest distance happened to select
the same path in this corpus because every candidate set retained the unrestricted
distance optimum. They remain distinct baselines; this agreement is not assumed
for another candidate generator, network, or parallel-edge representation.

## Decision

Use exact risk argmin as ground truth for tractable candidate sets. Retain all
other methods as distinct baselines, and keep unrestricted Dijkstra labeled
separately in downstream comparisons.
