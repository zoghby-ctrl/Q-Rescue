# Q-Rescue TA progress log

## CURRENT STEP

Final science freeze — Steps 9.2 and 9.6–9.8 are complete and validated,
including targeted K=16 and standard-X penalty controls. Step 10 physical
hardware execution has not started. Earlier dated entries are retained as a
historical progress log; this section and the final freeze entry supersede their
then-current status statements.

## STEP 4 — GENERAL TRAFFIC AI (COMPLETED 2026-08-10)

### WHAT WE BUILT

A location-neutral road-network schema, five deterministic road-like synthetic
topology generators, a complete-trip traffic simulator, strict four-way data
splits, and three regression baselines. All generalized artifacts live outside
Prototype 0.1.

### WHY

The old prototype used four manually assigned travel times and uncertainties.
Deadline-risk optimization needs predictions learned from data, and the paper
question requires the same model contract to operate across different graph
structures without a city identifier.

### INPUT

- Five directed networks: grid, radial, irregular geometric, sparse mixed, and
  dense mixed.
- Six controlled scenarios: normal, rush, rain, congestion, accident, and
  prediction noise.
- 4,800 synthetic complete-trip observations generated with seed 4200.
- Thirteen numeric features; network/topology/scenario/observation IDs are kept
  only for evaluation and traceability.

### OUTPUT

- Train: 2,400 rows.
- Calibration: 960 rows.
- Validation: 720 rows.
- Test: 720 rows.
- Raw data, splits, metrics, predictions, fitted model, quality report, and
  environment/config manifest in `results/research/step4_traffic_ai/`.

### MATH

For actual times y_i and predictions y_hat_i:

- MAE = (1/n) sum |y_i - y_hat_i|.
- RMSE = sqrt((1/n) sum (y_i - y_hat_i)^2).
- R-squared = 1 - sum (y_i-y_hat_i)^2 / sum (y_i-y_bar)^2.

Model selection minimizes validation RMSE only. The independent test set is
used once after selection.

### IMPORTANT CODE

- `src/network_schema.py`: canonical graph validation.
- `src/synthetic_networks.py`: five topology families.
- `src/traffic_scenarios.py`: controlled traffic distributions.
- `src/traffic_data.py`: trip generation and strict splits.
- `src/traffic_model.py`: regression models and metrics.
- `experiments/run_step4_traffic_ai.py`: reproducible benchmark runner.

### RESULT

Histogram gradient boosting had the lowest validation RMSE and remained the
best test-RMSE model. An initial target-leakage issue in the average-speed
feature was detected, fixed, and the entire benchmark was rerun before any
result was accepted.

### IMPORTANT NUMBERS

| Model | Test MAE (min) | Test RMSE (min) | Test R-squared |
|---|---:|---:|---:|
| Linear regression | 1.444 | 2.188 | 0.904 |
| Random forest | 0.957 | 1.780 | 0.936 |
| HistGradientBoosting | **0.904** | **1.713** | **0.941** |

Data checks: 4,800/4,800 unique IDs, zero missing values, zero split overlap,
all five networks in every split, and 98.65% multi-edge trips.

### WHAT I SHOULD SAY TO MY TA

“Step 4 replaces hand-written route times with a reproducible trip-level
prediction pipeline. The algorithm never sees the network or city name as a
feature. I trained on 2,400 rows, selected the model on a separate 720-row
validation set, reserved 960 rows exclusively for conformal calibration, and
reported the final score on an untouched 720-row test set. These are synthetic
ID results, not yet a claim of cross-network generalization.”

### QUESTIONS MY TA MAY ASK

1. Why keep a separate calibration set? Split conformal needs residuals from
   rows unused in model fitting or model selection.
2. Why not use network identity? That would let the model memorize a network
   rather than test transferable relationships.
3. Why choose by RMSE? Emergency routing is sensitive to large errors, so RMSE
   is the preregistered selection metric; MAE is still reported.
4. Is this real traffic? No. It is controlled simulated traffic on synthetic
   road graphs.

### LIMITATIONS

- This step is in-distribution and includes every topology in training.
- Only one network instance per topology is used here.
- Traffic is simulated, and good prediction scores partly reflect the known
  data-generating structure.
- Random row splits do not test temporal shift.
- Results do not establish real-world generalization or causal effects.

### NEXT STEP

Use only the 960-row calibration split to fit manual split-conformal residual
quantiles at alpha 0.05, 0.10, and 0.20. Then evaluate empirical coverage and
interval width overall, by network, by topology, and under an explicit shifted
traffic distribution.

## STEP 5 — CALIBRATED UNCERTAINTY (COMPLETED 2026-08-10)

### WHAT WE BUILT

A manual absolute-residual split-conformal regressor, a prediction-scaled
comparator, and a validation-trained locally scaled comparator, all with the correct finite-sample quantile rule,
nonnegative travel-time bounds, alpha values 0.05, 0.10, and 0.20, and
coverage/width reports overall, by network, and by topology.

### WHY

A point prediction alone cannot express whether an emergency route has a tight
or unreliable arrival estimate. Conformal intervals provide an auditable way
to turn held-out prediction errors into empirical travel-time uncertainty.

### INPUT

- The Step 4 histogram-gradient-boosting model.
- Exactly 960 calibration rows unused in model training and selection.
- 720 untouched ID test trips.
- 720 newly generated shifted-traffic trips on the same five graph structures.

### OUTPUT

Calibration residual quantiles, row-level lower/upper bounds, empirical
coverage, mean/median interval width, and group tables in
`results/research/step5_conformal/`.

### MATH

For calibration residuals r_i = |y_i - y_hat_i| and n calibration rows:

q_level = min(1, ceil((n+1)(1-alpha))/n).

q is the empirical residual quantile at that level using the “higher” rule.
The prediction interval is:

lower = max(0, y_hat - q), upper = y_hat + q.

Coverage is the fraction of actual y values satisfying lower <= y <= upper.
For the relative comparator, scores are |y-y_hat|/max(y_hat,1 minute), and the
route-specific half-width is q*max(y_hat,1 minute).
For local scaling, an identity-free residual model is fitted on validation
rows, and scores are |y-y_hat|/predicted_error_scale(x). Calibration remains a
separate later split.

### IMPORTANT CODE

- `src/conformal.py`: manual quantile, calibration, intervals, and metrics.
- `experiments/run_step5_conformal.py`: ID and shifted-traffic benchmark.
- `configs/conformal_benchmark.json`: alphas and shift seed/configuration.

### RESULT

ID coverage was near but consistently below nominal in this fixed run. Coverage
failed severely under traffic shift, exactly illustrating why ordinary split
conformal must not be assumed to remain calibrated after distribution change.

### IMPORTANT NUMBERS

| alpha | Nominal | ID coverage | Shifted coverage | Half-width q (min) |
|---:|---:|---:|---:|---:|
| 0.05 | 95% | 94.31% | 71.94% | 3.115 |
| 0.10 | 90% | 87.78% | 52.50% | 2.043 |
| 0.20 | 80% | 76.25% | 32.36% | 1.177 |

At alpha 0.10, topology-level ID coverage ranged from 76.39% to 97.22%.
The relative comparator at alpha 0.10 achieved 87.92% ID coverage and 45.56%
shifted coverage with mean ID width 3.926 minutes. It is useful for
route-specific uncertainty but did not solve OOD undercoverage.
The local-scale method at alpha 0.10 achieved 89.72% ID coverage, 45.83% shifted
coverage, and mean ID width 3.643 minutes. Its narrower, feature-dependent
intervals are a fixed-run result, not yet a general superiority claim.

### WHAT I SHOULD SAY TO MY TA

“I implemented split conformal manually: train the point model first, freeze
it, calculate absolute residuals on a separate calibration set, and take the
finite-sample corrected residual quantile. At alpha 0.10 the ID interval covered
87.8% of test trips, but only 52.5% after the controlled traffic shift. So the
method provides useful marginal uncertainty under its assumptions, but the
experiment honestly shows that it is not automatically OOD-calibrated.”

### QUESTIONS MY TA MAY ASK

1. Is q a probability? No. It is a travel-time residual quantile in minutes.
2. Why did ID coverage miss 90%? Coverage is marginal over exchangeable draws;
   a fixed test realization can miss nominal, and heterogeneous groups differ.
3. Does conformal guarantee each topology? No. Global split conformal gives a
   marginal guarantee, not conditional group coverage.
4. Why does shift fail? The residual distribution changes, violating the
   exchangeability assumption between calibration and evaluation rows.

### LIMITATIONS

- Synthetic data and one fixed split.
- Global intervals have constant q and do not adapt to route scale.
- Relative scaling adapts width to prediction scale but remains a marginal,
  exchangeability-dependent method.
- A normalized local-scale comparator is now included, but no adaptive or online
  conformal method is included.
- Traffic shift deliberately violates calibration assumptions.
- Coverage is about travel time, not emergency outcome probability.

### NEXT STEP

Convert prediction, bounds, deadline, and reroute cost into a transparent
minute-equivalent route objective. Run ablations A through E and keep every
component visible so the QUBO receives only a defensible route-risk vector.

## STEP 6 — GENERAL RISK OBJECTIVE (COMPLETED 2026-08-10)

### WHAT WE BUILT

A network-independent, minute-equivalent route score with visible expected
time, interval width, deadline exceedance, and rerouting components, plus the
requested A–E ablation variants.

### WHY

The optimizer needs one auditable number per route. Keeping the components in
minutes makes weights interpretable and prevents visually impressive but
scientifically meaningless objectives.

### INPUT

Point predictions and locally scaled conformal bounds at alpha 0.10, a
controlled deadline, route switching cost, and weights lambda_u=1, lambda_d=2,
lambda_r=1.

### OUTPUT

Every route component, all five ablation scores, and selected routes for 15
network/K decisions in `results/research/steps6_6_5_routing_risk/`.

### MATH

deadline_margin = deadline - upper_bound.

deadline_penalty = max(0, -deadline_margin).

risk = prediction + lambda_u*half_width + lambda_d*deadline_penalty
       + lambda_r*reroute_penalty_minutes.

### IMPORTANT CODE

- `src/deadline_risk.py`: objective and A–E variants.
- `src/route_features.py`: inference-time feature contract.
- `experiments/run_steps6_6_5_routing_risk.py`: validated ablation.

### RESULT

The formula was independently reconstructed from saved columns to 1.42e-14
maximum error. In this fixed benchmark, variants B–D selected the same route as
expected-time-only in all 15 decisions. Variant E changed 3/15 because of the
explicit switching penalty. We did not tune the weights to force a positive
uncertainty result.

### IMPORTANT NUMBERS

- 15 decisions per ablation.
- 0 conservative upper-bound deadline violations for every selected policy.
- Mean selected margin: 2.073 minutes for A–D and 2.015 for E.
- Variant E changed 3 of 15 decisions; B, C, and D changed zero.

### WHAT I SHOULD SAY TO MY TA

“The risk objective is dimensionally transparent: every term is measured in
minutes or weighted minute-equivalents. On this first fixed seed, uncertainty
and deadline terms did not change the winner; only rerouting cost did. I keep
that as a negative result and will test repeated networks, states, and weights
later rather than cherry-pick an example.”

### QUESTIONS MY TA MAY ASK

1. Why use the upper bound? It is a conservative travel-time estimate tied to
   the calibrated interval, not a probability.
2. Are weights clinical values? No. They are experimental policy parameters.
3. Why no improvement yet? The expected-time winner also had competitive
   uncertainty and deadline margin in these 15 fixed decisions.

### LIMITATIONS

- Controlled deadline defined from candidate upper-bound quantiles.
- One state seed and one network per topology.
- No observed arrivals or dynamic decisions in this step.
- No weight sensitivity yet.

### NEXT STEP

Encode each arbitrary risk vector as an exactly-one QUBO and verify K=4,6,8.

## STEP 6.5 — GENERAL CANDIDATE ROUTING (COMPLETED 2026-08-10)

### WHAT WE BUILT

Deterministic K-shortest loopless routing for arbitrary directed canonical
graphs, exact-K failure handling, route summaries, and directed-edge diversity
metrics.

### WHY

Prototype 0.1 hardcoded Route A/B/C/D. Generalization requires routes to be
data derived by a transparent graph algorithm, with x_0...x_(K-1) assigned only
after generation.

### INPUT

Each directed graph, its recorded origin/destination, edge travel-time weight,
and K in {4,6,8}.

### OUTPUT

Exactly 4, 6, and 8 candidates for each of five topology families, including
ordered nodes, edges, distance, base/expected time, interval, deadline margin,
risk, and pairwise edge diversity.

### MATH

Routes are ordered by additive shortest-simple-path weight. Pairwise diversity
is 1 minus the directed-edge Jaccard similarity:

diversity(P,Q) = 1 - |E(P) intersect E(Q)| / |E(P) union E(Q)|.

### IMPORTANT CODE

- `src/candidate_routes.py`: generation, summaries, and exact-K errors.
- `src/route_features.py`: model-compatible route features.
- `tests/test_candidate_routes.py`: K=4/6/8 and determinism checks.

### RESULT

All 15 network/K sets returned the requested number of valid loopless routes.
Topology produced visibly different overlap structures.

### IMPORTANT NUMBERS

- 5 networks x 3 K values = 15 candidate sets.
- 90 saved candidate-route rows.
- Mean Jaccard diversity: grid 0.787, dense mixed 0.708, sparse mixed 0.686,
  radial 0.512, irregular geometric 0.495.

### WHAT I SHOULD SAY TO MY TA

“Cities are not branches in the code. Every candidate is generated from the
same directed-graph schema using shortest simple paths. K is explicit, and the
runner fails rather than silently returning fewer routes.”

### QUESTIONS MY TA MAY ASK

1. Why K-shortest simple paths? It is transparent, reproducible, and prevents
   cycles; later diversity-aware alternatives can be compared.
2. Are all candidates independent? No. Their overlap is measured and reported.
3. Why directed-edge Jaccard? Direction matters on one-way road networks.

### LIMITATIONS

- K-shortest paths can be highly overlapping.
- Candidate quality depends on the edge weight supplied.
- This is single-origin, single-destination candidate generation.
- No real GraphML network is included yet.

### NEXT STEP

Build the generalized exactly-one QUBO, choose a demonstrably safe penalty,
enumerate all 2^K assignments, verify QUBO/Ising equivalence, and save scaling.

## STEP 7 — GENERALIZED QUBO (COMPLETED 2026-08-10)

### WHAT WE BUILT

An arbitrary-K exactly-one route QUBO with safe automatic penalty selection,
full brute-force tables, manual Ising conversion through the preserved
Prototype 0.1 algebra, and explicit scaling records.

### WHY

The historical QUBO was explained with four named routes. The paper pipeline
must accept any nonnegative risk vector and verify small K instances before a
variational quantum method is allowed to optimize them.

### INPUT

Thirty route-risk vectors: five networks, K=4/6/8, and risk variants D and E.

### OUTPUT

QUBO coefficients, every bitstring cost, exact selected route, Ising
equivalence error, penalty, and scaling tables in
`results/research/step7_qubo/`.

### MATH

Q(x) = sum c_i x_i + P(sum x_i - 1)^2.

Expanded:

- constant: P
- linear x_i: c_i - P
- pair x_i*x_j: 2P

The safe rule is P=max(c)+1 for these nonnegative route scores.

### IMPORTANT CODE

- `src/risk_qubo.py`: validated generalized coefficients and enumeration.
- `experiments/run_step7_qubo.py`: 30-instance verification.
- `tests/test_risk_qubo.py`: independent K=4/6/8 checks.

### RESULT

All 30 QUBO ground states were valid and matched direct classical argmin. The
largest QUBO/Ising energy difference was 1.18e-11 from floating-point rounding.

### IMPORTANT NUMBERS

| K | Qubits | States | Total QUBO terms | Nonidentity Hamiltonian terms |
|---:|---:|---:|---:|---:|
| 4 | 4 | 16 | 11 | 10 |
| 6 | 6 | 64 | 22 | 21 |
| 8 | 8 | 256 | 37 | 36 |

Total enumerated assignments: 3,360.

### WHAT I SHOULD SAY TO MY TA

“For any K-route risk vector, I expand the exactly-one penalty by hand. I use a
penalty above the largest valid cost, enumerate all 2^K states for K up to 8,
and compare the QUBO energy with the manually converted Ising energy. Every one
of 30 instances chose the same route as direct classical argmin.”

### QUESTIONS MY TA MAY ASK

1. Why P>max(c)? It is a clear sufficient condition ensuring the zero state is
   worse than every valid route when costs are nonnegative.
2. Is it the smallest possible P? No. It is a conservative reproducible rule;
   sensitivity will be tested later.
3. Why brute force? At K<=8 it is the correctness ground truth, not a claimed
   scalable solver.

### LIMITATIONS

- Search grows as 2^K.
- Penalty magnitude can hurt variational conditioning.
- Only exactly-one candidate selection is encoded.
- Encoding correctness says nothing about QAOA speed or success probability.

### NEXT STEP

Benchmark standard penalty X-mixer QAOA and a one-hot feasibility-preserving
XY-style method at p=1,2,3 across multiple fixed seeds, with exact classical
optima retained as ground truth. Do not run hardware.

## STEP 7.5 — QUANTUM METHODS (COMPLETED 2026-08-10)

### WHAT WE BUILT

Generalized PennyLane circuits for standard penalty X-mixer QAOA and a
one-hot feasibility-preserving ring-XY ansatz, seeded Adam optimization, exact
probabilities, separate 2,000-shot estimates, training histories, and logical
resource records.

### WHY

The standard mixer explores invalid bitstrings, while the XY method tests
whether preserving the exactly-one constraint improves useful sampling. Both
use the same cost Hamiltonian and exact classical optimum.

### INPUT

Verified risk-QUBO Hamiltonians; K=4 on all five topologies at p=1/2/3 and
three seeds; K=6/8 grid pilots at p=1/2 and two seeds.

### OUTPUT

106 complete run rows, 3,816 training-history rows, measurement counts,
method/depth aggregates, topology tables, angles, seeds, shot counts, circuit
depth, and logical gate counts in `results/research/step7_5_qaoa/`.

### MATH

Standard QAOA begins in a uniform superposition over all 2^K bitstrings and
alternates exp(-i gamma H_cost) with an X mixer.

Feasible XY begins in a W state uniform over the K one-hot bitstrings and uses
an XY ring mixer. XY interactions conserve Hamming weight, so probability
cannot leave the exactly-one subspace in the ideal circuit.

### IMPORTANT CODE

- `src/feasibility_qaoa.py`: state preparation, mixers, optimization, metrics.
- `experiments/run_step7_5_qaoa.py`: full 106-run benchmark.
- `tests/test_feasibility_qaoa.py`: bit order and feasibility invariance.

### RESULT

At K=4, XY achieved 100% valid probability by construction and much higher
mean optimal probability than standard X under the fixed optimizer budget.
XY mean optimal probability was 49.74%, 62.15%, and 60.59% at p=1,2,3;
standard X was 7.02%, 6.40%, and 6.83%. Increasing depth was not consistently
beneficial. K=8 standard-X validity fell to 2.65% at p=1 and 4.14% at p=2,
while XY remained feasible.

### IMPORTANT NUMBERS

- 106 runs; 35 Adam iterations per run; 2,000 shots per final distribution.
- K=4 exact route most probable: XY 15/15 at p=1, 15/15 at p=2, 13/15 at
  p=3; standard X 2/15, 4/15, 3/15.
- K=8 XY optimal probability: 21.94% p=1, 26.78% p=2 (two-seed grid pilot).
- K=8 standard optimal probability: 0.74% p=1, 0.58% p=2.

### WHAT I SHOULD SAY TO MY TA

“The main quantum finding is sampling efficiency, not speed. The standard X
mixer spends most probability on invalid states, especially as K grows. A W
state plus XY mixer preserves one-hot feasibility exactly on the ideal
simulator and substantially raises optimal-route probability in this small
benchmark. This is an established method used inside Q-Rescue, not our
invention, and it does not establish quantum advantage.”

### QUESTIONS MY TA MAY ASK

1. Why is XY validity exactly 100%? Its Hamiltonian conserves Hamming weight
   and the initial W state has Hamming weight one.
2. Is p=3 better? Not consistently under 35 iterations; optimization becomes
   harder and seed variance grows.
3. Are gate counts hardware-realistic? No. PennyLane counts StatePrep as a
   logical operation; native decomposition is not included.
4. Where is warm start? Deliberately unfinished; using the exact optimum would
   leak the answer, and a valid continuous-relaxation protocol is not ready.

### LIMITATIONS

- No noise or hardware.
- Small K, few seeds, and one K=6/8 topology.
- Fixed iteration budget may disadvantage deeper circuits.
- Different initial feasible priors are part of the method comparison.
- No warm-start implementation.

### NEXT STEP

Run edge-by-edge dynamic rerouting with normal, rush, rain, localized
congestion, accident, closure, and prediction-noise events. Compare fixed,
expected-time, risk-aware exact classical, and actual QAOA-assisted policies.

## STEP 8 — DYNAMIC NETWORK BENCHMARK (COMPLETED 2026-08-10)

### WHAT WE BUILT

An edge-by-edge simulator that updates traffic state, removes closures,
regenerates routes from the current node, predicts travel-time intervals,
scores deadline risk, chooses a policy route, realizes one noisy edge time, and
repeats until arrival or failure.

### WHY

Static route choice cannot test accidents, closures, oscillation, or whether a
quantum sample leads to a useful operational decision. Dynamic evaluation is
where feasibility and route quality must be separated.

### INPUT

Five topology networks, seven rotating event sequences, K=4, deadline equal to
1.8 times base shortest-path time, risk weights (1,2,1), and identical realized
edge-noise seeds across matched policies.

### OUTPUT

140 run summaries, 969 edge decisions, 264 QAOA decision audits, policy
aggregates, and topology tables in `results/research/step8_dynamic_rerouting/`.

### MATH

Dynamic expected edge time is base_time times:

1 + 0.90*congestion + 0.20*rain + 0.70*incident.

Each realized edge time adds zero-mean event-specific noise. Route stability is
1 - reroute_count/max(1, decision_count-1).

### IMPORTANT CODE

- `src/dynamic_rerouting.py`: events, policies, traversal, and metrics.
- `experiments/run_step8_dynamic_rerouting.py`: 140 matched cases.
- `configs/dynamic_rerouting_benchmark.json`: seeds and policy parameters.

### RESULT

Expected-time rerouting was strongest in this fixed benchmark. It violated the
deadline in 1/35 runs; exact uncertainty-aware classical in 2/35; QAOA-assisted
in 6/35. Never reroute failed all five cases where a closure occurred first.
QAOA remained feasible but matched exact risk argmin in only 85.23% of dynamic
decisions and oscillated badly on sparse mixed.

### IMPORTANT NUMBERS

- Expected-time: 100% arrivals, 2.86% violations, 0.89 reroutes/run.
- Risk-aware classical: 100%, 5.71%, 0.46 reroutes/run.
- QAOA-assisted: 100%, 17.14%, 1.43 reroutes/run.
- Never reroute: 85.71% arrivals, 14.29% violations.
- QAOA: 264 decisions, 100% valid, 56.00% mean optimal probability, 85.23%
  exact-risk-choice match.

### WHAT I SHOULD SAY TO MY TA

“Dynamic simulation exposed a failure that a static valid-probability chart
would hide. The feasible XY circuit always returned a valid route, but its
most likely route was not the exact risk optimum in about 15% of decisions.
Those misses accumulated into oscillation and deadline failures on the sparse
network. In this run, expected-time rerouting beat both uncertainty-aware
policies, so I cannot claim deadline-reliability improvement yet.”

### QUESTIONS MY TA MAY ASK

1. Why did QAOA fail despite 100% validity? Valid means one route selected, not
   the best route; optimal probability and repeated decisions matter.
2. Why did never-reroute arrival time look low? Its mean excludes five failed
   closure trips, so arrival rate and violation rate must be read with it.
3. Is an unnecessary reroute hindsight? It uses the current event-adjusted
   expected graph only, not future events.

### LIMITATIONS

- One network instance per family and one event seed.
- Synthetic event physics and experimental deadlines.
- Shallow p=1 QAOA with 12 iterations.
- No congestion interaction between vehicles.
- The risk objective has not yet shown reliability gain.

### NEXT STEP

Generate multiple seeded instances per topology, varying size/density and
traffic distributions. Report all metrics by family and preserve failure
regions rather than averaging them away.

## STEP 8.25 — SYNTHETIC TOPOLOGY GENERALIZATION (COMPLETED 2026-08-10)

### WHAT WE BUILT

A repeated benchmark with 15 road-like graphs: three independently seeded and
parameterized instances of each grid, radial, irregular geometric, sparse
mixed, and dense mixed family. The same prediction, conformal, candidate-route,
risk-QUBO, and QAOA pipeline operates on every graph.

### WHY

One graph per family cannot separate a topology effect from a lucky graph
instance. Repetition tests whether the pipeline remains functional when node
count, connectivity, edge-time scale, route diversity, incidents, congestion
variance, and observation noise change.

### INPUT

- 15 canonical directed graphs, three per topology family.
- 4 traffic regimes per graph: normal, rush, rain, and prediction noise.
- Incident probability 0.01–0.05, congestion standard deviation 0.07–0.18,
  and observation-noise fraction 0.03–0.25 within every topology.
- K=4 routes; alpha=0.10; risk variants A and D.
- One representative graph per topology for compact p=1 QAOA, with one fixed
  seed, 12 Adam iterations, standard X and feasible XY.

### OUTPUT

There are 1,680 traffic rows, network and scenario registries, point-model and
conformal tables, 60 route-risk rows, 15 decisions, 10 QAOA runs, structural
characteristics, exploratory correlations, source hashes, artifact hashes,
and a validation report in
`results/research/steps8_25_8_75_topology_benchmark/`.

### MATH

For topology family t, reported prediction metrics pool only held-out rows from
that family. The QAOA comparison reports analytic valid probability and exact
ground-state probability from the complete simulator distribution. Pearson
correlations are reported only when both variables have non-negligible
variance; a constant outcome has undefined correlation.

### IMPORTANT CODE

- `configs/topology_repeated_benchmark.json`: all 15 graph specifications.
- `experiments/run_steps8_25_8_75_topology.py`: end-to-end runner.
- `src/network_characterization.py`: OD-relevant structural measurements.
- `results/research/steps8_25_8_75_topology_benchmark/VALIDATION_REPORT.md`:
  accepted result and scope.

### RESULT

All 15 graphs were strongly and weakly connected and supplied all four
candidate routes. Feasible XY again spent all probability on one-hot states
and exceeded standard X optimal probability under the fixed compact budget.
Prediction error and interval width were worst on sparse mixed. These results
show repeated pipeline operation, not unseen-network transfer; training and
test rows still came from the same 15 graphs.

An initial correlation table was rejected because it correlated numerical
roundoff in the structurally constant XY-valid probability. The corrected
runner writes NaN plus `near_constant_qaoa_outcome`, and the invalid first
directory is preserved with suffix `SUPERSEDED_INVALID_CORRELATIONS`.

### IMPORTANT NUMBERS

- Point model: MAE 0.7570 min, RMSE 1.0653 min, R-squared 0.9566.
- Local conformal: 92.55% coverage at 90% nominal; mean width 3.1951 min.
- Risk variant D changed 3/15 choices; 0/15 conservative upper-bound
  deadline violations. These are model-based decisions, not observed arrivals.
- Feasible XY: valid 1.0000, optimal 0.3533, most-probable optimum 3/5.
- Standard X: valid 0.1515, optimal 0.0368, most-probable optimum 1/5.
- XY optimal probability by topology: sparse 0.4835, grid 0.3806,
  irregular 0.3520, dense 0.2931, radial 0.2575.

### WHAT I SHOULD SAY TO MY TA

“I repeated the full static pipeline on three different instances of five
road-like topology families. Every instance produced four routes. The XY
mixer remained feasible and had about 35.3% mean optimal probability versus
3.7% for standard X in a compact p=1 test. Sparse mixed was hardest for the
predictor but happened to give the highest XY optimal probability in its one
representative QAOA case. With only five quantum cases, that topology pattern
is exploratory and cannot support a statistical claim.”

### QUESTIONS MY TA MAY ASK

1. Does this prove cross-network generalization? No. Rows were held out, but
   every graph appeared in training. Step 9 separates whole networks.
2. Why only one QAOA graph per family? Simulator cost; this is a compact
   topology probe, not the final interaction study.
3. Why is XY validity not correlated with structure? It is exactly one by
   construction, so the correlation is undefined.
4. Are traffic distributions different only by topology? No. Every topology
   receives every regime, avoiding a topology/scenario confound.

### LIMITATIONS

- Three instances per family and one QAOA representative per family.
- Ideal simulator, K=4, p=1, one seed, and 12 iterations.
- Topology, node count, edge count, and route diversity co-vary.
- Traffic remains controlled simulation.
- No significance or causal interpretation of the structural correlations.

### NEXT STEP

Load several small cached OpenStreetMap/GraphML regions selected by measurable
structural contrast. Record extraction parameters and run fully offline after
the one-time download. Label all outcomes as real topology with controlled
simulated traffic.

## STEP 8.75 — NETWORK CHARACTERIZATION (COMPLETED 2026-08-10)

### WHAT WE BUILT

A table-ready characterization layer for structural measurements that may
plausibly explain routing and QAOA behavior. It records graph size, directed
edge count, intersection-degree distribution, edge-length/time variation,
OD shortest paths, K-route completeness, and route-edge diversity.

### WHY

Names such as “grid” and “irregular” are qualitative. The paper needs measured
structure so any optimization pattern can be compared with degree, time
heterogeneity, and route overlap rather than city labels.

### INPUT

The 15 Step 8.25 canonical graphs, each graph's fixed origin and destination,
and K=4 time-weighted loopless candidate routes.

### OUTPUT

`network_characteristics.csv`, topology aggregates, connectivity fields,
scenario parameters, and a small structure-by-QAOA correlation table. All 15
graphs have one weak and one strong component with largest-component fraction
1.0.

### MATH

Edge-time variation uses coefficient of variation CV = standard deviation /
mean. For route-edge sets E_i and E_j, Jaccard diversity is
1 - |E_i intersect E_j| / |E_i union E_j|. Directed density is
m/[n(n-1)], but it is explicitly labeled size-dependent and is never treated
as a scale-free road-complexity score.

### IMPORTANT CODE

- `src/network_characterization.py`: validated structural calculations.
- `tests/test_network_characterization.py`: OD and five-family tests.
- `network_characteristics.csv`: one auditable row per graph.
- `exploratory_structure_qaoa_correlations.csv`: defined/undefined flags.

### RESULT

The measurements distinguish the generated families and confirm K=4
completeness without expensive all-pairs computation. Some QAOA correlations
are numerically large, but each uses only five representatives and is
confounded by graph size and route structure. They are hypothesis generators.

### IMPORTANT NUMBERS

- Networks characterized: 15.
- K-complete: 15/15.
- Strongly connected: 15/15.
- QAOA representatives: 5 per method.
- All five XY-validity correlations: correctly undefined because validity is
  constant, not reported as numerical evidence.
- Largest descriptive coefficients include XY optimal probability versus
  route diversity r=0.883 and versus mean intersection degree r=-0.854.

### WHAT I SHOULD SAY TO MY TA

“I replaced topology names with measurable graph features. Route diversity is
one minus Jaccard edge overlap, and edge-time CV measures heterogeneity. A few
features line up with QAOA optimal probability in this tiny sample, but n=5
per method is too small and topology is confounded with size, so these are
exploratory observations only.”

### QUESTIONS MY TA MAY ASK

1. Why not average shortest paths over every node pair? It is expensive and
   less relevant to the fixed emergency OD task; Step 8.75 prioritizes
   decision-relevant measurements.
2. Is density comparable across graph sizes? Only cautiously; its caveat is
   saved beside the value.
3. Is r=0.883 statistically significant? No inference is made at n=5.
4. Why preserve undefined correlations? It documents that constant
   feasibility cannot explain variation.

### LIMITATIONS

- OD-focused rather than exhaustive all-pairs characterization.
- Synthetic families and small graph sizes.
- Correlations are unadjusted, underpowered, and confounded.
- No causal or universal topology claim.

### NEXT STEP

Apply the same schema and characterization code to cached real-road graphs,
then perform strict whole-network ID/OOD partitions in Step 9.

## STEP 8.5 — REAL NETWORK GENERALIZATION (COMPLETED 2026-08-10)

### WHAT WE BUILT

A generic Overpass/OpenStreetMap-to-canonical-GraphML adapter, a one-time cache
stage with endpoint failover, and an offline-only evaluator. Four compact road
extracts were selected for measured structural contrast: orthogonal grid,
radial hub, organic irregular, and mixed planned/radial. Place names exist only
in configuration and provenance data.

### WHY

Road-like synthetic generators cannot reproduce every feature of mapped street
structure. This step tests whether the exact same graph, predictor, conformal,
candidate-route, risk, QUBO, and QAOA interfaces operate on real topology
without adding city-specific branches.

### INPUT

- Four cached OpenStreetMap extracts with exact bounding boxes and queries.
- OSM base-road tags and speed limits/defaults converted into the canonical
  directed graph schema.
- The frozen Step 4 point model and Step 5 residual-scale/conformal calibration.
- The fixed 12.0819 km normalization learned from Step 4 training data; no
  real-test batch normalization.
- 384 newly simulated trips, 96 per graph, using controlled traffic regimes.
- K=4, risk variants A/D, and eight compact p=1 QAOA simulator runs.

### OUTPUT

Raw Overpass JSON, canonical GraphML, query/source timestamps, endpoint used,
ODbL attribution, hashes, network registry, characterization, model and
coverage tables, route-risk decisions, eight QAOA records, an offline guard
audit, manifest, and validation report.

### MATH

Mapped segment distance is computed geodesically from latitude/longitude.
Base time is 60 times distance_km divided by mapped or class-default speed_kph.
The model receives normalized distance d/12.0819 using the frozen training
scale. Local conformal applies the frozen calibration quantile to a frozen
validation-trained residual scale; it is evaluated, never recalibrated, on the
real-topology test rows.

### IMPORTANT CODE

- `src/osm_adapter.py`: location-neutral Overpass conversion and OD selection.
- `src/network_loader.py`: offline JSON/GraphML normalization and explicit
  parallel-edge projection provenance.
- `configs/real_topology_benchmark.json`: bounding boxes and selection reasons.
- `experiments/cache_step8_5_real_topologies.py`: one-time network cache.
- `experiments/run_step8_5_real_topology_offline.py`: guarded offline run.
- `results/research/step8_5_real_topology/VALIDATION_REPORT.md`: accepted scope.

### RESULT

The engineering generalization succeeded: every cached graph reloaded offline,
passed the schema, and supplied four routes. Statistical transfer failed. The
frozen synthetic-trained model had negative pooled R-squared and frozen 90%
local conformal intervals covered only 54.43%. This is “real road topology with
controlled simulated traffic,” not real traffic. The risk term changed no
route among the four paired A/D decisions.

Feasible XY remained exactly one-hot but did not always rank the optimum first.
Its mean optimal probability exceeded standard X, while standard X happened to
make the exact route most probable on more of these four cases. The sample is
too small for a topology claim.

### IMPORTANT NUMBERS

- Networks and sizes: orthogonal 145 nodes/181 edges; radial 590/915;
  organic 727/1,160; mixed planned/radial 1,568/2,279.
- K=4 completeness: 4/4.
- Frozen model: MAE 1.3432 min, RMSE 1.4857 min, R-squared -0.1866.
- Frozen local conformal: 54.43% coverage at 90% nominal.
- Per-network coverage: 47.92%–60.42%.
- Risk D changed 0/4 expected-time choices.
- Feasible XY: valid 1.0000, mean optimal 0.4328, most-probable optimum 2/4.
- Standard X: valid 0.4848, mean optimal 0.1267, most-probable optimum 3/4.
- Focused tests 36/36; full suite 128/128 at acceptance.

### WHAT I SHOULD SAY TO MY TA

“I cached four small OSM road graphs once and proved the experiment reruns with
network access disabled. The graph-independent software worked on all four,
but the frozen synthetic-trained uncertainty model did not transfer: 90%
intervals covered only 54.4%. That negative result is central evidence that
schema generality is not the same as statistical generalization. I call this
real road topology with simulated traffic, never real traffic.”

### QUESTIONS MY TA MAY ASK

1. Is the traffic real? No. Only the road topology is mapped; congestion and
   outcomes are controlled simulation.
2. Why is R-squared negative when MAE is moderate? On these narrow per-region
   outcome ranges, the frozen model is worse than the pooled mean baseline.
3. Why not recalibrate on each real graph? That would answer adaptation, not
   zero-shot transfer; this step deliberately freezes training/calibration.
4. Are parallel OSM roads preserved? The canonical schema is a DiGraph, so any
   source parallel edges are projected with counts, alternatives, and a clear
   caveat. Full-multigraph distance claims are rejected when projection matters.
5. Did XY solve every case? It always sampled a valid route, but its most
   probable valid route was optimal in only 2/4 cases.

### LIMITATIONS

- Four small bounding boxes selected for topology contrast, not population
  representativeness.
- OSM completeness, mapped speeds, and default-speed assumptions vary.
- Real topology but simulated traffic; no clinical or operational evidence.
- Frozen model trained only on five synthetic graphs.
- Four QAOA cases, one seed, p=1, ideal simulator.
- Canonical simple-graph projection cannot represent every parallel lane/road
  alternative or independent closure.

### NEXT STEP

Use strict network-level train, validation, calibration, TEST-ID,
TEST-OOD-topology, TEST-OOD-traffic, and combined OOD partitions. Measure point
error, coverage, route regret, deadline failures, and QAOA sampling separately
instead of treating the real-topology result as universal.

## STEP 9 — CROSS-NETWORK GENERALIZATION (COMPLETED 2026-08-10)

### WHAT WE BUILT

A fully network-separated benchmark with four train networks, four validation
networks, four calibration networks, four unseen TEST-ID networks from known
families, and four networks from an entirely held-out topology family. Two
additional evaluation views apply an unseen traffic/noise distribution to the
TEST-ID and held-out-topology networks. The same frozen model, preprocessing,
conformal models, route selector, QUBO, and QAOA code is used everywhere.

### WHY

Randomly splitting trips from the same networks can reward memorization of a
generator instance. Step 9 directly tests transfer to new graph instances,
unseen topology, traffic shift, and their combination. It also keeps point
prediction, interval calibration, route outcomes, and quantum sampling as
separate questions instead of calling one metric “generalization.”

### INPUT

- 20 distinct synthetic road networks: four instances from each of five
  topology families.
- 9,408 trip observations across base and shifted evaluation sets.
- Whole-network train, validation, calibration, TEST-ID, and OOD-topology
  partitions with ten pairwise zero-overlap checks.
- A held-out `irregular_geometric` family and a configured traffic/noise shift.
- K=4 route decisions, local conformal alpha=0.10 for the risk selector, and
  p=1 QAOA with 15 optimization iterations and 2,000 simulated shots.

### OUTPUT

Thirty result files: the effective configuration, partition and overlap
audits, frozen-normalization record, raw observations, model selection,
predictions, conformal calibration and interval tables, 64 outcome-backed
route decisions, 20 QAOA runs, summaries, saved models, source/artifact hashes,
and a validation report.

### MATH

The model's only distance normalization is

`normalized_distance = route_distance_km / max_train(route_distance_km)`.

The denominator 6.918166845 km is learned once from training networks, then
applied without clipping. Split conformal uses validation networks to learn a
residual-scale function and calibration networks to choose finite-sample
quantiles. At alpha=0.10, the target marginal coverage is 0.90 only under the
exchangeability conditions; distribution shift is expected to break that
guarantee. Route regret is selected actual time minus the fastest candidate's
actual time, computed after both selectors make their choices.

### IMPORTANT CODE

- `configs/cross_network_benchmark.json`: the complete, readable partition and
  shift design.
- `experiments/run_step9_cross_network.py`: guarded end-to-end runner.
- `results/research/step9_cross_network/VALIDATION_REPORT.md`: accepted result.
- `point_predictions.csv`, `conformal_intervals.csv`,
  `route_decisions.csv`, and `qaoa_runs.csv`: raw evidence tables.
- `manifest.json`: 29 artifact hashes and 17 transitive source hashes.

### RESULT

The predictor transfers moderately to new instances and the unseen synthetic
topology, but degrades sharply under traffic shift. Ninety-percent local
conformal coverage is 91.22% on TEST-ID and 90.33% on topology OOD, then fails
at 47.47% on traffic OOD and 50.37% on combined OOD. Calibrated-risk selection
has descriptively lower mean regret in all four groups but does not reduce the
deadline-violation rate in these 16-event-per-group samples. Feasible XY keeps
all probability on valid one-hot states and assigns much more probability to
the modeled optimum than standard X, but this compact ideal-simulator subset
does not support a speedup or population claim.

### IMPORTANT NUMBERS

- TEST-ID MAE/RMSE/R-squared: 1.1606 / 2.0755 / 0.8922.
- OOD-topology: 1.4990 / 2.6016 / 0.8622.
- OOD-traffic: 3.2740 / 4.6137 / 0.6745.
- OOD-combined: 3.9733 / 5.4153 / 0.6005.
- Local alpha=0.10 coverage, in the same order: 0.9122, 0.9033, 0.4747,
  and 0.5037.
- Deadline violations for expected/calibrated-risk selectors: both 81.25%
  TEST-ID, 75% topology OOD, and 100% in both shifted groups.
- Mean regret, expected versus calibrated risk: 0.631/0.565, 0.563/0.349,
  1.371/0.644, and 0.973/0.726 minutes.
- Across ten matched QUBOs, XY versus X analytic valid probability is
  1.0000 versus 0.2404; modeled-optimum probability is 0.4962 versus 0.0609.
- Full suite at acceptance: 128/128 tests.

### WHAT I SHOULD SAY TO MY TA

“Step 9 separates whole networks, not just rows. The frozen pipeline works
reasonably on unseen instances and one held-out synthetic topology, but the
configured traffic shift roughly triples MAE and destroys nominal conformal
coverage. Risk-aware choices lower regret descriptively without changing the
deadline-failure count. The strongest honest quantum result is feasibility and
sampling efficiency on a small simulator subset, not speed or operational
superiority.”

### QUESTIONS MY TA MAY ASK

1. Does TEST-ID contain known networks? No. It contains new networks from known
   generator families; training, validation, calibration, and TEST-ID network
   IDs are disjoint.
2. Why may OOD-traffic share network IDs with TEST-ID? It is an intentional
   evaluation view that isolates distribution shift on the same unseen graphs;
   it is not a training or calibration partition.
3. Why does conformal fail under shift? Ordinary split conformal needs
   calibration and test scores to be exchangeable. The traffic shift violates
   that assumption; wider intervals alone do not restore the guarantee.
4. Did uncertainty reduce emergency deadline failures? Not in this small
   experiment. It lowered mean regret descriptively, but both selectors had the
   same violation rate in every group.
5. Is XY universally better? It is exactly feasibility preserving here, but
   only ten QUBOs, one depth, one optimizer seed, and an ideal simulator were
   used. The larger Step 7.5 benchmark supplies the stronger comparison.

### LIMITATIONS

- Synthetic graphs, simulated traffic, and illustrative deadlines only.
- One held-out topology family and one configured traffic-shift preset.
- Sixteen decision events per group, insufficient for an advantage claim.
- QAOA subset has ten K=4 QUBOs, p=1, and no noise or hardware.
- The calibration failure is measured, not repaired with adaptive or
  shift-aware conformal methods.
- This is empirical cross-network evidence, never universal generalization.

### NEXT STEP

Repeat transfer evaluation with every network held out in turn. Within each
fold, learn preprocessing, select the model, fit residual scale, and calibrate
quantiles using only the other networks; then report every unfavorable fold
and aggregate the empirical cross-network result.

## STEP 9.1 — LEAVE-ONE-NETWORK-OUT (COMPLETED 2026-08-10)

### WHAT WE BUILT

A 20-fold leave-one-network-out experiment over all 20 Step 9 networks and all
five topology families. In every fold, one entire network is held out; the
other 19 are deterministically rotated into 9 training, 5 validation, and 5
calibration networks. Each validation and calibration split contains exactly
one network per topology family.

### WHY

One fixed train/test partition can be unusually easy or hard. LONO asks whether
transfer persists when every network becomes the unseen evaluation network in
turn. Whole-network roles also prevent trips from the held-out graph from
influencing normalization, model fitting, model choice, residual scale, or the
conformal quantile.

### INPUT

- 20 synthetic networks, four from each of five road-like families.
- 4,320 fixed trip observations, 216 per network.
- Twenty fold-local 9/5/5/1 train/validation/calibration/held-out assignments.
- Three candidate point models and absolute, relative, and local split
  conformal methods at a predeclared alpha=0.10.
- Fold-local training-distance normalization without clipping.

### OUTPUT

Twenty model-selection tables and fitted model bundles, twenty point-metric
rows, 60 conformal fold/method rows, all 4,320 held-out predictions, all 12,960
held-out intervals, topology summaries, explicit worst-fold tables, exact role
and overlap audits, configuration/source/artifact hashes, and a validation
report.

### MATH

For fold f, the divisor is learned only from its training networks:

`z_f = distance / max(distance in fold-f training networks)`.

The point model is fit on training and selected by validation RMSE. The local
scale model is fit on validation residuals. Calibration scores and the finite
sample quantile use only calibration networks. The held-out network is then
used once for evaluation. Pooled metrics weight every held-out observation;
macro metrics average the 20 fold values. The folds are dependent because the
same networks recur in fitting roles, so their standard deviation is
descriptive rather than an independent-sample confidence interval.

### IMPORTANT CODE

- `configs/leave_one_network_out_benchmark.json`: fixed fold protocol.
- `experiments/run_step9_1_leave_one_network_out.py`: complete runner and hard
  leakage assertions.
- `fold_split_audit.csv`: role IDs, overlap checks, and exclusion hashes.
- `point_metrics_by_fold.csv` and `conformal_metrics_by_fold.csv`: every fold.
- `results/research/step9_1_leave_one_network_out/VALIDATION_REPORT.md`:
  accepted result with unfavorable folds retained.

### RESULT

Across one held-out evaluation of every network, pooled MAE is 0.893 minutes
and local 90% intervals cover 90.05% with the narrowest mean width of the three
methods. Transfer is heterogeneous: radial networks are easiest, sparse mixed
networks are hardest, and aggregate calibration hides material fold failures.
For example, the sparse-mixed calibration-labelled instance has RMSE 3.656 and
only 69.91% absolute coverage. The original Step 9 labels are only metadata in
this experiment; every network rotates through all four fold roles.

### IMPORTANT NUMBERS

- Folds and held-out networks: 20 and 20.
- Every fold: 9 train, 5 validation, 5 calibration, 1 held out.
- Maximum network overlap and observation overlap: both zero.
- Pooled held-out MAE/RMSE/R-squared: 0.8930 / 1.6375 / 0.9421.
- Macro-fold RMSE: 1.4526 ± 0.7756 minutes.
- Absolute coverage/width: 0.8965 / 4.5590 minutes.
- Relative coverage/width: 0.8986 / 4.1398 minutes.
- Local coverage/width: 0.9005 / 3.7221 minutes.
- Topology RMSE: radial 0.5808, grid 0.9195, dense mixed 1.5172,
  irregular geometric 1.8930, sparse mixed 2.5177 minutes.
- HGB selected in 19/20 folds; random forest in 1/20.
- Twenty-one artifact hashes and twelve source hashes verified.
- Full suite at acceptance: 128/128 tests.

### WHAT I SHOULD SAY TO MY TA

“Every network is held out once, and each fold relearns preprocessing and
calibration from the other networks. Overall local conformal coverage is almost
exactly 90%, but that aggregate is not a universal guarantee: sparse and dense
instances contain clear undercoverage. The result supports empirical transfer
on these five synthetic families while also showing why per-network reporting
is essential.”

### QUESTIONS MY TA MAY ASK

1. Are the 20 folds independent replications? No. They share fitting networks,
   so pooled and macro metrics are descriptive cross-validation evidence.
2. Can the held-out network affect normalization? No. Each divisor uses only
   the nine fold-training networks; held-out values above one are not clipped.
3. Why have separate validation and calibration roles? Validation chooses the
   point model and learns residual scale; calibration alone determines the
   conformal quantile.
4. Why is pooled coverage near 90% when some folds fail? Overcoverage on easier
   networks offsets undercoverage on harder networks. Marginal aggregation can
   hide conditional failures.
5. Does this prove city-level generalization? No. It is empirical transfer
   across 20 simulated networks from five configured generator families.

### LIMITATIONS

- Only four networks per topology family and one traffic generator.
- Reused fitting networks make folds statistically dependent.
- Topology, network size, density, and base-time multiplier remain confounded.
- No traffic-distribution shift is introduced inside this LONO experiment;
  Step 9 supplies that separate stress test.
- No route decision, QAOA, hardware, medical, or operational claim follows
  from these prediction and coverage folds.

### NEXT STEP

Use matched evidence to separate three questions: how uncertainty changes the
modeled route choice, how topology and QAOA depth affect feasible/optimal
sampling, and whether distribution shift simultaneously worsens prediction,
coverage, and realized deadline outcomes. Keep those evidence blocks separate
rather than pooling incompatible populations.

## STEP 9.2 — HISTORICAL PRE-RUN ENTRY (SUPERSEDED)

### WHAT WE BUILT

A draft configuration and runner for a paired interaction study, plus an
explicit then-current `INCOMPLETE.md` status marker. That marker was removed
after the replacement full study passed and `COMPLETE.md` became authoritative.

### WHY

The intended analysis would ask whether adding uncertainty changes the exact
route optimum before comparing how the two QAOA mixers respond.

### INPUT

Planned matched candidate sets, deadlines, risk variants, mixers, and fixed
seeds. No accepted execution population exists.

### OUTPUT

No result CSV, validation report, manifest, or paper number was produced.

### MATH

The proposed contrast holds the point/deadline terms fixed and compares a
score without uncertainty to the corresponding score with uncertainty, then
uses the exact argmin as the reference for paired QAOA cells.

### IMPORTANT CODE

- `configs/topology_uncertainty_qaoa_interaction.json`: draft protocol.
- `experiments/run_step9_2_topology_uncertainty_qaoa.py`: unexecuted runner.
- `results/research/step9_2_topology_uncertainty_qaoa/COMPLETE.md`: current status.

### RESULT

Incomplete. There are no numerical Step 9.2 results.

### IMPORTANT NUMBERS

Accepted runs: zero. Reportable results: zero.

### WHAT I SHOULD SAY TO MY TA

“At this historical checkpoint, the first Step 9.2 draft had been stopped and
excluded. The later replacement matched design supersedes this entry.”

### QUESTIONS MY TA MAY ASK

1. Why not show preliminary values? There is no complete, reconciled result
   population, so preliminary values would not meet the project’s evidence
   standard.
2. Does this invalidate Step 9.4? No. Step 9.4 is a separately completed and
   validated robustness matrix.

### LIMITATIONS

No interaction conclusion can be drawn, and the draft must not be cited as an
experiment.

### NEXT STEP

Keep this analysis deferred until after the paper scope is agreed. If run, use
predeclared paired cells and verify exact route-optimum changes first.

## STEP 9.3 — CLASSICAL BASELINES (COMPLETED 2026-08-10)

### WHAT WE BUILT

Deterministic candidate-set and graph-wide classical selectors with stable tie
handling, exact-risk ground truth, per-instance results, and topology/K
summaries.

### WHY

Quantum-method metrics are uninterpretable without strong classical route
selection baselines and a clear distinction between candidate and graph-wide
search spaces.

### INPUT

Fifteen decisions across five topology families and K=4,6,8, with the accepted
Step 6 risk vectors.

### OUTPUT

Ninety method-instance rows, tie checks, 4,500 engineering timing repetitions,
validation checks, and a hash manifest in
`results/research/step9_3_classical_baselines/`.

### MATH

Exact-choice rate compares each method’s selected candidate with the exact
risk argmin. Regret is selected risk minus minimum candidate risk. Graph-wide
methods are reported separately because their feasible set differs.

### IMPORTANT CODE

- `src/classical_baselines.py`: selectors and trace records.
- `experiments/run_step9_3_classical_baselines.py`: accepted benchmark.
- `baseline_summary.csv`: main comparison.

### RESULT

Exact-risk selection recovers 15/15 exact solutions. Candidate expected-time
and minimum upper bound recover 12/15 with mean regret 0.2421. Candidate
shortest distance recovers 6/15 with regret 7.7996. Graph-wide base-time
Dijkstra recovers a comparable candidate in 9/15 with regret 7.3094. All 90
selections are stable under reordered inputs.

### IMPORTANT NUMBERS

- Decisions: 15; method rows: 90.
- Exact-risk: 15/15, zero regret, zero deadline proxies.
- Candidate expected-time and minimum upper bound: 12/15 each.
- Candidate distance: 6/15, seven deadline proxies.
- Graph base-time Dijkstra: 9/15, six deadline proxies.
- Tied rows recorded: six.

### WHAT I SHOULD SAY TO MY TA

“The exact classical risk selector is the ground truth for these small
candidate sets. QAOA is studied for feasibility and sampling behavior, not
because the classical problem is hard at this scale.”

### QUESTIONS MY TA MAY ASK

1. Are graph-wide and candidate methods equivalent? No. Their agreement on
   some instances is corpus-specific.
2. Can timing repetitions show speedup? No. The workload and implementations
   are not a fair performance benchmark.

### LIMITATIONS

Only 15 decisions are included, the model is not integrated into arbitrary
graph-edge weights, and distance projection choices can matter on parallel
road edges.

### NEXT STEP

For future work, align learned edge costs with graph-wide objectives and expand
the decision corpus before making comparative optimization claims.

## STEP 9.4 — ROBUSTNESS AND ABLATION (COMPLETED 2026-08-10)

### WHAT WE BUILT

Isolated risk-term removals, alpha and weight sweeps, prediction-noise and
coverage stress tests, a QUBO penalty sweep, eight missing p=3 K=6/8 QAOA
cells, and 175 paired dynamic ablation runs.

### WHY

The integrated pipeline needs failure boundaries, not only best-case averages.
This step tests which terms actually change decisions and where the encoding
or dynamic policy becomes brittle.

### INPUT

Fifteen static decisions, K=4/6/8 QUBOs, inherited 106 QAOA runs plus eight
targeted cells, and five dynamic conditions over five networks and seven event
rotations.

### OUTPUT

Twenty-six result tables plus an index, validation checks, and a manifest with
29 output and 22 source hashes in
`results/research/step9_4_robustness_ablation/`.

### MATH

The QUBO sweep varies P around the minimum candidate cost. Paired dynamic
deltas compare each ablation with the full objective on the same
network/event rotation.

### IMPORTANT CODE

- `configs/robustness_ablation_benchmark.json`: fixed matrix.
- `experiments/run_step9_4_robustness_ablation.py`: accepted runner.
- `qubo_penalty_aggregate.csv` and `dynamic_ablation_summary.csv`: boundaries.

### RESULT

Only removing the reroute penalty changes static choices (3/15). QUBO
penalties below the minimum candidate cost yield an invalid zero ground state;
equality creates a 50/50 valid/invalid ground-state tie; tested penalties just
above the boundary restore validity. At p=3, feasible-XY validity is 1.000 for
K=6/8 versus 0.0363/0.0229 for standard-X. Removing the reroute penalty raises
mean reroutes from 0.4571 to 0.8571 without changing deadline violations.

### IMPORTANT NUMBERS

- Total QAOA runs reconciled: 114 = 106 inherited + 8 targeted.
- Paired dynamic runs: 175.
- Deadline violations: 1/35 in every dynamic condition.
- Repeated-topology local coverage: 0.7925 on 53 rows.
- All artifact and source hashes matched at acceptance.

### WHAT I SHOULD SAY TO MY TA

“The ablation shows two separate mechanisms: the algebraic penalty controls
one-hot validity, while the reroute penalty controls route churn. Uncertainty
and deadline terms were mostly inactive at these score margins.”

### QUESTIONS MY TA MAY ASK

1. Is 1.01 times the minimum always sufficient? No; it worked on the tested
   finite instances. The conservative accepted rule remains max cost plus one.
2. Did fewer reroutes improve deadlines? No. The paired deadline count was
   unchanged.
3. Can every p=3 row be reconstructed from stored angles? The eight targeted
   rows did not save optimized angles; their metrics and hashes are valid, but
   angle-level reproduction needs a deterministic rerun.

### LIMITATIONS

Small populations, ideal simulation, unequal K replication, no stored angles
for eight targeted rows, and no completed warm-start baseline.

### NEXT STEP

Do not expand the sweep for the paper. Carry the validated boundaries into the
claims and hardware design.

## STEP 9.5 — PAPER-READY PACKAGE (COMPLETED 2026-08-10)

### WHAT WE BUILT

Fourteen manuscript sections, eight traceable figures, experiment/figure/table
indexes, a final status, learning guide, presentation script, demo guide, and
Step 10 hardware plan.

### WHY

The research now benefits more from preservation, interpretation, and exact
claim boundaries than from additional experiment scope.

### INPUT

Only validated artifacts from Prototype 0.1 and Steps 4, 5, 6/6.5, 7, 7.5,
8, 8.25/8.75, 8.5, 9, 9.1, 9.3, and 9.4. Step 9.2 contributes only its
incomplete status.

### OUTPUT

- Manuscript sections in `paper/`.
- Evidence matrices in `paper/evidence/`.
- Static figures in `paper/figures/`.
- Root presentation, demo, learning, status, and index documents.

### MATH

No new estimator or experiment was introduced. Figures and prose reconcile
stored values for prediction, coverage, route risk, QUBO feasibility, QAOA
sampling, and dynamic outcomes.

### IMPORTANT CODE

- `experiments/build_step9_5_figures.py`: validated-table-only plot builder.
- `paper/RESULTS.md`: numerical narrative.
- `paper/CLAIMS.md`: supported and prohibited claims.
- `FINAL_RESEARCH_STATUS.md`: handoff summary.

### RESULT

The project is paper-ready as a reproducible prototype and negative-results
study. It is not submission-ready as a hardware, real-traffic, or operational
validation paper.

### IMPORTANT NUMBERS

- Manuscript files requested: 14/14 present.
- Step 9.5 figures: eight.
- Literature master matrix after the final literature update: 33 primary papers.
- Incomplete Step 9.2 numerical results included: zero.
- Final clean suite: 128/128 tests passed in 13.53 seconds, with bytecode,
  third-party plugin autoload, and pytest cache disabled.

### WHAT I SHOULD SAY TO MY TA

“I stopped expanding the scope and converted every accepted result into a
traceable paper package. The strongest result is feasibility from the XY mixer;
the strongest scientific evidence is the set of transfer, calibration, and
dynamic failures that bound what we can claim.”

### QUESTIONS MY TA MAY ASK

1. What is the current exact step? See the top/final freeze entries; Step 10
   hardware is not started.
2. What is the paper’s novelty claim? An integrated, traceable evaluation
   framework and matched feasibility/robustness evidence, phrased as potential
   contribution pending venue review.
3. What must happen before submission? Venue formatting, citation integration,
   version/environment freezing, advisor review, and preferably observed
   traffic or clearly scoped future-work language.

### LIMITATIONS

Markdown is not yet typeset in a venue template; no Git commit provenance or
locked container exists; no external replication has been performed.

### NEXT STEP

Review the claims with the TA, select the venue, freeze the repository, and
decide whether Step 10 hardware or observed-traffic validation is required
before submission. Do not start additional sweeps merely to fill tables.

---

## FINAL SCIENCE FREEZE — COMPLETED 2026-08-13

This entry supersedes the earlier publication-freeze entry above, whose
Step 9.2 and test-count statements were correct at that earlier checkpoint but
are no longer the final repository state.

### WHAT WE BUILT

- Completed the predeclared Step 9.2 matrix: 1,800/1,800 accepted trials over
  five topology families, three uncertainty levels, K=4/8, p=1/3/5, X/XY, and
  ten seeds; 300 problems, 50 networks, and 900 matched mixer pairs.
- Added cluster-aware bootstrap/sign-flip inference, topology-mechanism
  diagnostics, online adaptive conformal replay, a shot-based local-noise
  ablation, a targeted K=16 scaling study, and a held-out standard-X penalty
  sensitivity study.
- Rebuilt 17 traceable figures, the final table package, a 33-primary-paper
  evidence matrix, a 14-section generated manuscript, abstract, claims ledger,
  reproducibility record, final freeze report, and machine-verifiable manifest.

### WHY

The final work tests the strongest plausible alternative explanations for the
negative result: topology/uncertainty confounding, static calibration under
shift, ideal-statevector optimism, small-K scaling, and an unfair standard-X
penalty. None reverses the scientific conclusion.

### IMPORTANT NUMBERS

- XY minus X feasibility: +0.820436, cluster-bootstrap 95% CI
  [0.813327, 0.827442]; optimum probability: +0.386714
  [0.354827, 0.417694]. All primary matched effects remain significant after
  Holm correction (adjusted p=7.99992e-5).
- Within-network entropy association: optimum probability Spearman rho
  -0.731930 [-0.801157, -0.652508]. This is association, not causation.
- Adaptive conformal coverage under traffic/combined shift is 0.8430/0.8460,
  but mean interval width expands to 14.909/17.957 and its selected routes are
  identical to the local conformal baseline on all 64 method events.
- Under the declared phenomenological local-noise model, XY noisy-minus-ideal
  feasibility is -0.24944 at p=1 and -0.42915 at p=3; optimum mass also falls.
- K=16 XY optimum probability is 0.104401 at p=1 and 0.122348 at p=3. The
  matched K=16 minus K=8 effect is -0.141319 [-0.163381, -0.118350].
- The penalty validation procedure selects the existing safe default. On held-
  out problems its X feasibility/optimum probability is 0.051334/0.006646;
  the largest means among the algebraically valid declared settings are only
  0.068122/0.011040. The complete seven-condition grid is still reported;
  penalty tuning does not rescue standard X.
- Step 9.2 checkpoint span: 64.31 minutes; cumulative per-trial optimizer wall
  time: 9,891.64 seconds. K=16 span: 2,158.166 seconds; cumulative trial wall:
  6,533.282 seconds. Penalty span: 960.590 seconds; cumulative trial wall:
  2,901.638 seconds.
- Final repository gate: 158/158 tests must pass, all local links and stored
  hashes must validate, and all 33 literature rows must be present before
  `SCIENCE_FREEZE_MANIFEST.json` is written.

### EXECUTION INCIDENTS RETAINED IN THE RECORD

- An early Step 9.2 launch produced duplicate live workers, which were stopped;
  accepted final checkpoints remain unique and complete.
- The first penalty launch rejected 42 attempts before acceptance because
  non-finite JSON values violated strict serialization. The serializer was
  corrected and regression-tested; all 42 failure records are preserved and
  hashed, while the final 350-trial design has zero accepted failures.
- Step 9.7 exposed a CSV decimal-replay precision defect. Replaying the original
  source generator reproduces static intervals to 7.105e-15; no accepted source
  experiment was altered.

### RESULT

The project supports a careful negative-results paper: constraint-preserving XY
prevents infeasible leakage, but deeper circuits do not create reliable optimum
concentration, ideal simulation does not establish quantum advantage, adaptive
intervals do not change routing decisions here, local noise worsens the quantum
diagnostics, K=16 strengthens the adverse scaling trend, and standard-X penalty
tuning does not close the gap. The routing instances remain exactly and
classically tractable.

### WHAT I SHOULD SAY TO MY TA

“I completed the full predeclared matrix and then tried to falsify the negative
result with cluster-aware inference, shift adaptation, noise, K=16, and held-out
penalty tuning. The feasibility advantage of constraint-preserving XY is robust,
but there is no quantum advantage, speedup, hardware result, or evidence that
QAOA improves routing. The honest contribution is a traceable diagnostic and
negative-results study.”

### REMAINING LIMITATIONS AND NEXT STEP

Traffic is synthetic, candidate sets are small and enumerable, the noise model
is not backend-calibrated, dynamic evidence spans five network clusters, and the
workspace lacks Git/container provenance and independent replication. Freeze
the accepted artifact set for advisor/venue review. Step 10 hardware is useful
only as a small, predeclared feasibility/concentration test—not as a rescue or
speedup claim. Observed-traffic validation or independent replication may be
the higher-value next study.
