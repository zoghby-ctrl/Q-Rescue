# Step 7.5 validation report

## Overall assessment: Ready for controlled simulator comparison

The benchmark completed all 106 preregistered runs. It compared standard
penalty X-mixer QAOA with a uniform-one-hot, ring-XY feasibility-preserving
ansatz using the same verified cost Hamiltonians. Results are ideal statevector
simulations plus separate 2,000-shot multinomial estimates; no hardware ran.

## Scope

- Primary: five K=4 topology instances, p=1/2/3, three optimization seeds, two
  methods = 90 runs.
- Scaling: one fixed grid instance at K=6/8, p=1/2, two seeds, two methods =
  16 runs.
- Adam: 35 iterations, step size 0.05.
- Exact classical route optimum retained for every run.

## Primary K=4 result

| p | Method | Mean valid probability | Mean optimal probability | Exact route most probable |
|---:|---|---:|---:|---:|
| 1 | standard X | 34.69% | 7.02% | 2/15 |
| 1 | feasible XY | 100.00% | 49.74% | 15/15 |
| 2 | standard X | 31.87% | 6.40% | 4/15 |
| 2 | feasible XY | 100.00% | 62.15% | 15/15 |
| 3 | standard X | 27.81% | 6.83% | 3/15 |
| 3 | feasible XY | 100.00% | 60.59% | 13/15 |

The feasibility result is structural: W-state initialization and the XY mixer
remain in the one-hot subspace. It is not an empirical speedup claim. The
optimal-probability comparison is specific to this small fixed benchmark and
optimization budget.

## K scaling pilot

- K=6: XY valid 100% at p=1/2; mean optimal probability 20.68%/33.07%.
  Standard X valid 10.73%/19.11%; optimal 2.10%/2.52%.
- K=8: XY valid 100% at p=1/2; mean optimal probability 21.94%/26.78%.
  Standard X valid 2.65%/4.14%; optimal 0.74%/0.58%.

Only two seeds and one topology were used in the K=6/8 pilot, so these are
scaling diagnostics, not topology-general conclusions.

## Topology and depth observations

At K=4/p=2, XY mean optimal probability ranged from 53.35% (irregular) to
70.25% (radial). Seed variability was material: radial p=3 standard deviation
was 28.02 percentage points. Increasing p did not consistently help: aggregate
XY optimal probability rose from p=1 to p=2, then declined slightly at p=3
under the fixed 35-iteration optimizer budget.

## Resource accounting

PennyLane logical/device-expanded resource records were saved. At K=4,
standard-X circuit depth was 8/14/20 for p=1/2/3 and XY depth was 12/22/32.
The XY records include `StatePrep` as one logical operation; they therefore
understate hardware-native W-state preparation cost and must not be presented
as backend gate counts.

## Finite-shot validation

All sampled probabilities use 2,000 shots per run with recorded seeds. Maximum
absolute analytic-vs-sampled differences were 3.20 percentage points for valid
probability and 2.81 points for optimal probability, consistent with finite
sampling variability at this scale.

## Warm-start status

Warm-start QAOA was not implemented. A defensible implementation requires a
specified continuous relaxation and its corresponding modified mixer/state.
Initializing from the exact enumerated optimum would leak the answer and would
not be a valid warm-start experiment. This remains an explicit unfinished
method, not a zero or negative result.

## Required caveats

- Ideal simulation only; no device noise, compilation, or hardware queue.
- Small K and few seeds; no quantum advantage or runtime claim.
- The two methods use different feasible-search priors (uniform all states vs
  uniform one-hot), which is part of the intended constraint-handling comparison.
- Optimizer budget is fixed, so p=3 may need more iterations; no lucky seed was
  selected or reported alone.
- XY circuit resources omit native decomposition cost of state preparation.

