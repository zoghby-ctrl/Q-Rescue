# Steps 6 and 6.5 validation report

## Overall assessment: Ready for generalized QUBO construction

The candidate generator returned exactly K=4, 6, and 8 loopless routes for
every one of the five networks. Every route was transformed through the same
feature model and locally scaled conformal interval. The A–E objective was
recomputed independently from saved components with maximum numerical error
1.42e-14.

## Objective definition

All components have minute or minute-equivalent units:

`risk = prediction + lambda_u*half_width + lambda_d*max(0, upper-deadline)
        + lambda_r*reroute_minutes`

The accepted weights are lambda_u=1, lambda_d=2, and lambda_r=1. Variant A
keeps prediction only; B adds uncertainty; C adds deadline exceedance; D adds
both; E additionally adds rerouting cost.

The deadline in this controlled benchmark is the 40th percentile of candidate
upper bounds for each network/K decision. This normalizes deadline tightness
across very different network travel-time scales. It is an experimental
difficulty definition, not a clinical deadline.

## Candidate-route checks

- 90 route rows in 15 decision sets: each network has K=4, 6, and 8 exactly.
- All numeric inputs are finite.
- Route IDs follow x_0 through x_(K-1) in deterministic shortest-simple-path
  order.
- Mean directed-edge Jaccard diversity by topology ranges from 0.495
  (irregular geometric) to 0.787 (grid), showing that candidate-set structure
  differs by topology.

## Ablation result

| Variant | Decisions | Changed from A | Conservative violations | Mean margin (min) |
|---|---:|---:|---:|---:|
| A expected time | 15 | 0 | 0 | 2.073 |
| B + uncertainty | 15 | 0 | 0 | 2.073 |
| C + deadline | 15 | 0 | 0 | 2.073 |
| D + both | 15 | 0 | 0 | 2.073 |
| E + reroute cost | 15 | 3 | 0 | 2.015 |

This fixed set does **not** show a benefit from uncertainty or deadline terms:
the expected-time winner was also the best risk route in all 15 cases. Only
the explicit switching cost changed three decisions. This negative result is
preserved; weights or seeds were not changed to manufacture a favorable case.

## Required caveats

- The traffic states and deadlines are controlled synthetic inputs.
- A conservative upper-bound violation is not an observed late arrival.
- One state seed and one instance per topology are insufficient for a paper
  conclusion about route changes or reliability.
- The rerouting term assumes x_0 is the current route in this static ablation;
  its operational meaning is tested later in the dynamic simulator.

## Decision

Proceed to Step 7. Use each saved arbitrary-length risk vector directly and
verify every K=4/6/8 QUBO state by exhaustive enumeration.

