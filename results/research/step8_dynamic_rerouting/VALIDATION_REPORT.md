# Step 8 validation report

## Overall assessment: Valid negative benchmark result

All 140 matched simulations completed: five networks, seven rotating first
events, and four policies. The QAOA-assisted policy used an actual p=1
feasible-XY ideal-simulator optimization at every nontrivial decision. It was
not replaced by an exact classical selector.

## Event and policy scope

Events: normal, rush hour, rain, localized congestion, accident, road closure,
and prediction noise. Rotating the sequence makes each event occur first once
per network. Each decision traverses one edge, then updates the graph and
replans from the current node.

Policies:

- never reroute: follows the base shortest path and fails if its next edge is
  closed;
- expected-time rerouting: minimizes the current point prediction;
- uncertainty-aware classical: exact argmin of the full risk score;
- uncertainty-aware QAOA: most probable valid route from p=1 feasible XY after
  12 Adam iterations.

## Aggregate result

| Policy | Arrival rate | Deadline violation rate | Mean arrival (arrivals only, min) | Mean reroutes | Mean unnecessary reroutes |
|---|---:|---:|---:|---:|---:|
| Expected-time rerouting | 100% | **2.86%** | 14.03 | 0.89 | 0.34 |
| Never reroute | 85.71% | 14.29% | 13.79 | 0.00 | 0.00 |
| Uncertainty-aware classical | 100% | 5.71% | 14.17 | 0.46 | 0.23 |
| Uncertainty-aware QAOA | 100% | 17.14% | 15.71 | 1.43 | 0.83 |

Never-reroute mean arrival time excludes its five closure failures and is not
directly comparable without its arrival rate.

## Quantum decision audit

- 264 saved QAOA decision records.
- Analytic valid probability remained 100% within numerical tolerance.
- Mean optimal-route probability: 56.00%; median 60.80%.
- Most probable route matched exact risk argmin in 85.23% of decisions.
- Sparse mixed was the failure concentration: QAOA deadline violations 4/7
  and mean five reroutes, versus zero violations for the two classical
  rerouting policies on that topology.

This shows why feasibility probability and route-quality probability must be
reported separately.

## “Unnecessary reroute” definition

A reroute is marked unnecessary when the old plan is still feasible and has no
higher remaining expected time under the current event-adjusted graph. It is a
current-event oracle diagnostic, not a claim about unknowable future traffic.

## Required caveats

- One event/state seed, one network instance per topology, and controlled
  simulated traffic.
- Deadlines are 1.8 times base shortest-path time, not clinical thresholds.
- p=1 and 12 optimization iterations intentionally make the dynamic quantum
  selector lightweight; they are not guaranteed sufficient.
- The risk weights did not improve deadline reliability here. Expected-time
  rerouting was the strongest policy in this fixed benchmark.
- Arrival-time means exclude failed trips; use deadline violation and arrival
  rate together.
- No quantum speed or hardware claim.

## Decision

Keep the failure as evidence. Proceed to repeated synthetic topology instances
and distribution-shift studies before drawing any reliability conclusion.

