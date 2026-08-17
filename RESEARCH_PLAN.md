# Q-Rescue Research Plan

## Problem statement

Emergency routing based only on mean travel time can prefer a route whose arrival time is highly uncertain. This prototype asks an ambulance to choose exactly one of four candidate routes while accounting for both expected travel time and manually simulated uncertainty.

For route `i`, the prototype uses

```text
risk_score_i = expected_time_i + uncertainty_weight * uncertainty_i
```

with `uncertainty_weight = 1.0`. This is a transparent teaching score, not a clinical risk model.

## Main research question

Can a small risk-aware emergency-route choice be represented transparently as a QUBO, converted manually to an Ising Hamiltonian, and explored with a reproducible PennyLane QAOA workflow whose output is checked against exhaustive classical search?

The milestone evaluates correctness and explainability. It does not evaluate speedup.

## Proposed future contribution

A future Q-Rescue system may combine calibrated traffic prediction intervals with risk-aware optimization and dynamic rerouting. The intended research contribution would be an auditable connection between calibrated predictive uncertainty and route decisions, followed by careful classical, simulated-quantum, noisy-quantum, and possibly hardware comparisons.

This is a proposal for later investigation, not a novelty or performance claim.

## Current prototype boundaries

Included now:

- one ambulance and one hospital;
- four pre-enumerated candidate routes;
- a six-node synthetic directed graph;
- fixed expected times and manually simulated uncertainty values;
- one linear risk score with weight 1.0;
- one exactly-one constraint with penalty 50;
- exhaustive search over 16 states;
- a noiseless four-qubit PennyLane simulator;
- standard QAOA depths 1 and 2;
- analytic training and finite-shot sampling.

Explicitly out of scope:

- learned traffic prediction;
- calibrated prediction intervals;
- live or historical traffic feeds;
- dynamic rerouting;
- multiple emergency vehicles;
- vehicle-to-incident assignment;
- realistic map ingestion;
- congestion interaction;
- quantum-noise experiments;
- error mitigation;
- quantum hardware execution;
- application dashboards or external APIs.

## Important prototype decisions

### Data and graph

- Route expected times are 18, 21, 24, and 20 minutes for A through D.
- Uncertainty values are manually simulated as 10, 2, 3, and 7 minutes.
- Each route is a separate two-edge branch from Ambulance Station through its named junction to Hospital.
- Edge times sum exactly to the stated expected route times.

### Risk model

- The uncertainty weight is 1.0 so every uncertainty minute adds one minute-equivalent unit to the score.
- The expected-time baseline is retained as the first comparison.
- No probabilistic interpretation is assigned to the risk score in this milestone.

### Binary encoding and QUBO

- `x_0`, `x_1`, `x_2`, and `x_3` represent Routes A, B, C, and D in that order.
- A bitstring is displayed in that same order. For example, `0100` means Route B. Bits are never reversed.
- The constraint penalty is 50. It makes the zero-route cost 50, above every valid route score, and adds at least 50 to every multiple-route assignment.
- The expanded QUBO has constant 50, linear coefficients `[-22, -27, -23, -23]`, and pairwise coefficients 100.
- The readable symmetric matrix places half of each pairwise coefficient in each mirrored off-diagonal cell. Thus `x^T Q x` reconstructs the full pair coefficient.

### Ising and QAOA

- The substitution is implemented manually as `x_i = (1 - Z_i) / 2`.
- The Hamiltonian omits the identity term and returns its known constant offset separately. Adding that offset makes every computational-basis energy equal its QUBO cost.
- PennyLane wire 0 maps to `x_0`, wire 1 to `x_1`, wire 2 to `x_2`, and wire 3 to `x_3`.
- The device is `qml.device("default.qubit", wires=4)`.
- The circuit begins with one Hadamard gate per wire, then alternates PennyLane cost and Pauli-X mixer layers.
- Training uses an analytic expectation-value QNode. Sampling uses a separate QNode with 4,000 shots.
- QAOA depths 1 and 2 are trained with PennyLane's Adam optimizer for 100 iterations at step size 0.02.
- The fixed starting angles are gamma 0.02 and beta -0.20 in every layer.
- The sampling seed is 2026. Fixed values make the notebook repeatable; they do not guarantee that a route must lead in another run.
- The APIs were checked locally against installed PennyLane 0.45.1. Shots are set on the sampling QNode, following the supported API rather than the deprecated device-level setting.

## Planned milestones

1. **Transparent four-route prototype — current.** Verify the graph, score, QUBO, Ising conversion, classical optimum, and small QAOA experiment.
2. **Calibrated predictive uncertainty.** Train a simple traffic-time model on an appropriate offline dataset, produce prediction intervals, and measure empirical coverage and width.
3. **Time-dependent routing.** Update scores as traffic estimates change and compare periodic rerouting policies.
4. **Larger routing and dispatch.** Introduce more candidate paths and multiple emergency vehicles with carefully defined classical baselines.
5. **Quantum robustness study.** Compare exact simulation, shot noise, controlled device-noise models, and mitigation methods.
6. **Optional hardware study.** Run only suitably small circuits on available hardware and report queue, noise, and reproducibility limits.

Only milestone 1 is implemented in this repository.

## Baselines

- **Expected-time baseline:** select the smallest expected travel time; this chooses Route A.
- **Risk-aware enumeration:** calculate every route risk score directly; this chooses Route B.
- **Brute-force QUBO solver:** evaluate all 16 assignments; this is the correctness reference and chooses `0100`, Route B.
- **Uniform sampling reference:** before useful QAOA concentration, a uniform four-qubit state gives valid probability `4 / 16 = 25%` and optimal-route probability `1 / 16 = 6.25%`.

## Metrics

- expected travel time of the selected route;
- risk score of the selected route;
- QUBO cost and constraint validity;
- analytic cost-Hamiltonian expectation during training;
- final count of every four-bit string;
- valid-solution probability;
- optimal-route probability, where Route B is the exhaustive optimum;
- most frequently measured valid route;
- agreement of QUBO and Hamiltonian energies up to the returned offset.

## Claims we are not making

- No quantum advantage, speedup, or scaling benefit is claimed.
- No publication novelty is claimed.
- No medical, dispatch, or operational recommendation is made.
- The synthetic uncertainty values are not calibrated or learned.
- A lower risk score is not claimed to be a real-world probability of a safer outcome.
- QAOA is not claimed to return the optimum on every run.
- Simulator results are not claimed to predict hardware performance.
- Results from four routes are not claimed to generalize to realistic cities.

