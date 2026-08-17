# Final Learning Guide

## The project in one sentence

Q-Rescue asks whether traffic prediction, calibrated uncertainty, risk-aware route selection, and constraint-preserving QAOA can form a defensible emergency-routing research pipeline—and records where that pipeline fails.

## The conceptual chain

1. Predict each candidate route’s travel time.
2. Calibrate uncertainty using data not used to fit or select the model.
3. Convert prediction, uncertainty, deadline pressure, and reroute cost into one explicit score.
4. Select exactly one of K candidates using a one-hot QUBO.
5. Compare a standard mixer that can leave feasibility with an XY mixer that stays feasible.
6. Test whether the selected policy improves realized outcomes under dynamic events.
7. Repeat the evaluation across held-out networks, shifted traffic, and cached real road topology.

## The four ideas to understand

### Calibration is conditional on assumptions

Split conformal prediction offers a marginal finite-sample guarantee under exchangeability. Q-Rescue’s ID coverage is near target, while shifted coverage collapses. That is the key lesson: a mathematically correct interval can fail when its deployment population differs from calibration.

### Feasibility and optimality are different

A valid sample selects exactly one route. An optimal sample selects the best route under the encoded score. Feasible-XY guarantees the former structurally in these circuits, but it does not guarantee the latter or establish an advantage over exact classical selection.

### Static objectives and dynamic outcomes are different

Risk scores and deadline proxies are computed at a decision point. Closures and later events determine realized arrival and deadline outcomes. The dynamic experiment shows why a plausible static risk objective cannot be assumed to improve operations.

### Generalization has several meanings

Held-out traffic rows, held-out networks, held-out topology families, shifted traffic on known networks, combined shifts, and real road graphs are different tests. Q-Rescue reports each separately instead of calling all of them “OOD.”

## Key equations

Risk score:

\[
c_i=\hat y_i+\lambda_u u_i+\lambda_d\max(0,U_i-d)+\lambda_r\rho_i.
\]

One-hot QUBO:

\[
Q(x)=\sum_i c_ix_i+P\left(\sum_i x_i-1\right)^2.
\]

Finite-sample conformal quantile:

\[
q=\operatorname{Quantile}_{higher}\left(s,\frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right).
\]

## The results to remember

- Cross-network local coverage: 0.912 ID, 0.475 traffic shift, 0.504 combined shift.
- LONO pooled local coverage: 0.900, with important fold failures.
- Real-topology transfer: R² -0.187 and coverage 0.544.
- K=8, p=3 validity: standard-X 0.023 versus feasible-XY 1.000.
- Dynamic deadline violations: expected-time 2.86% versus risk QAOA 17.14%.
- Removing reroute penalty: reroutes 0.457 to 0.857, deadline count unchanged.
- Classical exact risk: 15/15 exact; candidate expected time: 12/15.

## How to answer the hardest questions

**Did quantum win?** No. The validated result is constraint-preserving feasibility on an ideal simulator, not quantum advantage.

**Was real data used?** Real road graph topology was used offline; traffic was simulated.

**Did conformal prediction work?** It calibrated well on ID and pooled LONO data, and failed badly under traffic and combined shift.

**Did risk awareness improve routing?** Not on the tested dynamic deadline outcome. It sometimes reduced route churn or regret, but expected-time routing remained stronger.

**Why is the work useful?** It supplies a reproducible, modular testbed and an unusually explicit negative-results boundary for future hardware and real-data studies.

## Reading order

Read [FINAL_RESEARCH_STATUS.md](FINAL_RESEARCH_STATUS.md), [paper/RESULTS.md](paper/RESULTS.md), [paper/CLAIMS.md](paper/CLAIMS.md), [paper/LIMITATIONS.md](paper/LIMITATIONS.md), and then the detailed [TA_PROGRESS.md](TA_PROGRESS.md).
