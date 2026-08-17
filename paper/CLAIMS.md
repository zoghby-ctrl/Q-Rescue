# Claims Ledger — Science Freeze

Every claim below names its experimental population. “Supported” does not mean
externally replicated, causal, or operationally general.

## Supported findings

| Claim | Strength and scope | Evidence |
|---|---|---|
| One-hot feasible-XY maintained `P_feasible=1` while standard-X averaged 0.180 in the matched K=4/8 ideal-statevector factorial. | Strong structural mixer contrast; 1,800 trials, 50 independent synthetic networks | [Step 9.2](../results/research/step9_2_topology_uncertainty_qaoa/VALIDATION_REPORT.md), [Step 9.6](../results/research/step9_6_statistics/VALIDATION_REPORT.md) |
| Feasibility did not imply exact-optimum concentration: feasible-XY mean `P_opt=0.425`, and K=8 cell means remained 0.228–0.309 across p=1/3/5. | Central empirical finding; fixed 15-step optimizer budget | [findings](../STEP_9_2_FINDINGS.md) |
| K=8 versus K=4 reduced XY `P_opt` by 0.308 and raised normalized entropy by 0.206. | Matched, 50-network contrasts with 95% cluster-bootstrap intervals excluding zero | [factorial contrasts](../results/research/step9_6_statistics/factorial_contrasts.csv) |
| Within-network XY entropy was strongly negatively associated with `P_opt` and modestly positively associated with encoded risk regret. | Diagnostic association, not causal mediation | [entropy associations](../results/research/step9_6_statistics/entropy_associations.csv) |
| Increasing depth from p=1 to p=5 improved XY concentration but did not resolve deadline violation or realized-regret effects. | Matched contrast; operational intervals include zero | [factorial contrasts](../results/research/step9_6_statistics/factorial_contrasts.csv) |
| XY materially outperformed X on feasible mass, optimum mass, encoded risk regret, and one-decision outcomes, but not the exact classical optimum. | Mixer comparison only; no advantage claim | [major mixer effects](../results/research/step9_6_statistics/major_mixer_effects.csv) |
| Traffic and combined shift caused severe static-conformal undercoverage. | Strong negative finding on controlled shifted data | [Step 9](../results/research/step9_cross_network/VALIDATION_REPORT.md) |
| The fixed ACI-style baseline recovered part of shifted coverage but stayed below 0.90, greatly widened intervals, and did not change routes relative to local split conformal. | One sequential method; 1,344 rows and 16 events per group | [Step 9.7](../results/research/step9_7_adaptive_conformal/VALIDATION_REPORT.md) |
| The declared local noise model reduced XY feasible and optimum mass, especially at p=3. | K=4 phenomenological simulation; not hardware | [Step 9.8](../results/research/step9_8_noise/VALIDATION_REPORT.md) |
| Expected-time routing remained stronger than the inherited QAOA-assisted policy on controlled dynamic deadlines. | Adverse, exploratory; 35 scenarios but only five network clusters | [dynamic effects](../results/research/step9_6_statistics/dynamic_policy_effects.csv) |
| The frozen point-deadline term changed none of 300 static choices; adding uncertainty changed 18% without a resolved operational benefit. | Objective-isolation result in one-decision realizations | [objective effects](../results/research/step9_6_statistics/objective_ablation_effects.csv) |
| Frozen-model transfer to four cached real road topologies failed statistically. | Real topology, simulated traffic only | [Step 8.5](../results/research/step8_5_real_topology/VALIDATION_REPORT.md) |
| Exact-risk candidate selection is the correctness reference and strongest tested classical candidate selector. | 15 historical decisions; candidate-set scope | [Step 9.3](../results/research/step9_3_classical_baselines/VALIDATION_REPORT.md) |

## Negative and null findings that must remain visible

- High versus low controlled uncertainty did not materially change XY `P_opt`
  or entropy; K was the clearer concentration stressor.
- Entropy explained `P_opt` and encoded regret much better than the binary
  deadline event; its deadline association interval included zero.
- Deeper p=5 circuits improved probability concentration but produced no
  resolved deadline or realized-regret improvement.
- The uncertainty term changed some choices without improving mean realized
  time or violation rate; the point-deadline term changed none.
- Adaptive conformal recovery came with large interval-width inflation and no
  route change versus the local baseline in this controlled event set.
- Ideal XY feasibility did not survive the declared noise model exactly.
- The inherited dynamic QAOA-assisted policy had more deadline violations and
  longer arrival time than expected-time rerouting in an exploratory comparison.

## Unsupported claims

Do not claim:

- quantum advantage, speedup, useful quantum scale, or classical intractability;
- QAOA superiority over exact risk minimization or classical routing;
- that `P_feasible` proxies `P_opt`, regret, or operational quality;
- equal hardware cost for X and XY, backend behavior, or physical validation;
- observed real-traffic, emergency-service, clinical, city-wide, or deployment
  validity;
- conformal coverage under arbitrary shift or optimality of the ACI baseline;
- universal topology causation, cross-city generalization, or a population
  estimate from the controlled network generator;
- a full shortest-path, VRP, multi-vehicle, or dispatch solution;
- invention of XY or constraint-preserving mixers;
- exhaustive novelty, “no prior work,” or a “first ever” contribution.

## Recommended central claim

“Across small, exactly auditable candidate-route problems, constraint-preserving
XY QAOA eliminated one-hot invalid mass but did not guarantee concentration on
the exact risk-optimal route or improved downstream outcomes. Candidate count,
fixed-budget optimization, objective–realization mismatch, calibration shift,
and local noise separate feasibility from utility. These findings diagnose a
failure mode; they do not establish quantum or operational advantage.”
