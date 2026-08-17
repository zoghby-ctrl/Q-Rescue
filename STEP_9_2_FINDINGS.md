# Step 9.2 Findings — Feasibility Without Utility

Decision-gate date: 2026-08-13

## Status and inferential scope

Step 9.2 is complete. The controlled primary design contains 1,800 ideal
statevector QAOA trials on 300 matched route-selection problems from 50
independent network instances (five topology families × ten matched seeds).
Every topology × uncertainty × K × depth × mixer cell contains ten trials.
The two mixers saw identical candidate routes, risk vectors, deadlines,
uncertainty states, exact optima, and optimization seeds.

Inference uses the network instance as the independent unit, not the 1,800
repeated rows. Confidence intervals are 95% cluster-bootstrap intervals;
paired sign-flip p-values use Holm family-wise correction. The complete
outputs are in
`results/research/step9_2_topology_uncertainty_qaoa/`,
`results/research/step9_6_statistics/`, and the explicitly post-hoc
`results/research/step9_6b_topology_mechanisms/` analysis.

## Supported findings

1. **Constraint preservation and useful concentration are distinct.**
   Feasible-XY has structural mean P_feasible = 1.000, while standard-X has
   mean P_feasible = 0.180. The matched XY − X feasibility effect is 0.820
   [0.813, 0.827]. Yet mean feasible-XY P_opt is only 0.425 overall and 0.228,
   0.276, and 0.309 at K=8 for p=1, 3, and 5. Thus all XY probability mass is
   valid without being concentrated on the exact minimum-risk route.

2. **The feasibility–concentration disconnect is statistically supported.**
   XY improves P_opt over X by 0.387 [0.355, 0.418] and
   P_opt_given_feasible by 0.245 [0.216, 0.273], but neither reaches one.
   Moving from K=4 to K=8 reduces feasible-XY P_opt by 0.308
   [-0.352, -0.264] and increases its normalized feasible entropy by 0.206
   [0.175, 0.239]. Route-set size is therefore a stronger concentration
   stressor than predictive-uncertainty regime in this design.

3. **Entropy diagnoses encoded concentration and regret.** Among feasible-XY
   trials, the mean within-network Spearman association between normalized
   entropy and P_opt is -0.732 [-0.801, -0.653]. The association with encoded
   risk regret is 0.270 [0.172, 0.364]. Both survive Holm correction.

4. **Depth helps concentration but does not close the utility gap.** From p=1
   to p=5, feasible-XY P_opt increases by 0.128 [0.101, 0.155] and normalized
   entropy decreases by 0.174 [0.144, 0.204]. The corresponding deadline-
   violation change is -0.023 [-0.070, 0.020], and realized-regret change is
   -0.239 minutes [-0.636, 0.116]. Better state concentration at the tested
   depth and optimizer budget therefore did not yield a resolved operational
   improvement.

5. **XY is materially better than standard-X, but not better than the exact
   classical decision.** In the controlled one-decision outcomes, XY − X
   changes deadline-violation rate by -0.328 [-0.413, -0.237] and realized
   regret versus the actual fastest route by -2.330 minutes
   [-2.983, -1.683]. This establishes a mixer contrast, not quantum advantage:
   exact enumeration remains the reference and is trivial at K=4/8.

## Rejected or unsupported hypotheses

1. **Predictive uncertainty does not materially reduce encoded P_opt by
   itself.** The high − low uncertainty contrast for feasible-XY P_opt is
   0.004 [-0.016, 0.024]. Its entropy contrast is -0.002
   [-0.016, 0.012]. Uncertainty changes realized conditions, but it is not the
   primary explanation for state-concentration failure here.

2. **Entropy is not a demonstrated direct explanation of deadline failure.**
   Its mean within-network association with one-decision deadline violation is
   0.102 [-0.005, 0.205] (Holm p = 0.070). The association with realized regret
   is positive but small, 0.149 [0.064, 0.237]. Entropy explains encoded
   P_opt/risk-regret behavior more clearly than the binary operational event.

3. **The added point-deadline term does not alter these static selections.**
   Objective B selects exactly the same route as expected-time objective A in
   all 300 problems. Adding uncertainty in C changes 18% of selections, but C
   − B changes realized time by 0.089 minutes [-0.293, 0.431] and violation
   rate by 0.010 [-0.020, 0.047]. There is no evidence of operational benefit
   from this uncertainty term in the primary one-decision realization.

4. **The inherited dynamic benchmark does not support QAOA-assisted routing
   superiority.** Across 35 matched scenarios from five networks, the
   uncertainty-aware QAOA policy minus expected-time rerouting changes
   deadline-violation rate by +0.143 [0.000, 0.371] and arrival time by
   +1.682 minutes [0.020, 4.632]. With only five clusters these estimates are
   exploratory, but their direction is adverse and must be preserved.

## Topology and route-structure diagnostics

At K=8, topology-level feasible-XY mean P_opt ranges from 0.245 for grid to
0.312 for sparse mixed; the ten-network bootstrap intervals overlap. No
feasible-XY graph-property association has |Spearman rho| ≥ 0.5. Because the
controlled topology families deliberately determine density and path
statistics, those graph correlations are descriptive and cannot identify a
causal topology mechanism.

The post-hoc within-network route-set analysis suggests that wider candidate
expected-time spread co-occurs with lower XY P_opt (mean rho -0.806
[-0.868, -0.709]) and higher XY entropy (0.822 [0.776, 0.858]); a larger
best-versus-second risk gap co-occurs with higher XY P_opt (0.306
[0.168, 0.440]). These six-problem correlations are partly coupled to K and
are mechanism diagnostics, not confirmatory evidence.

## Mechanistic conclusion

The data support a qualified mechanism: the XY ansatz solves the validity
problem exactly, but shallow, fixed-budget optimization leaves probability
broadly distributed over the one-hot subspace, especially as K grows. Higher
entropy then predicts lower P_opt and greater encoded risk regret. This is not
a complete operational mechanism: realization noise, objective/decision
misalignment, and the limited binary deadline event weaken the link from
encoded concentration to downstream utility.

Accordingly, **P_feasible is not a proxy for P_opt or operational quality**.
The evidence does not support quantum advantage, quantum speedup, superiority
over exact risk-aware selection, or emergency-deployment readiness.

## Decision gate

- Step 9.6 is justified and complete because point estimates alone were
  inadequate and the matched clustered design required corrected inference.
- Step 9.7 remains justified as a limited diagnostic of whether static
  conformal failure under shift contributes to downstream objective error.
- Step 9.8 remains justified because exact XY feasibility in an ideal
  statevector need not survive finite shots and local NISQ-style noise.
- Targeted K=16 and penalty-sensitivity studies remain justified as scaling
  and reviewer-robustness checks. K=16 remains classically trivial.
- A broad p=8 sweep is not justified: p=5 already improves concentration
  without a resolved operational effect. No p=8 run will be used to search for
  a favorable result.

