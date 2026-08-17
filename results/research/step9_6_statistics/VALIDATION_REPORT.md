# Validation Report — Step 9.6 Statistical Rigor

## Status

- Status: **PASS** for matched, cluster-aware execution.
- Step 9.2 rows analyzed: 1800; independent network-instance clusters: 50.
- Bootstrap resampling unit: complete network instance.
- Multiple-comparison control: Holm family-wise adjustment within declared families.
- No naive row-independent factorial ANOVA was used.

## Primary matched mixer effects

- XY − X feasibility probability: 0.820436, 95% cluster-bootstrap CI [0.813327, 0.827442].
- XY − X optimal-state probability: 0.386714, 95% CI [0.354827, 0.417694].
- Mean within-network entropy/P_opt Spearman association (XY): -0.731930, 95% CI [-0.801157, -0.652508].
- Dynamic QAOA − expected-time deadline-violation rate: 0.142857, 95% five-network cluster-bootstrap CI [0.000000, 0.371429].

Consult the CSV tables for all effect sizes, sample sizes, unadjusted and Holm-adjusted p-values. P-values are secondary to effects and intervals.

## Assumptions and limitations

1. Network instances are treated as independent; repeated uncertainty, K, depth, and mixer observations within an instance are not.
2. The five-topology design has 50 independent instances overall but only 10 per topology. High-order mixed models would be unstable, and XY feasibility is on a boundary; prespecified marginal within-network contrasts are used instead.
3. Percentile cluster-bootstrap intervals quantify finite-design uncertainty and do not turn the controlled synthetic networks into a population-representative road sample.
4. Dynamic inference has only five network clusters and remains exploratory even when an interval excludes zero.
5. Spearman associations are diagnostic, not causal mediation estimates.
