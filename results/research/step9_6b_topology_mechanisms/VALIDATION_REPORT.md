# Validation Report — Step 9.6b Topology Mechanisms

## Status

- Status: **PASS** for the declared exploratory analysis.
- Independent network instances: 50 (10 per topology family).
- Topology-summary rows: 100.
- Graph-level association rows: 140.
- Within-network problem association rows: 60.
- Design status: post-hoc exploratory; none of these associations is confirmatory or causal.

## Descriptive findings

- At K=8, the lowest topology-level feasible-XY mean P_opt was grid (0.245143); the highest was sparse_mixed (0.312410).
- The largest absolute mean within-network route/problem association was candidate_expected_time_spread versus optimal_state_probability for standard_x (mean Spearman rho -0.868037, 95% network-bootstrap CI [-0.878310, -0.847492]).

## Interpretation guardrails

1. Topology class and graph statistics are intentionally coupled, so graph-level correlations cannot identify a causal graph mechanism.
2. Each within-network correlation has only six K-by-uncertainty problems and is used as a diagnostic; the network, not the problem row, is the inferential unit.
3. Holm-adjusted exploratory p-values are secondary to effect size, interval width, and mechanistic plausibility.
4. These controlled synthetic networks are not a population sample of road networks.
