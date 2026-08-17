# Figure Index — Science Freeze

Figures 1–8 are rebuilt by `experiments/build_step9_5_figures.py`, which writes
`paper/figures/step9_5_figure_manifest.json`. Figures
9–17 are rebuilt by `experiments/build_science_freeze_figures.py` as both PNG
and vector-text PDF. The latter builder writes source/output hashes to
`paper/figures/science_freeze_figure_manifest.json`. No builder trains a model,
runs QAOA, or embeds manually typed result values.

| Fig. | Artifact | Authoritative source | Main message | Required caveat |
|---:|---|---|---|---|
| 1 | [Cross-network coverage](paper/figures/fig01_cross_network_coverage.png) | `step9_cross_network/conformal_metrics_overall.csv` | Traffic shift breaks static local coverage | Nominal line is not a shifted-data guarantee |
| 2 | [LONO transfer](paper/figures/fig02_lono_transfer.png) | Step 9.1 fold metrics | Pooled success hides network heterogeneity | Folds share fitting networks |
| 3 | [Real-topology transfer](paper/figures/fig03_real_topology_transfer.png) | Step 8.5 coverage by network | All four cached road graphs under-cover | Topology is real; traffic is simulated |
| 4 | [Historical mixer/depth](paper/figures/fig04_qaoa_method_depth.png) | Step 7.5 QAOA aggregate | Initial K=4 XY/X feasibility contrast | Superseded inferentially by Step 9.2 |
| 5 | [Historical K scaling](paper/figures/fig05_qaoa_k_scaling.png) | Step 7.5 p=1 aggregate | Standard-X feasibility declines with K | Unequal historical replication |
| 6 | [Dynamic policies](paper/figures/fig06_dynamic_policy.png) | Step 8 policy summary | Expected-time policy has best deadlines in corpus | Five networks, controlled simulation |
| 7 | [Robustness boundaries](paper/figures/fig07_robustness_boundaries.png) | Step 9.4 penalty/dynamic summaries | Penalty validity boundary; reroute cost controls churn | Finite tested instances |
| 8 | [Classical baselines](paper/figures/fig08_classical_baselines.png) | Step 9.3 baseline summary | Exact risk is candidate ground truth | Candidate/graph-wide spaces differ |
| **9** | [Feasibility and optimum mass vs depth](paper/figures/fig09_feasibility_optimality_depth.png) | Step 9.6 descriptive primary | Feasibility, optimum mass, and conditional optimum mass are distinct | Ideal statevector; cluster intervals, not row independence |
| **10** | [Entropy by uncertainty/topology](paper/figures/fig10_entropy_uncertainty_topology.png) | Step 9.2 raw QAOA runs | Entropy varies more by topology/K structure than controlled uncertainty | Descriptive topology interaction |
| **11** | [Risk regret vs entropy](paper/figures/fig11_entropy_risk_regret.png) | Step 9.2 raw QAOA runs | Higher entropy co-occurs with encoded regret | Association is not causal |
| **12** | [Operational violation vs Popt](paper/figures/fig12_operational_violation_popt.png) | Step 9.2 raw QAOA runs | Popt relates to but does not determine realized utility | Binned one-decision outcomes, not trajectories |
| **13** | [Adaptive conformal](paper/figures/fig13_adaptive_conformal.png) | Step 9.7 coverage summary | Adaptive coverage recovery costs width | One order-sensitive baseline; still below target under shift |
| **14** | [Noise robustness](paper/figures/fig14_noise_robustness.png) | Step 9.8 raw noise runs | Declared noise reduces XY feasible/optimal mass | Phenomenological model, not hardware |
| **15** | [Conceptual pipeline](paper/figures/fig15_conceptual_pipeline.png) | Validated framework structure | Separates prediction, risk, QUBO, QAOA, decoding, outcome | Conceptual, not a performance result |
| **16** | [Targeted K scaling](paper/figures/fig16_targeted_k_scaling.png) | Step 9.2 + K=16 matched runs | XY remains feasible while concentration worsens | Nine networks; K=16 classically trivial |
| **17** | [Penalty sensitivity](paper/figures/fig17_penalty_sensitivity.png) | Held-out penalty test grid | Standard-X behavior depends on penalty/scale beyond validity | Validation-selected; fixed optimizer budget |

For the main text, prioritize Figures 9, 11, 13, 14, and 15. Figures 10, 12,
16, and 17 are useful mechanism/reviewer controls and may move to supplementary
material depending on venue length.
