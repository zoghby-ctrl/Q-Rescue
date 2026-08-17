# Demo Guide

## Goal

Demonstrate the accepted research evidence without rerunning long experiments.

## Before the meeting

- Open [FINAL_RESEARCH_STATUS.md](FINAL_RESEARCH_STATUS.md).
- Open Figures [9](paper/figures/fig09_feasibility_optimality_depth.png), [11](paper/figures/fig11_entropy_risk_regret.png), [13](paper/figures/fig13_adaptive_conformal.png), [14](paper/figures/fig14_noise_robustness.png), and [15](paper/figures/fig15_conceptual_pipeline.png).
- Keep [paper/CLAIMS.md](paper/CLAIMS.md) and [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) ready.
- Do not launch QAOA, network download, benchmark, or ablation runners.

## Five-minute walkthrough

1. Show the evidence index and state that Steps 9.2 and 9.6–9.8 are complete.
2. Show Figure 9 and separate feasible mass, exact-optimum mass, and conditional optimum mass.
3. Show Figure 11 and explain that entropy diagnoses concentration better than operational utility.
4. Show Figures 13 and 14 for calibration-shift and local-noise failure boundaries.
5. State the adverse inherited dynamic result and the exact-classical reference.
6. Finish with the claims ledger and conditional Step 10 recommendation.

## Optional artifact drill-down

- For a coverage question, open [`conformal_metrics_overall.csv`](results/research/step9_cross_network/conformal_metrics_overall.csv).
- For a fold question, open [`unfavorable_folds.csv`](results/research/step9_1_leave_one_network_out/unfavorable_folds.csv).
- For a QAOA question, open [`major_mixer_effects.csv`](results/research/step9_6_statistics/major_mixer_effects.csv).
- For a dynamic question, open [`dynamic_policy_summary.csv`](results/research/step8_dynamic_rerouting/dynamic_policy_summary.csv).
- For the penalty boundary, open [`qubo_penalty_aggregate.csv`](results/research/step9_4_robustness_ablation/qubo_penalty_aggregate.csv).
- For classical baselines, open [`baseline_summary.csv`](results/research/step9_3_classical_baselines/baseline_summary.csv).

## Questions to answer directly

**Is this real traffic?** No. The real-topology experiment uses cached real road graphs with controlled simulated traffic.

**Did QAOA beat classical methods?** No. The result is stronger constraint feasibility for feasible-XY on an ideal simulator.

**Why show failures?** They define where calibration, transfer, and dynamic assumptions break and make the next experiment scientifically credible.

**Can Step 9.2 be shown?** Yes. It has 1,800 accepted trials, a validation report, cluster-aware inference, and a separate findings gate.

## Recovery plan

If a notebook or plot viewer fails, use the static PNGs and Markdown reports. The demo does not depend on a live kernel or network connection.
