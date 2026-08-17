# Table Index — Science Freeze

`experiments/build_science_freeze_tables.py` regenerates every table below from
validated CSV artifacts and writes
`paper/tables/science_freeze_table_manifest.json`. Markdown values are display-
rounded; companion CSVs retain full precision. No paper table is hand-entered.

| Table | Artifact | Source population | Purpose |
|---:|---|---|---|
| 1 | [Topology statistics](paper/tables/table01_topology_statistics.md) | 50 primary network instances | Controlled graph/candidate-family characteristics |
| 2 | [Prediction and conformal calibration](paper/tables/table02_prediction_conformal.md) | Step 9 point metrics; Step 9.7 replay/adaptive coverage | ID and shifted point error, coverage, and interval width |
| 3 | [QAOA diagnostics](paper/tables/table03_qaoa_diagnostics.md) | 1,800 primary trials | Feasible/optimal/conditional mass, entropy, and regret by K/p/mixer |
| 4 | [Dynamic outcomes](paper/tables/table04_dynamic_outcomes.md) | 35 scenarios per policy, five networks | Policy summaries and cluster-aware contrasts |
| 5 | [Statistical findings](paper/tables/table05_statistical_findings.md) | 50 primary network clusters | Matched mixer effects and prespecified factorial contrasts |
| 6 | [Noise robustness](paper/tables/table06_noise_robustness.md) | 120 K=4 condition rows | Ideal, finite-shot, and declared local-noise results |
| 7 | [Ablation summary](paper/tables/table07_ablation_summary.md) | Objective, adaptive-route, and penalty analyses | Decision-component and reviewer-robustness controls |
| S1 | [Targeted K=16 scaling](paper/tables/tableS01_k16_scaling.md) | 72 K=16 trials and matched K=8 cells | Indicative concentration scaling |

## Supporting machine-readable tables

Each Markdown table links conceptually to one or more CSVs with the same table
number (`table02a`, `table02b`, and so on). Full raw populations remain in
`results/research/`; publication tables are summaries, never substitutes for
raw checkpoints or manifests.
