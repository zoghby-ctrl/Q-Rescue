# Experiment Index — Science Freeze

This is the authoritative map from research stage to accepted evidence.
“Validated” means stored design identities, raw tables, reconciliation checks,
and hashes passed; it does not broaden external validity.

| Stage | Status | Population | Evidence | Defensible takeaway |
|---|---|---:|---|---|
| Prototype 0.1 | Historical validated baseline | 4 routes | [results](results/) | End-to-end proof of concept only |
| Step 4 traffic AI | Validated | 4,800 synthetic trips | [report](results/research/step4_traffic_ai/VALIDATION_REPORT.md) | Accurate configured-domain prediction |
| Step 5 conformal | Validated | disjoint ID/shift partitions | [report](results/research/step5_conformal/VALIDATION_REPORT.md) | Traffic shift causes undercoverage |
| Steps 6/6.5 risk/candidates | Validated | 15 decisions, K=4/6/8 | [report](results/research/steps6_6_5_routing_risk/VALIDATION_REPORT.md) | Auditable candidate score; limited route changes |
| Step 7 generalized QUBO | Validated | 30 exact QUBOs | [report](results/research/step7_qubo/VALIDATION_REPORT.md) | Ground-state equivalence on tested instances |
| Step 7.5 QAOA | Validated historical | 106 runs | [report](results/research/step7_5_qaoa/VALIDATION_REPORT.md) | Initial ideal XY/X contrast |
| Step 8 dynamic routing | Validated negative | 35 scenarios × 4 policies | [report](results/research/step8_dynamic_rerouting/VALIDATION_REPORT.md) | Expected-time policy strongest in corpus |
| Steps 8.25/8.75 topology | Validated | 15 networks, 1,680 trips | [report](results/research/steps8_25_8_75_topology_benchmark/VALIDATION_REPORT.md) | Topology characterization and compact QAOA |
| Step 8.5 real topology | Validated negative | 4 cached OSM graphs, 384 simulated trips | [report](results/research/step8_5_real_topology/VALIDATION_REPORT.md) | Engineering replay works; statistical transfer fails |
| Step 9 cross-network/OOD | Validated | 20 networks, 9,408 trips | [report](results/research/step9_cross_network/VALIDATION_REPORT.md) | Traffic/combined shift damage prediction/calibration |
| Step 9.1 LONO | Validated | 20 folds, 4,320 held-out rows | [report](results/research/step9_1_leave_one_network_out/VALIDATION_REPORT.md) | Pooled success hides network heterogeneity |
| **Step 9.2 interaction** | **Complete and validated** | **1,800 trials, 300 problems, 50 networks** | [report](results/research/step9_2_topology_uncertainty_qaoa/VALIDATION_REPORT.md), [findings](STEP_9_2_FINDINGS.md) | XY solves validity, not concentration or utility |
| Step 9.3 classical baselines | Validated | 15 decisions, 90 selections | [report](results/research/step9_3_classical_baselines/VALIDATION_REPORT.md) | Exact-risk is candidate ground truth |
| Step 9.4 robustness | Validated | static/penalty/QAOA and 175 paired dynamic rows | [report](results/research/step9_4_robustness_ablation/VALIDATION_REPORT.md) | Penalty boundary and reroute-churn effect |
| **Step 9.6 statistics** | **Complete and validated** | **50 primary network clusters** | [report](results/research/step9_6_statistics/VALIDATION_REPORT.md) | Cluster-aware effects and uncertainty intervals |
| Step 9.6b mechanisms | Validated exploratory | 100/140/60 summary/association rows | [report](results/research/step9_6b_topology_mechanisms/VALIDATION_REPORT.md) | Post-hoc route/topology diagnostics, noncausal |
| **Step 9.7 adaptive conformal** | **Complete and validated** | **16,128 intervals, 192 decisions** | [report](results/research/step9_7_adaptive_conformal/VALIDATION_REPORT.md) | Partial coverage recovery, width cost, no local-route gain |
| **Step 9.8 local noise model** | **Complete and validated** | **120 condition rows, 10 problems** | [report](results/research/step9_8_noise/VALIDATION_REPORT.md) | Ideal XY invariant degrades under declared noise |
| Targeted K=16 | Validated indicative | 72 trials, 18 problems, 9 networks | [report](results/research/targeted_k16_scaling/VALIDATION_REPORT.md) | Feasible mass stays one; concentration worsens |
| Standard-X penalty | Validated reviewer control | 350 trials, 15 validation/35 test networks | [report](results/research/standard_x_penalty_sensitivity/VALIDATION_REPORT.md) | Complete held-out penalty-scale sensitivity |
| Science-freeze package | Complete synthesis | 17 figures, 7 main + 1 supplementary tables | [freeze](FINAL_SCIENCE_FREEZE.md) | Artifact-derived manuscript package |
| Step 10 hardware | Not started | — | [plan](NEXT_STEP_HARDWARE.md) | Optional small feasibility/noise follow-up only |

## Historical prototype artifacts

- [brute-force evaluation](results/brute_force_evaluation.csv)
- [QAOA metrics](results/qaoa_metrics.csv)
- [measurement counts](results/measurement_counts.csv)
- [optimized parameters](results/optimized_parameters.csv)
- [training history](results/training_history.csv)

The historical prototype remains available for traceability but is not pooled
with the definitive Step 9.2 factorial.
