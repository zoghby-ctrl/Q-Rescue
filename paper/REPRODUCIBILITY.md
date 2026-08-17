# Reproducibility

## Reproduction levels

1. **Artifact audit.** Verify the stored config, source, output, and checkpoint
   hashes; recompute summaries and publication graphics from accepted CSV/JSON
   files. This is the fastest path for auditing every manuscript value.
2. **Classical replay.** Regenerate controlled traffic, fit frozen models,
   calibrate intervals, enumerate candidate routes/QUBOs, and replay classical
   and dynamic policies from their declared partitions and seeds.
3. **Quantum-simulator replay.** Rerun the statevector factorial, targeted K=16,
   penalty analysis, and local density-matrix noise study. These are classical
   simulations and are substantially slower than artifact reconciliation.
4. **Real-topology replay.** Load the cached GraphML files under the offline
   guard. Download provenance is separate and need not be repeated.

## Final scientific runners

From the repository root:

```text
python experiments/run_step9_2_topology_uncertainty_qaoa.py all
python experiments/run_step9_6_statistics.py
python experiments/run_step9_6b_topology_mechanisms.py
python experiments/run_step9_7_adaptive_conformal.py
python experiments/run_step9_8_noise.py
python experiments/run_targeted_k16_scaling.py all --workers 3
python experiments/run_standard_x_penalty_sensitivity.py all --workers 3
python experiments/build_step9_5_figures.py
python experiments/build_science_freeze_figures.py
python experiments/build_science_freeze_tables.py
python -m pytest -q
```

The long QAOA runners atomically checkpoint each completed trial and resume only
missing run IDs. Finalization refuses incomplete or duplicate designs. Plot and
table builders read frozen artifacts only; they do not train, optimize, download,
or manually embed numerical values. Their manifests hash every source and output.

## Evidence and design identity

- Step 9.2 stores 1,800 trial checkpoints, 1,800 run rows, 10,800 feasible-route
  probabilities, and 28,800 training-history rows. The manifest includes every
  checkpoint hash and the 9,891.64-second cumulative optimizer wall-time sum.
- Mixers within a matched problem share its problem SHA-256, candidate order,
  risk vector, uncertainty state, deadline, exact optimum, and optimization seed.
- Step 9.6 treats 50 network instances—not repeated factorial rows—as independent
  clusters. Bootstrap and randomization seeds are frozen in config.
- Step 9.7 deterministically regenerates the original pre-CSV floating-point
  observations from hash-verified Step 9 sources, then reproduces accepted static
  intervals to a maximum absolute error of 7.105e-15 before adaptation.
- Step 9.8 reproduces all 40 inherited analytic K=4 source runs before shot/noise
  evaluation; its 4,096-shot seeds and effective channels are stored.
- Targeted K=16 reproduces every accepted K=8 prefix before appending eight
  candidates. Penalty tuning uses validation seeds 0–2 only and reports every
  frozen condition on disjoint test seeds 3–9.

## Environment and provenance

The accepted runtime uses Python 3.12.10, PennyLane 0.45.1, NumPy 2.4.4,
pandas 2.3.3, NetworkX 3.6.1, matplotlib 3.11.1, scikit-learn 1.7.1, SciPy
1.18.0, and pytest 8.3.4. Package versions are repeated in experiment manifests.
`pip check` passed in the scientific audit.

This directory is not a Git worktree, so `git_commit` is correctly recorded as
null. Config/source/artifact hashes are the available snapshot provenance. The
environment is not a fully locked container and therefore cannot guarantee
bitwise reproduction across dependency or platform changes.

## Final verification rule

For a submission snapshot, do not regenerate accepted experiments merely to
refresh timestamps. Verify every manifest against current bytes, confirm all
indexed artifacts exist, rebuild figures/tables from the saved data, run the
full suite in a clean process, audit Markdown links, and record both the test
count and science-freeze manifest hash in `TA_PROGRESS.md`.
