# Q-Rescue repository audit

Audit date: 2026-08-10 (Africa/Cairo)

## Outcome

The repository is a coherent, validated Prototype 0.1 and has not yet reached
Step 4 of the generalized research plan. Later prototype primitives are useful,
but they do not constitute completed generalized stages.

## Verified historical baseline

- `python -m pytest -q`: 11 passed, 0 failed.
- Exhaustive table: all 16 four-bit assignments are present.
- Exact optimum: Route B / `0100` / cost 23.
- p=1: 2,545/4,000 valid (63.625%); 668/4,000 optimal (16.700%).
- p=2: 3,962/4,000 valid (99.050%); 1,036/4,000 optimal (25.900%).
- Route B is the most frequent valid route at both depths.
- The saved notebook has outputs for all 16 code cells and records seed 2026,
  4,000 shots, 100 Adam iterations, and PennyLane 0.45.1.

These values are descriptive simulator results, not evidence of quantum
advantage or reliable optimum concentration. In particular, the p=2 Route B
count is only slightly above the other valid routes.

## Step status at audit

| Requested step | Audit status |
|---|---|
| 4 traffic AI | Not started |
| 5 split conformal uncertainty | Not started |
| 6 deadline/reroute risk objective | Prototype precursor only |
| 6.5 arbitrary-graph candidate routing | Four hard-coded routes only |
| 7 generalized QUBO | Variable-length primitive; only K=4 validated |
| 7.5 quantum comparison | Standard X mixer, p=1/2, one seed only |
| 8–9.2 dynamic/topology/OOD studies | Not started |
| 9.3 classical baselines | Prototype expected-time and enumeration only |
| 9.4 ablation | Not started |
| 9.5 paper package | Not started |

## Reproducibility risks found

- The workspace is not a Git checkout, so no commit hash is available.
- The global Python 3.12.10 environment was active rather than a project venv.
- The declared `jupyter==1.1.1` meta-package was absent, although the installed
  Notebook, JupyterLab, and nbconvert components were usable.
- No lockfile, CI, artifact checksums, or experiment manifest existed.
- Historical tests did not execute QAOA training/sampling or regression-test
  the saved quantum metrics.
- The historical notebook writes to generic `results/` filenames and could
  overwrite its own baseline artifacts if repurposed.

## Preservation decision

All original source, tests, notebook, documentation, requirements, and result
files are treated as Prototype 0.1. Generalized work uses new modules and writes
new outputs only below `results/research/`. The boundary is recorded in
`prototype_0_1/BASELINE_MANIFEST.md`.

## Earliest next step

Step 4 is the first scientifically valid continuation: create location-neutral
network/data schemas, generate strictly separated synthetic observations, train
the specified regression baselines without network identity as a feature, and
save overall plus per-network/topology metrics.
