# Q-Rescue GitHub Release Guide

This guide separates a public software repository from the private science-freeze and manuscript archive. It does not authorize publication. As of 2026-08-17, this folder is not a Git worktree and has no remote.

## Release decisions required first

1. **License:** no license exists. A license decision is required before public GitHub release. Until one is selected by the owner, normal copyright applies and reuse rights are not granted.
2. **Citation author:** `CITATION.cff` contains an explicit author placeholder because no unambiguous author, affiliation, ORCID, DOI, venue, or publication date is present in the repository.
3. **Manuscript approval:** `paper/MANUSCRIPT_DRAFT.md`, `paper/ABSTRACT_DRAFT.md`, their manifests, and other paper/review material must remain private until the TA/supervisor approves disclosure.
4. **Machine-path provenance:** immutable experiment manifests contain `C:\Users\Zoghby\...\.codex\...` paths and the local Python executable. The originals must not be edited because they are frozen evidence. Keep them in the private archive or create a separately reviewed, clearly labeled sanitized release export.

## Publication classification

| Class | Treatment | Contents |
|---|---|---|
| A. Commit to GitHub | Public software repository | `app.py`, `.streamlit/config.toml`, `requirements.txt`, `requirements-research.txt`, `src/*.py`, `ui/*.py`, `tests/*.py`, non-sensitive `configs/`, cached `.graphml` topology files plus `cache_registry.csv`, runtime model/calibration artifacts, result summaries used by the Research Results tab, public-facing guides, `paper/CLAIMS.md`, `paper/LIMITATIONS.md`, `paper/REPRODUCIBILITY.md`, and the three final QA screenshots. |
| B. Optional release/archive artifact | Preserve outside ordinary Git or use an approved archive/LFS release | Complete `results/research/` evidence, checkpoint collections, large row-level interval tables, training histories, raw Overpass responses, notebooks, superseded-result folders, old screenshots, and paper figures/tables. These are evidence, not disposable cache. |
| C. Exclude from Git | Regenerable or local-only build state | `.venv/`, `__pycache__/`, `.pytest_cache/`, editor files, coverage/build output, logs, temporary files, and Playwright CLI scratch data. |
| D. Private/local only | Do not publish without explicit review | `.streamlit/secrets.toml`, `.env*`, credentials/keys, manuscript and abstract drafts, submission-review material, the original path-bearing science/experiment manifests, and any TA/supervisor correspondence. |

The checked-in `.gitignore` implements the conservative default. It does not delete any local evidence.

## Repository size audit

Audit date: 2026-08-17.

- Repository: **143,638,571 bytes (136.98 MiB / 0.134 GiB)** across 2,915 files.
- `results/`: **126.52 MiB**; `results/research/`: **125.83 MiB**.
- All checkpoint directories: **24.21 MiB** across 2,223 files.
- `data/`: **3.44 MiB**; `paper/`: **2.88 MiB**; `output/`: **0.99 MiB**.
- Largest research directories: leave-one-network-out 46.49 MiB; Step 9.2 28.16 MiB; cross-network 22.33 MiB.

Largest individual files:

| File | Bytes | Recommendation |
|---|---:|---|
| `results/research/step9_1_leave_one_network_out/fold_models.joblib` | 40,939,278 | Release/archive or Git LFS; not ordinary Git. |
| `results/research/step9_cross_network/conformal_intervals.csv` | 15,124,967 | Release/archive evidence. |
| `results/research/step9_1_leave_one_network_out/heldout_conformal_intervals.csv` | 4,946,176 | Release/archive evidence. |
| `results/research/step9_7_adaptive_conformal/coverage_intervals.csv` | 4,408,506 | Release/archive evidence. |
| `results/research/step5_conformal/intervals.csv` | 3,785,091 | Release/archive evidence. |
| `results/research/step9_cross_network/traffic_observations.csv` | 3,601,690 | Retain when source-level adaptive-conformal reproduction is required. |
| `results/research/step9_2_topology_uncertainty_qaoa/qaoa_training_history.csv` | 3,351,062 | Release/archive evidence. |

No file exceeds GitHub's 100 MiB single-file limit, but the 39.04 MiB serialized fold-model collection and thousands of checkpoints are poor ordinary-Git candidates.

## Exact MVP deployment set

The deployed app's runtime surface is **64 files / 6,602,871 bytes (6.30 MiB)**:

- `app.py`, `.streamlit/config.toml`, and `requirements.txt`;
- all Python files directly under `ui/` and `src/`;
- `configs/topology_repeated_benchmark.json` and `configs/real_topology_benchmark.json`;
- `data/real_networks/cache_registry.csv` and the four `data/real_networks/*.graphml` files;
- synthetic inference: `best_model.joblib`, `residual_scale_model.joblib`, `conformal_calibration.json`, and `traffic_observations.csv` under `results/research/steps8_25_8_75_topology_benchmark/`;
- real-transfer inference: `results/research/step4_traffic_ai/best_model.joblib`, `train.csv`, `results/research/step5_conformal/residual_scale_model.joblib`, and `calibration_quantiles.csv`;
- Research Results tab summaries: cross-network `conformal_metrics_overall.csv`, LONO `conformal_metrics_aggregate.csv`, real-topology `model_metrics_overall.csv` and `conformal_coverage_overall.csv`, robustness `qaoa_aggregate.csv` and `dynamic_ablation_summary.csv`, and dynamic `dynamic_policy_summary.csv`;
- `paper/CLAIMS.md`.

No Codex-local directory, environment variable, secret, API key, network download, raw Overpass response, checkpoint, or manuscript file is required by the default app.

## Additional research-reproducibility set

For the complete frozen research archive, preserve the MVP set plus:

- all `configs/`, `experiments/`, `tests/`, `requirements-research.txt`, and the teaching notebook;
- all accepted `results/research/` artifacts, including manifests, checkpoints, failures retained for audit, row-level tables, and training histories;
- `data/real_networks/cache_manifest.json` and raw `.overpass.json` responses;
- the science-freeze documents and immutable `SCIENCE_FREEZE_MANIFEST.json`;
- paper sources, evidence matrices, figures, tables, and generation manifests.

This additional set is the private/research archive boundary. It is not an instruction to expose the manuscript. A public reproducibility deposit should be assembled only after TA/supervisor review and should document any sanitized path metadata without altering the originals.

## Secret and path audit result

- No `st.secrets`, environment-variable credential reads, API-key literals, token files, or credential filenames were found.
- `.streamlit/secrets.toml` is absent and ignored if later created.
- The default MVP makes no external network request and needs no secret.
- Machine-specific absolute paths occur in frozen result/cache manifests and failure records; they are provenance metadata, not runtime dependencies.
- Local bytecode and pytest caches exist and are ignored.

## Safe first-publication workflow

Do not run these steps until the four release decisions above are resolved.

```powershell
git init -b main
git add .
git status --short
git diff --cached --stat
git diff --cached -- . ':!*.png' ':!*.pdf' ':!*.joblib'
```

Before committing, verify that ignored/private files are absent from the staged set, confirm the final screenshots are present, and run the tests documented in `README.md`. Then create the first commit locally. Add a GitHub remote and push only after explicit owner approval; this task intentionally does neither.

## Pre-release checklist

- [ ] License selected by the owner and added.
- [ ] Real author/affiliation confirmed in `CITATION.cff`.
- [ ] TA/supervisor decides whether any manuscript or review material is public.
- [ ] Public/research-archive boundary approved.
- [ ] Immutable path-bearing manifests reviewed for privacy.
- [ ] Staged files reviewed for secrets and unexpected binaries.
- [ ] Science-freeze hashes rechecked.
- [ ] Full test suite green.
- [ ] Streamlit app checked at desktop and narrow widths.
- [ ] No public push or deployment occurs without explicit authorization.
