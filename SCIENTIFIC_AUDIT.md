# Scientific Repository Audit — Pre-Freeze

Audit date: 2026-08-13 (Africa/Cairo)

## Scope inspected

- Required status and planning files: `FINAL_RESEARCH_STATUS.md`,
  `TA_PROGRESS.md`, `RESEARCH_PLAN.md`, `EXPERIMENT_INDEX.md`,
  `FIGURE_INDEX.md`, and `TABLE_INDEX.md`.
- All current manuscript sections and the three literature evidence matrices.
- Every accepted validation report from Steps 4 through 9.4, including the
  superseded-correlation warning and the Step 9.2 incomplete marker.
- Existing generalized QAOA implementation and runners for Steps 7.5, 8,
  8.25/8.75, 8.5, 9, 9.2, and 9.4.
- Repository-wide local Markdown links, accepted CSV readability, exact-row
  duplication, numeric finiteness, and stored manifest hashes.

## Baseline validation

- Repository files inventoried by `rg --files`: 428.
- Clean pre-change test run: **135 passed in 21.78 seconds**.
- Accepted CSV files inspected: 152, containing 138,815 rows.
- CSV read failures: 0.
- CSVs with exact duplicate rows: 0.
- Local Markdown links checked: 141; missing targets: 0.
- Existing Step 9.2 accepted numerical runs: 0.

Missing numeric values occur only in fields whose semantics permit them, such
as an absent selected route for invalid QUBO states, failed dynamic arrivals,
non-comparable graph-wide baselines, and deliberately undefined correlations
against structurally constant feasible-XY validity. They are not silently
imputed.

## Manifest and provenance finding

Every stored artifact hash present in the modern manifests matched its file.
The Step 8.25/8.75, Step 9, and Step 9.1 manifests each report one current
source mismatch: `requirements-research.txt`. The expected hash is
`4385da421c796e2381245ce041b6e8e50954f51e54b6b32ea1f3cd771c210cd1`; the
current hash is
`8583ddfa73b68dafa109594dbebad93caf6f51fecadd9311c84af9d58a0b15e0`.
The file was extended later with Streamlit/Starlette dependencies for the MVP.
No scientific Python source hash or accepted result artifact hash failed.

Impact assessment: this is environment-file provenance drift, not evidence
that historical numerical results changed. Historical manifests are preserved
as the record of their accepted environment. New experiments must hash the
current dependency files and state that Git metadata is unavailable because
this workspace is not a Git worktree.

## Step 9.2 design audit

The existing draft is not suitable for the requested science freeze. It would
run only 60 ideal K=4, p=1 cells, use one optimizer seed per network, and write
outputs only after all runs finish. It does correctly reconstruct objective
components and preserve matched X/XY inputs, but it lacks:

- K=8 and p=3/5 primary cells;
- ten matched replications per factorial cell;
- genuine low/medium/high predictive-uncertainty regimes;
- incremental atomic checkpoints and resume support;
- feasible-conditional optimal probability, entropy, concentration, regret,
  route-rank, and full feasible-distribution diagnostics;
- inferential statistics and confidence intervals.

The unfinished runner and configuration may therefore be replaced without
altering any accepted scientific module or result.

## Runtime preflight

Representative existing risk vectors were timed for K in {4, 8}, p in
{1, 3, 5}, both mixers, and eight Adam iterations. Per-run times ranged from
0.55 seconds (K=4, p=1, standard-X) to 7.55 seconds (K=8, p=5,
feasible-XY). Linear extrapolation gives approximately 2.6 hours for 1,800
sequential runs at 15 iterations, before checkpoint and process overhead, and
more than six hours at 35 iterations.

Decision: use a predeclared 15-iteration primary budget with convergence
diagnostics, atomic per-trial checkpoints, and matched seeds. Treat deeper or
larger budgets as targeted follow-ups rather than silently reducing the
factorial design. No simulator timing will be interpreted as quantum speed.

## Scientific guardrails carried forward

- Feasibility, encoded optimality, and realized utility remain separate.
- The primary controlled deadline is derived from point predictions and held
  fixed across uncertainty regimes; uncertainty is not folded into the
  deadline term.
- The uncertainty regimes must alter the residual-generating distribution,
  not merely rescale a reported result after optimization.
- Exact enumeration remains the ground truth for every K=4/8 problem.
- Negative findings and incomplete/failed trials remain visible.
- Stable scientific modules and the Streamlit UI remain unchanged unless a
  documented defect is discovered.

## Step 9.2 execution audit

The replacement primary experiment completed 1,800 unique trial IDs with zero
failed checkpoints. Final artifact validation found 300 unique problem hashes,
900 matched X/XY pairs, 180 balanced factorial cells, finite required metrics,
non-negative energy regret, normalized probability masses to floating-point
tolerance, and complete 16-record training histories for every trial. The
post-finalization repository suite passed 151 tests before subsequent stages.

One execution-control incident occurred during the resumable run. A shell
timeout left an earlier three-worker process tree active, and a second launch
temporarily recomputed deterministic trial IDs. The exact parent/child process
trees were identified and stopped; the final segment resumed only the 880
missing checkpoint IDs. Atomic writes, identical problem hashes, deterministic
seeds, and a final inventory of exactly 1,800 unique checkpoint filenames mean
the numerical corpus contains neither duplicated observations nor substituted
runs. The incident wasted simulator time but did not alter scientific values.

The saved trials report 9,891.64 cumulative optimizer wall-seconds summed over
workers. Checkpoint timestamps span 64.3 minutes from the first to last unique
saved trial, including interruption/cleanup overhead. These are provenance
measurements, not evidence of quantum or classical speed.

During the pre-run design audit, the unfinished draft was found to compute
separate K=4 and K=8 deadlines. That would have confounded candidate-set size
with the operational threshold. The new runner computes one K=8-pool deadline
per network and uses it for K=4/8 and all uncertainty regimes. Because the old
Step 9.2 draft had never produced an accepted numerical result, no historical
scientific result is affected. A regression test now enforces shared deadlines
and K=4-as-prefix candidate construction.

## Step 9.7 source-replay defect

The first adaptive-conformal execution stopped before writing results because
its static-reproduction guard found identical conformal half-widths but
different interval centers on 628 replayed rows. Investigation showed that
decimal CSV serialization perturbed 25,159 numeric cells by at most
7.11e-15. Some perturbed features lay exactly on fitted histogram-gradient-
boosting split thresholds, producing prediction changes as large as 2.376
minutes when the saved CSV was passed back through the model.

This is a reproducibility defect in the historical storage contract, not a
defect in the accepted Step 9 numerical results: those results and their saved
predictions/intervals were computed from the original in-memory values.
Regenerating those values with the hash-matching accepted Step 9 runner,
configuration, and Python sources reproduces every saved prediction to
3.55e-15. Step 9.7 now regenerates the exact deterministic source floats,
checks IDs/metadata exactly and saved numeric values to 1e-12, and then demands
static interval reproduction to 1e-9. A regression test enforces this path.

## Standard-X penalty checkpoint defect

The first penalty-sensitivity launch produced zero accepted checkpoints. In 42
completed attempts, scientifically undefined diagnostics (for example an
approximation-ratio field represented as `NaN`) reached a strict JSON writer
that deliberately rejects nonstandard IEEE tokens. The optimizer results were
therefore rejected before checkpoint creation. The worker tree was stopped,
and every failure record was retained under
`results/research/standard_x_penalty_sensitivity/failures/`.

The runner now recursively converts non-finite undefined diagnostics to JSON
`null`, while preserving finite NumPy scalars as ordinary JSON numbers and
booleans. No problem specification, penalty, seed, optimizer setting, or
scientific calculation changed. A regression test covers Python/NumPy NaN,
infinity, integer, boolean, and nested values. Finalization hashes the preserved
failure records in `prior_failure_hashes.json` and reports their count, so this
execution failure cannot disappear from the accepted provenance record.
