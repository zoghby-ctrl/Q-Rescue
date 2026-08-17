# Next Step: Hardware

## Objective

Test whether the validated one-hot QUBO and feasible-XY construction remain usable under native-gate compilation and device noise. The objective is hardware feasibility and error characterization, not advantage.

## Predeclared instance set

- Use a small K=4 subset selected before hardware results are seen.
- Include exact classical ground truth, at least one close-score instance, and more than one topology family.
- Freeze risk vectors, QUBO penalty, seeds, optimizer budget, shot count, and success metrics.
- Exclude Step 9.2 until its classical exact-choice interaction is independently resolved.

## Simulator gates before hardware

1. Compile standard-X and feasible-XY circuits to the target device’s native basis and connectivity.
2. Report logical depth, transpiled depth, one- and two-qubit gate counts, SWAPs, state-preparation cost, and qubit layout.
3. Run ideal and device-noise-model simulations with identical shots.
4. Continue only if the compiled circuits complete and retain measurable feasible probability.

## Hardware protocol

- Record provider, backend, date/time, calibration snapshot, topology, queue time, execution time, shots, transpiler settings, and job identifiers.
- Use multiple jobs or calibration windows rather than one opportunistic run.
- Preserve raw counts before mitigation.
- Report raw and mitigated validity, optimal probability, most-probable exact recovery, energy, and confidence intervals over repeated jobs.
- Retain failed jobs and compilation failures in the denominator or failure log.

## Comparators

- Exact enumeration and exact-risk candidate selection.
- Standard-X versus feasible-XY using the same instances and declared shot budget.
- Ideal simulator, noise-model simulator, and hardware.
- Optional valid warm start only if it comes from a documented relaxation or heuristic and does not use the exact solution.

## Stop conditions

Stop and report a negative result if native compilation is infeasible, two-qubit depth is clearly beyond the device’s calibrated regime, job failure prevents the predeclared matrix, or feasible/optimal probabilities are indistinguishable from a simple sampling baseline. Do not expand the sweep to search for favorable cells.

## Claims boundary

A successful Step 10 may support “small-instance feasibility on named hardware under documented calibration.” It will not by itself support speedup, scaling, operational routing, or quantum advantage.
