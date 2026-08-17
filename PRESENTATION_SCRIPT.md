# Presentation Script

## Slide 1 — Question

“Q-Rescue studies a narrow but difficult question: can we combine travel-time prediction, calibrated uncertainty, candidate-route optimization, and QAOA without hiding feasibility or transfer failures?”

## Slide 2 — Pipeline

“We predict time for K candidate paths, calibrate a conformal interval, form an explicit risk score, encode exactly-one selection as a QUBO, and compare classical selection with standard-X and feasible-XY QAOA. A dynamic simulator evaluates realized outcomes.”

## Slide 3 — Generalization protocol

“I separate in-distribution testing, held-out topology, shifted traffic, combined shift, leave-one-network-out folds, and cached real road topology. The real graphs are authentic OpenStreetMap topology, but their traffic is simulated.”

## Slide 4 — Calibration result

Show [Figure 1](paper/figures/fig01_cross_network_coverage.png).

“Local 90% coverage is 91.2% on TEST-ID and 90.3% under topology shift, but only 47.5% under traffic shift and 50.4% under combined shift. The intervals widen but still fail. I treat this as a central negative result.”

## Slide 5 — Held-out networks and real topology

Show [Figures 2–3](paper/figures/fig02_lono_transfer.png).

“Across 20 leave-one-network-out folds, pooled local coverage is 90.0%, but difficult sparse networks fail. Transfer to four real road graphs is worse: the frozen model has negative R-squared and only 54.4% coverage. Engineering transfer succeeded; statistical transfer failed.”

## Slide 6 — QAOA feasibility

Show [Figure 4](paper/figures/fig04_qaoa_method_depth.png).

“The clearest quantum-method result is feasibility, not advantage. Feasible-XY stays at validity one, while standard-X loses feasibility. At K=8 and p=3 the comparison is 1.000 versus 0.023. These are ideal-simulator results with unequal resource structures.”

## Slide 7 — Dynamic routing

Show [Figure 6](paper/figures/fig06_dynamic_policy.png).

“Risk-aware QAOA did not improve the controlled dynamic task. Expected-time routing violated 2.86% of deadlines; risk QAOA violated 17.14% and rerouted more. This prevents a superiority claim.”

## Slide 8 — Robustness and baselines

“The QUBO penalty has an empirical failure boundary: below minimum candidate cost the all-zero state wins; equality ties; just above it restores validity on tested instances. Removing the reroute penalty nearly doubles reroutes without changing deadline violations. Exact-risk classical selection recovers 15 of 15 optima.”

## Slide 9 — Contribution

“The contribution is an evidence-traceable research framework and a set of defensible positive and negative findings: constraint-preserving feasibility, explicit shift failure, real-topology transfer failure, and dynamic-policy failure.”

## Slide 10 — Boundaries and next step

“I do not claim quantum advantage, hardware performance, real-traffic accuracy, or operational benefit. Step 9.2 and the statistical, adaptive-conformal, and local-noise follow-ups are complete. Step 10 is worthwhile only as a small predeclared hardware-feasibility study after native compilation, not as an advantage demonstration.”

## Closing line

“The honest result is not that quantum routing already works; it is that we now know exactly which parts of this hybrid pipeline are validated, which fail, and what a credible hardware study must measure next.”
