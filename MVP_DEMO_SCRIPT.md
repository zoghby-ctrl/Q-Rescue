# Q-Rescue 2-Minute TA Demo

## 0:00–0:15 — Load the scenario

1. Run `streamlit run app.py`.
2. Select **LOAD DEMO SCENARIO**.
3. Point out the deterministic synthetic grid, rain scenario, 8-minute deadline, K=4, and feasible-XY QAOA at p=1.

Say: “The graph and candidate paths are generated classically. The quantum problem begins only after those paths become four binary route-choice variables.”

## 0:15–0:35 — Explain the classical AI

Select **RUN Q-RESCUE**, then open **How did the AI predict this?**

Say: “The frozen classical model receives network and observable traffic features and predicts travel time. Held-out calibration residuals produce a locally scaled split-conformal interval. The interval is not a confidence claim about arbitrary distribution shift.”

## 0:35–0:55 — Explain uncertainty and deadline risk

Use the candidate table and recommended-route card.

Say: “Each route retains its point prediction, lower and upper interval bounds, interval width, and conservative deadline margin. Deadline risk means the upper bound exceeds the configured deadline; it is not a clinical judgment.”

## 0:55–1:20 — Show QUBO and QAOA

Open **How was this encoded?** and then show the quantum result card.

Say: “K candidate routes become K one-hot variables and K qubits. The QUBO combines route risk with an exactly-one constraint. Feasible-XY stays inside the one-hot subspace in the ideal simulator; standard-X may allocate probability to invalid states.”

Point to **valid-solution probability** and **optimal-route probability** separately.

Say: “The recommendation is the most probable valid decoded route, not a claim that a quantum computer found the fastest route.”

## 1:20–1:40 — Compare methods honestly

Scroll to **Classical vs quantum · same scenario**.

Say: “All four methods see the same candidates. The table preserves disagreements. Exact classical risk is the ground-truth argmin for these small instances, and QAOA is not made to look superior when it misses it.”

## 1:40–2:00 — Show limitations

Open **RESEARCH RESULTS**, then **WHAT Q-RESCUE DOES NOT CLAIM**.

Say: “The saved artifacts show both favorable and unfavorable findings: feasibility preservation is strong, but shift coverage, real-topology transfer, and the tested dynamic risk-aware QAOA policy expose important failures. There is no quantum advantage, speedup, live-traffic validation, medical deployment, or hardware result.”

Optional if time: return to the console, open **SIMULATE DISRUPTION**, choose **Road closure**, and show the before/after route trace from the existing dynamic backend.

