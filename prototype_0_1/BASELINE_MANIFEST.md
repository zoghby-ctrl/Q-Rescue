# Prototype 0.1 preservation manifest

This directory marks the historical Q-Rescue Prototype 0.1 boundary. The
prototype itself remains in its original locations because the executed
notebook imports the original `src` modules and its saved figures/tables use
the original `results` paths.

Protected historical files:

- `src/classical_solver.py`
- `src/quantum_qaoa.py`
- `src/qubo.py`
- `src/risk_model.py`
- `src/route_data.py`
- `src/visualization.py`
- `tests/test_classical_solver.py`
- `tests/test_qubo.py`
- `tests/test_risk_model.py`
- `notebooks/01_qrescue_prototype.ipynb`
- all files that existed in `results/` before the generalized research work
- `README.md`, `RESEARCH_PLAN.md`, `LEARNING_GUIDE.md`, and `requirements.txt`

Validated baseline (reconciled on 2026-08-10):

- 11 tests passed.
- Exact optimum: Route B / `0100` / QUBO cost 23.
- p=1: 63.625% valid and 16.700% optimal over 4,000 shots.
- p=2: 99.050% valid and 25.900% optimal over 4,000 shots.
- Route B was the most frequent valid sample at both depths.
- PennyLane version: 0.45.1; Python version: 3.12.10.

This workspace is not a Git checkout, so no historical commit hash is
available. Generalized experiments must write beneath `results/research/`
and must not overwrite the files listed above.
