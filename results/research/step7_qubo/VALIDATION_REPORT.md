# Step 7 validation report

## Overall assessment: Ready for quantum-method benchmarking

Thirty arbitrary route-risk vectors were encoded: five networks, K=4/6/8,
and objective variants D and E. Exhaustive search evaluated all 3,360 states.
Every ground state was one-hot valid and matched direct risk-vector argmin.

## Formulation

For costs c_i and binary x_i:

`Q(x) = sum_i c_i x_i + P (sum_i x_i - 1)^2`.

The expansion has constant P, linear coefficient c_i-P, and pair coefficient
2P. The builder requires P greater than the maximum route risk; the accepted
automatic value is max(c)+1. This is a conservative sufficient condition that
makes every valid route cheaper than the all-zero state for nonnegative costs.

## Verification

- K=4: 10 instances × 16 states; maximum QUBO/Ising discrepancy 1.14e-13.
- K=6: 10 instances × 64 states; maximum discrepancy 1.36e-12.
- K=8: 10 instances × 256 states; maximum discrepancy 1.18e-11.
- Direct route argmin and QUBO ground route agreed in 30/30 cases.
- Valid one-hot ground state in 30/30 cases.

The discrepancies are rounding at large coefficient scales, not logical
energy differences.

## Scaling

| K/qubits | Search states | QUBO terms incl. constant | Hamiltonian nonidentity terms |
|---:|---:|---:|---:|
| 4 | 16 | 11 | 10 |
| 6 | 64 | 22 | 21 |
| 8 | 256 | 37 | 36 |

## Required caveats

- Exhaustive verification is practical only at these small K values.
- A safe penalty can widen Hamiltonian coefficient ranges and make variational
  optimization harder; penalty sensitivity belongs in Step 9.4.
- Correct encoding does not imply QAOA will sample the optimum reliably.
- No runtime or quantum-advantage claim follows from this table.

## Decision

Proceed to standard X-mixer and feasibility-preserving XY-style QAOA using the
same verified Hamiltonians and exact optima as ground truth.

