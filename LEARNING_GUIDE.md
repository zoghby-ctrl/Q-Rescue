# Q-Rescue Learning Guide

This guide explains the prototype from ordinary routing language through the QAOA circuit. The central habit is to keep three objects separate:

1. the route decision and its risk score;
2. the QUBO cost assigned to every binary string;
3. the probability distribution produced by the quantum circuit.

The brute-force solver proves the answer for this tiny instance. QAOA is an experiment on the same cost function, not a replacement for that proof.

## Core routing ideas

### Graph

A graph represents places as **nodes** and road segments as **edges**. The prototype uses a directed NetworkX graph because travel is represented from Ambulance Station toward Hospital. Every edge stores a travel time in minutes.

### Route

A route is an ordered path through the graph. Route A is:

```text
Ambulance Station -> Junction A -> Hospital
```

The edge times 8 and 10 add to 18 expected minutes. The other three routes use the matching B, C, and D junctions.

### Expected travel time

Expected travel time is the central or average travel-time estimate used by the simple baseline. The four values are 18, 21, 24, and 20 minutes. Therefore the expected-time baseline chooses Route A.

### Uncertainty

Uncertainty describes how imprecise a travel-time estimate is. Here the values 10, 2, 3, and 7 minutes are **manually simulated teaching inputs**. They are not standard deviations, confidence intervals, or calibrated prediction intervals. A later milestone will define and validate machine-learning prediction intervals.

### Risk score

Before using code, define the score:

```text
risk_score_i = expected_time_i + uncertainty_weight * uncertainty_i
```

At weight 1.0, the scores are:

```text
Route A: 18 + 1(10) = 28
Route B: 21 + 1( 2) = 23
Route C: 24 + 1( 3) = 27
Route D: 20 + 1( 7) = 27
```

Route B has the smallest score. The score uses minute-equivalent units and expresses a chosen preference for less uncertainty; it is not a probability.

## Binary optimization ideas

### Binary variable

A binary variable is either zero or one. Q-Rescue assigns:

```text
x_0 = 1 when Route A is selected
x_1 = 1 when Route B is selected
x_2 = 1 when Route C is selected
x_3 = 1 when Route D is selected
```

Exactly one bit must be one. In this project, wire and bit order is always `x_0 x_1 x_2 x_3`; `0100` is Route B. Nothing is reversed for display.

For a binary value, `x_i^2 = x_i`. Checking both cases proves it: `0^2 = 0` and `1^2 = 1`. This fact turns the squared constraint into a quadratic polynomial.

### QUBO

QUBO means **quadratic unconstrained binary optimization**. “Unconstrained” means validity is encouraged through cost rather than enforced by an external constraint solver. The prototype minimizes:

```text
C(x) = sum_i(risk_score_i * x_i)
       + penalty * (sum_i(x_i) - 1)^2
```

The first sum is the route objective. The squared expression is zero only when one route is selected.

### Penalty

Let `S = sum_i x_i`. The penalty is `P(S - 1)^2`. Expand it before coding:

```text
P(S - 1)^2 = P(S^2 - 2S + 1)
```

For four binary variables,

```text
S^2 = sum_i x_i^2 + 2 * sum_(i<j) x_i*x_j
    = sum_i x_i   + 2 * sum_(i<j) x_i*x_j
```

Substitution gives:

```text
P(S - 1)^2 = P - P*sum_i x_i + 2P*sum_(i<j) x_i*x_j
```

Therefore:

- `P` is the constant term;
- `risk_score_i - P` is the linear coefficient of `x_i`;
- `2P` is the quadratic coefficient of every `x_i*x_j` pair.

With `P = 50`, the constant is 50, the linear coefficients are `[-22, -27, -23, -23]`, and each of the six pair coefficients is 100.

Selecting zero routes is invalid because no ambulance route is chosen; its penalty is `50(0 - 1)^2 = 50`. Selecting two or more routes is invalid because the model asks for one decision; two routes add `50(2 - 1)^2 = 50`, three add 200, and four add 450. The penalty must be large enough that an invalid state cannot lower the total below a valid state. Here the smallest invalid cost is 50, while valid route costs are 23 through 28.

The displayed symmetric matrix uses

```text
C(x) = constant + x^T Q x
```

Each off-diagonal matrix entry is 50, half of the explicit pair coefficient 100, because `x^T Q x` includes both `Q_ij x_i x_j` and `Q_ji x_j x_i`.

## From binary variables to quantum observables

### Ising variable

A computational-basis qubit has a Pauli-Z eigenvalue of `+1` in state `|0>` and `-1` in state `|1>`. The manual substitution is:

```text
x_i = (1 - Z_i) / 2
```

When `Z_i = +1`, this gives `x_i = 0`. When `Z_i = -1`, it gives `x_i = 1`.

Apply the rule separately to each QUBO term:

```text
a_i x_i = a_i/2 - (a_i/2) Z_i
```

and

```text
b_ij x_i x_j
= b_ij/4 * (1 - Z_i - Z_j + Z_i Z_j)
```

Collecting terms produces an identity offset, one-qubit `Z_i` coefficients, and two-qubit `Z_i Z_j` coefficients. The source code performs these arithmetic steps directly. It returns the identity offset separately, so:

```text
QUBO cost = Hamiltonian basis-state energy + offset
```

For this instance the offset is 152.5. Omitting it during angle optimization is safe because adding the same constant to every state does not change which state has the smallest cost or the optimizer's gradient.

### Pauli-Z

`qml.PauliZ(wire)` is the observable that distinguishes computational-basis zero from one using eigenvalues `+1` and `-1`. A product such as `qml.PauliZ(0) @ qml.PauliZ(1)` supplies the `Z_0 Z_1` interaction created by a quadratic QUBO term. These operations are diagonal, so the cost Hamiltonian assigns a cost-related phase without directly flipping bits.

### Hamiltonian

A Hamiltonian here is a weighted sum of Pauli observables:

```text
H_C = sum_i h_i Z_i + sum_(i<j) J_ij Z_i Z_j
```

It encodes the QUBO energy landscape apart from the recorded constant offset. It is a mathematical cost operator, not a model of ambulance physics.

## QAOA circuit ideas

### Superposition and the Hadamard gate

The circuit applies `qml.Hadamard` to every qubit. A Hadamard maps `|0>` to an equal superposition of `|0>` and `|1>`. Four Hadamards therefore start with equal amplitudes over all 16 four-bit strings.

### Cost layer

`qml.qaoa.cost_layer(gamma, cost_hamiltonian)` applies cost evolution:

```text
U_C(gamma) = exp(-i * gamma * H_C)
```

This gives different phases to bitstrings according to their cost-Hamiltonian energies. It does not by itself turn those phases into larger or smaller measurement probabilities.

PennyLane displays this layer as `ApproxTimeEvolution`. Its product-formula approximation uses one step. In this prototype all Z and ZZ terms inside the cost Hamiltonian commute with one another, so that one-step factorization is exact for the cost layer.

### Mixer layer and Pauli-X

The mixer Hamiltonian is:

```text
H_M = X_0 + X_1 + X_2 + X_3
```

`qml.PauliX` flips `|0>` and `|1>`. `qml.qaoa.mixer_layer(beta, mixer_hamiltonian)` applies:

```text
U_M(beta) = exp(-i * beta * H_M)
```

The mixer turns cost-dependent phase differences into interference and lets probability move between bitstrings. The standard X mixer does not preserve the exactly-one subspace, so invalid samples can occur.

PennyLane also displays the mixer layer as `ApproxTimeEvolution`. The separate `X_i` terms commute because they act on different wires, so its one-step factorization is exact here as well.

### Gamma

Gamma is the adjustable angle controlling each cost layer. It determines how strongly cost energies affect phases at that layer.

### Beta

Beta is the adjustable angle controlling each mixer layer. It determines how strongly amplitudes mix between bitstrings.

### QAOA depth

Depth `p` is the number of cost-layer/mixer-layer pairs. At `p = 1` there is one gamma and one beta. At `p = 2` there are two of each. Greater depth adds flexibility but also more gates and parameters; it is not guaranteed to improve a finite optimization run.

### Expectation value

The analytic training QNode returns `qml.expval(cost_hamiltonian)`. An expectation value is the probability-weighted average Hamiltonian energy over the circuit's complete distribution. The classical optimizer tries to reduce this average. It is not the cost of a single measured route.

### Classical optimizer

PennyLane's Adam optimizer calculates gradients of the analytic QNode and updates gamma and beta on an ordinary computer. The fixed starting values are 0.02 for every gamma and -0.20 for every beta; the step size is 0.02 and there are 100 iterations.

### Measurement

The separate finite-shot QNode returns `qml.sample(wires=[0, 1, 2, 3])`. One shot produces four zeros or ones. Repeating 4,000 times estimates a probability distribution. Seed 2026 makes this teaching run reproducible, but other seeds can change close rankings.

### Bitstring

A bitstring is the displayed sample, such as `0100`. Position zero is wire 0 and variable `x_0`; position three is wire 3 and variable `x_3`. The conversion helper joins the sample in the order returned and never reverses it.

### Valid solution

A valid sample contains exactly one `1`. The valid-solution probability is:

```text
number of one-hot samples / total number of shots
```

The optimal-route probability is the count of `0100` divided by total shots. The most frequent valid route is found only from one-hot counts; code does not assume it must be Route B.

### Why QAOA is hybrid

QAOA alternates between two kinds of work:

- the quantum circuit prepares a parameterized distribution and supplies an expectation value or samples;
- the classical Adam optimizer updates gamma and beta using the analytic expectation value.

The loop is therefore hybrid quantum-classical. In this repository both parts run on PennyLane's classical `default.qubit` simulator.

## Source-file reference

### `src/route_data.py`

- `create_route_dataframe() -> pandas.DataFrame`: takes no inputs. Returns four rows with route name, expected minutes, and manually simulated uncertainty minutes.
- `create_road_network() -> networkx.DiGraph`: takes no inputs. Returns the six-node directed graph with `time_minutes` on every edge.
- `get_candidate_paths() -> dict[str, list[str]]`: takes no inputs. Returns each route name mapped to its ordered node path.
- `calculate_path_time(graph, path) -> float`: takes the graph and an ordered node sequence. Returns the sum of its edge times in minutes.

### `src/risk_model.py`

- `calculate_risk_score(expected_time_minutes, uncertainty_minutes, uncertainty_weight=1.0) -> float`: takes the three values in the risk equation. Returns expected time plus weighted uncertainty and rejects a negative weight.
- `add_risk_scores(route_data, uncertainty_weight=1.0) -> pandas.DataFrame`: takes the route table and weight. Returns a copy with a `risk_score` column.
- `choose_expected_time_route(route_data) -> str`: takes the route table. Returns the name in the row with minimum expected time.
- `choose_risk_aware_route(scored_routes) -> str`: takes a table containing `risk_score`. Returns the name in the minimum-score row.

### `src/qubo.py`

- `build_qubo(risk_scores, penalty=50.0) -> tuple`: takes one score per variable and the constraint penalty. Returns the constant float, linear NumPy array, pair-coefficient dictionary, and labeled symmetric pandas matrix.
- `evaluate_qubo(assignment, constant, linear, quadratic) -> float`: takes one bit assignment and the displayed coefficients. Returns its full QUBO cost.
- `calculate_constraint_penalty(assignment, penalty=50.0) -> float`: takes a bit assignment and penalty strength. Returns `P(sum(x)-1)^2`.
- `is_valid_assignment(assignment) -> bool`: takes a bit assignment. Returns true exactly when one bit is one.
- `qubo_to_ising(constant, linear, quadratic) -> tuple`: takes the QUBO coefficients. Returns the PennyLane Hamiltonian without its identity term, the identity offset, the one-qubit Z coefficients, and the two-qubit ZZ coefficients.
- `calculate_ising_energy(assignment, offset, z_coefficients, zz_coefficients) -> float`: takes a basis assignment and the converted coefficients. Returns the full Ising energy including offset for a direct arithmetic check.

### `src/classical_solver.py`

- `generate_binary_assignments(number_of_variables=4) -> list[tuple]`: takes a variable count. Returns all binary tuples in left-to-right product order.
- `decode_assignment(assignment, route_names) -> str | None`: takes bits and matching route names. Returns the selected name for a one-hot assignment and otherwise `None`.
- `evaluate_all_assignments(route_names, risk_scores, penalty=50.0) -> pandas.DataFrame`: takes route names, scores, and penalty. Returns all assignments with bits, validity, objective, penalty, total cost, and selected labels.
- `solve_qubo(route_names, risk_scores, penalty=50.0) -> tuple`: takes the same inputs. Returns the full table, minimum-cost bit tuple, and decoded route name.

### `src/quantum_qaoa.py`

- `create_mixer_hamiltonian(number_of_qubits=4) -> qml.Hamiltonian`: takes the qubit count. Returns the sum of one Pauli-X observable per wire.
- `create_training_qnode(cost_hamiltonian, mixer_hamiltonian, depth, number_of_qubits=4) -> qml.QNode`: takes both Hamiltonians, QAOA depth, and qubit count. Returns an analytic expectation-value QNode.
- `create_sampling_qnode(cost_hamiltonian, mixer_hamiltonian, depth, shots=2000, seed=2026, number_of_qubits=4) -> qml.QNode`: takes the circuit description and sampling settings. Returns a finite-shot sample QNode with direct wire order.
- `optimize_qaoa(training_qnode, depth, iterations=100, step_size=0.02, initial_gammas=None, initial_betas=None) -> tuple`: takes the analytic QNode and optimizer settings. Returns optimized PennyLane NumPy gamma and beta arrays plus a pandas training-history table.
- `sample_to_bitstring(sample) -> str`: takes one ordered sample. Returns its characters without reversing them.
- `sample_to_route(sample, route_names) -> str | None`: takes one ordered sample and route names. Returns its route if exactly one bit is set; otherwise `None`.
- `summarize_samples(samples, route_names, optimal_route) -> dict`: takes the complete sample array, ordered route names, and classically verified optimum. Returns all 16 counts, valid route counts, valid probability, optimum probability, and the measured most frequent valid route.

### `src/visualization.py`

- `save_road_network_figure(graph, candidate_paths, output_path) -> None`: saves four network panels, each with one route highlighted.
- `save_route_comparison_figure(route_data, output_path) -> None`: saves grouped expected-time and risk-score bars.
- `save_qubo_matrix_figure(qubo_matrix, output_path) -> None`: saves a labeled coefficient heatmap.
- `save_qaoa_circuit_figure(qnode, gammas, betas, output_path) -> None`: uses PennyLane's Matplotlib drawer and saves the instantiated QAOA circuit.
- `save_training_history_figure(histories, output_path) -> None`: saves one cost-expectation curve per depth.
- `save_measurement_probabilities_figure(summaries, optimal_bitstring, output_path) -> None`: saves all bitstring probabilities for each depth and colors optimal, other valid, and invalid states separately.

## How to explain the result carefully

The strongest defensible summary is:

> On this four-route synthetic instance, exhaustive search verifies that uncertainty weight 1.0 changes the choice from Route A to Route B. A small simulator-based QAOA experiment can place measurable probability on that optimum, but its distribution depends on depth, initialization, optimization, and finite-shot randomness. This is a correctness and teaching prototype, not evidence of quantum advantage.
