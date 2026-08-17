# Q-Rescue MVP GUI Guide

## Purpose

The Streamlit application is a presentation layer over the validated Q-Rescue Step 9.5 backend subset. Publication-freeze Steps 9.2 and 9.6–9.8 remain artifact/document workflows and were intentionally not added as product features. The GUI does not reimplement the routing, uncertainty, QUBO, or QAOA algorithms. Its central question is:

> Given this emergency scenario, what route does each method recommend, and why?

The application is a research instrument. It is not a dispatch system, a medical device, a live-traffic service, or evidence of quantum advantage.

## Install

Use Python 3.12 in a clean virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-research.txt
```

`requirements-research.txt` includes the scientific dependencies plus the Streamlit runtime. The pinned Starlette version is required by Streamlit 1.57.

## Run

From the repository root:

```powershell
streamlit run app.py
```

Streamlit prints the local URL, normally `http://localhost:8501`.

## Default demo scenario

Select **LOAD DEMO SCENARIO** to restore the reproducible presentation configuration:

- network: synthetic grid, validated replicate `grid_r0`;
- origin/destination: the graph's saved benchmark pair;
- traffic: rain;
- deadline: 8 minutes;
- candidate routes: K = 4;
- active method: feasible-XY QAOA;
- QAOA depth: p = 1.

The controls and candidate-route preview load immediately. Expensive optimization begins only after **RUN Q-RESCUE**.

## What each panel means

### Pipeline strip

The strip separates the classical graph and prediction stages from the reduced quantum optimization stage:

1. road network;
2. AI travel-time prediction;
3. conformal uncertainty;
4. deadline risk;
5. QUBO;
6. QAOA or classical solver;
7. decoded route decision.

The graph itself is never encoded node-by-node on qubits. Muted stages are intentionally
inactive for the selected method: for example, the expected-time baseline does not use the
conformal, deadline-risk, or QUBO stages to choose its route.

### Scenario controls

- **Network** selects one of five deterministic synthetic topology families or one of four cached real-road case studies.
- **Origin / destination** defaults to each graph's reproducible benchmark pair. Custom nodes are allowed when at least four routes exist.
- **Traffic scenario** uses the repository's presets: normal, rush hour, rain, accident, road closure, or prediction noise.
- **Emergency deadline** is expressed in minutes.
- **K** enables only validated values (4, 6, 8) that the current graph and O/D pair can support.
- **Routing method** selects expected-time, exact risk-aware classical, standard-X QAOA, or feasible-XY QAOA.
- **QAOA depth** selects p = 1, 2, or 3 and also controls the same-scenario comparison runs.

Changing widgets does not run QAOA. The form submission is the execution boundary.

### Network view

Muted edges are the road graph. Candidate paths receive a secondary gray overlay. The active recommendation receives the only cyan route glow. Origin and destination have distinct marker shapes and text labels. Accident and closure edges receive an ember dashed overlay plus an `×` marker.

Synthetic coordinates come from deterministic generators. Cached real case studies preserve WGS84 geometry and show the required **real road topology + simulated traffic** warning.

### Candidate-route table

Every row comes from the active pipeline and retains:

- distance;
- point prediction;
- lower and upper conformal bounds;
- realized interval width;
- upper-bound deadline margin;
- risk score;
- fastest, lowest-risk, and recommended labels.

Method disagreements remain visible.

### Recommended-route card

The deadline label is determined by the backend's conservative rule:

- **LIKELY WITHIN DEADLINE** when the conformal upper bound does not exceed the configured deadline;
- **DEADLINE RISK** when the conformal upper bound exceeds it.

This is a routing-risk status, not a clinical interpretation.

### Quantum result

The quantum panel shows analytic ideal-simulator results for the selected mixer:

- K and number of qubits;
- QAOA depth;
- valid-solution probability;
- exact-optimum probability;
- most probable valid bitstring;
- decoded route;
- expected QUBO energy;
- expanded circuit depth and gate count;
- optimization iterations.

Valid probability sums every one-hot route state. Optimal probability is the probability of only the exact minimum-risk route state. They are intentionally separate.

### Classical-versus-quantum comparison

For every submitted scenario, the application evaluates all four methods against the same candidate table and QUBO. Quantum rows are not visually promoted when they perform worse. Classical probability fields are shown as not applicable.

### Research results

The second tab reads headline values from the saved CSV artifacts at runtime. It includes:

- cross-network and shifted conformal coverage;
- pooled leave-one-network-out coverage;
- real-topology transfer;
- K=8, p=3 mixer feasibility;
- dynamic deadline violations;
- reroute-penalty ablation.

If an expected artifact is missing or ambiguous, the interface shows **Result unavailable**.

### Optional disruption simulation

After a static run, the disruption expander invokes the existing edge-by-edge simulator for accident, congestion, closure, or rain. It reports the route before the event, the updated remaining route, backend reroute count, minute-equivalent switching cost, and final deadline margin. This is controlled simulated traffic.

## Backend function map

| GUI action | Existing validated backend function or artifact |
|---|---|
| Build synthetic graph | `src.synthetic_networks.generate_synthetic_network` |
| Load cached real graph | `src.network_loader.load_network_graphml` |
| Validate graph | `src.network_schema.validate_network` |
| Generate K candidates | `src.candidate_routes.generate_candidate_routes` |
| Resolve traffic preset | `src.traffic_scenarios.get_scenario` |
| Apply congestion, incident, rain, closure | `src.dynamic_rerouting.apply_dynamic_event` |
| Build features, predict, interval, and score | `src.dynamic_rerouting.prepare_dynamic_candidates` |
| Point prediction | serialized `src.traffic_model.FittedTrafficModel` |
| Conformal interval | serialized/reconstructed `src.conformal.LocallyScaledSplitConformalRegressor` |
| Deadline-risk components | `src.deadline_risk.score_route_table` (called by the dynamic candidate pipeline) |
| Expected-time selection | `src.classical_baselines.select_shortest_expected_time_candidate` |
| Exact risk selection | `src.classical_baselines.select_exact_risk_candidate` |
| Generalized QUBO | `src.risk_qubo.build_risk_qubo` |
| Exact QUBO enumeration | `src.risk_qubo.brute_force_risk_qubo` |
| QUBO-to-Ising conversion | `src.risk_qubo.risk_qubo_to_ising` |
| Standard-X and feasible-XY circuits | `src.feasibility_qaoa.create_expectation_qnode` and `create_probability_qnode` |
| QAOA optimization | `src.feasibility_qaoa.optimize_qaoa_angles` |
| Probability metrics | `src.feasibility_qaoa.summarize_probabilities` |
| Circuit resource metrics | `src.feasibility_qaoa.circuit_resource_record` |
| Dynamic traversal | `src.dynamic_rerouting.simulate_dynamic_route` |
| Research findings | validated CSVs under `results/research/` |
| Supported/unsupported claims | `paper/CLAIMS.md` |

The UI adapter in `ui/backend.py` is orchestration only. It loads frozen artifacts, passes parameters to these functions, and packages their return values.

## What is classical and what is quantum

Classical:

- graph loading and topology handling;
- candidate-route generation;
- traffic-feature construction;
- machine-learning prediction;
- conformal calibration and interval prediction;
- risk-score construction;
- QUBO construction;
- exact enumeration used as ground truth;
- decoding and visualization.

Quantum simulation:

- K route-selection variables are mapped to K qubits;
- the Ising cost Hamiltonian is evaluated with standard-X or feasibility-preserving XY QAOA;
- the most probable valid one-hot state is decoded back to a candidate route.

No physical quantum hardware is used.

## Route encoding

For K candidate routes:

```text
x_i = 1  route i is selected
x_i = 0  route i is not selected
```

Exactly one bit should be 1. For K = 4, the valid states are `1000`, `0100`, `0010`, and `0001`. The current one-hot formulation therefore uses K variables and K qubits.

The QUBO is:

```text
Q(x) = Σ_i R_i x_i + P(Σ_i x_i - 1)²
```

`R_i` contains the configured expected-time, conformal half-width, deadline-exceedance, and optional reroute terms. `P` is chosen by the existing safe-penalty function.

## Performance and caching

- Graphs and frozen model bundles use Streamlit resource caching.
- Complete submitted scenarios use data caching keyed by network, O/D, traffic scenario, deadline, K, method, and depth.
- QAOA is never triggered by an ordinary widget rerun.
- K=8, p=3 is materially more expensive than K=4, p=1 because the analytic state space and circuit both grow.
- The optional dynamic quantum policy can run QAOA at several edge decisions and is therefore the slowest interactive path.

### Ambient visual layer

- The cyan/blue/violet pointer glow is installed with Streamlit's supported non-iframed `st.html(..., unsafe_allow_javascript=True)` API using only static, repository-owned JavaScript.
- Pointer coordinates remain entirely in the browser. A passive `pointermove` listener stores only the latest local coordinates, and `requestAnimationFrame` applies CSS custom properties at most once per display frame.
- The glow does not update `session_state`, call Python, invoke a Streamlit callback, trigger QAOA, write data, or make a network request.
- The installer is idempotent: it reuses one `#q-pointer-glow` element and removes its previous listeners on a Streamlit rerun.
- It targets only the standard browser document/body and its own element ID; it does not traverse undocumented Streamlit component selectors.
- If JavaScript is unavailable, the static layered graphite/cyan/blue/violet background remains as the safe fallback.
- `prefers-reduced-motion: reduce` and coarse/touch pointers disable the pointer layer while preserving the static theme.

## Accessibility

- Text labels accompany every color state.
- Origin and destination use different marker shapes.
- Risk and selection remain explicit in tables and badges.
- Focus-visible outlines are enabled for keyboard controls.
- motion is disabled under `prefers-reduced-motion`;
- charts have visible titles, exact labels, and restrained non-color hierarchy.

## Known limitations

- No demonstrated quantum advantage or speedup.
- No physical-device run.
- No observed live-traffic validation.
- Real-topology statistical transfer was unfavorable in the saved experiments.
- No operational medical or dispatch deployment claim.
- Custom O/D pairs may not provide K candidate paths.
- A closure can reduce route availability below K and is rejected rather than silently changing K.
- The UI uses fixed validated model and calibration artifacts; it does not retrain.
- Dynamic simulations are controlled synthetic events and do not model a complete city or ambulance fleet.
