"""Decision, quantum, encoding, and AI explanation panels."""

from __future__ import annotations

import html

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ui.backend import DynamicDemoResult, PipelineResult
from ui.network_view import build_probability_figure


def _minutes(value: float, *, signed: bool = False) -> str:
    return f"{value:+.2f} min" if signed else f"{value:.2f} min"


def render_decision_panel(result: PipelineResult) -> None:
    row = result.selected_row
    deadline_risk = bool(row["upper_bound"] > result.deadline_minutes)
    status = "DEADLINE RISK" if deadline_risk else "LIKELY WITHIN DEADLINE"
    card_class = "q-card selected risk" if deadline_risk else "q-card selected"
    badge_class = "risk" if deadline_risk else "good"
    st.markdown(
        f"""
        <section class="{card_class}" aria-label="Recommended route">
          <div class="q-eyebrow">RECOMMENDED ROUTE</div>
          <div class="q-route-id">{html.escape(str(result.selected_route_id))}</div>
          <div class="q-badge {badge_class}">{status}</div>
          <div class="q-kv">
            <span>Selected method</span><strong>{html.escape(result.selected_method_label)}</strong>
            <span>Predicted time</span><strong>{_minutes(float(row['prediction']))}</strong>
            <span>Uncertainty interval</span><strong>{float(row['lower_bound']):.2f}–{float(row['upper_bound']):.2f} min</strong>
            <span>Emergency deadline</span><strong>{_minutes(result.deadline_minutes)}</strong>
            <span>Deadline margin</span><strong>{_minutes(float(row['deadline_margin']), signed=True)}</strong>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="q-section-label">WHY THIS ROUTE?</div>', unsafe_allow_html=True)
    if result.selected_method == "expected_time":
        st.write(
            "It has the smallest model-predicted travel time among the K generated candidates. "
            "Uncertainty and deadline terms are shown, but they do not determine this baseline."
        )
    elif result.selected_method == "exact_risk":
        st.write(
            "It is the exact minimum of the current risk score: expected travel time plus the "
            "weighted conformal half-width and any conservative deadline exceedance."
        )
    else:
        quantum = result.quantum_results[result.selected_method]
        st.write(
            "This route decodes from the most probable valid one-hot route-selection state "
            f"returned by the configured {quantum.method_label} ideal-simulator experiment. "
            "It is not a claim that a quantum computer found the fastest route."
        )

    if result.selected_route_id != result.fastest_route_id:
        st.info(
            f"Method disagreement: the expected-time baseline selects {result.fastest_route_id}, "
            f"while the active method selects {result.selected_route_id}."
        )
    if result.selected_route_id != result.lowest_risk_route_id:
        st.warning(
            f"The active method did not select the exact risk optimum ({result.lowest_risk_route_id})."
        )


def render_quantum_panel(result: PipelineResult) -> None:
    if not result.is_quantum:
        return
    quantum = result.quantum_results[result.selected_method]
    st.markdown('<div class="q-section-label">QUANTUM RESULT</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <section class="q-card selected" aria-label="Quantum result">
          <div class="q-eyebrow">METHOD</div>
          <div style="font-weight:760; margin:.35rem 0 .75rem;">{html.escape(quantum.method_label)}</div>
          <div class="q-kv">
            <span>K / number of qubits</span><strong>{quantum.number_of_qubits} / {quantum.number_of_qubits}</strong>
            <span>QAOA depth</span><strong>p = {quantum.depth}</strong>
            <span>Valid-solution probability</span><strong>{quantum.valid_probability:.2%}</strong>
            <span>Optimal-route probability</span><strong>{quantum.optimal_probability:.2%}</strong>
            <span>Most probable valid state</span><strong>{html.escape(quantum.most_probable_valid_bitstring)}</strong>
            <span>Decoded route</span><strong>{html.escape(quantum.selected_route_id)}</strong>
            <span>Expected QUBO energy</span><strong>{quantum.expected_qubo_energy:.3f}</strong>
            <span>Circuit depth / operations</span><strong>{quantum.circuit_depth} / {quantum.gate_count}</strong>
            <span>Optimization iterations</span><strong>{quantum.iterations}</strong>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    figure = build_probability_figure(
        quantum.probability_table,
        title="One-hot route probabilities · analytic ideal simulator",
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption(
        "Valid-solution probability sums all one-hot states. Optimal-route probability is only "
        "the probability assigned to the exact minimum-risk one-hot state."
    )


def render_encoding_explainer(result: PipelineResult) -> None:
    with st.expander("How was this encoded?"):
        st.write(
            r"For candidate route $i$, $x_i=1$ means the route is selected and $x_i=0$ "
            r"means it is not. Exactly one route should be selected."
        )
        columns = st.columns(3)
        columns[0].metric("Candidate routes", result.k)
        columns[1].metric("Binary variables", result.k)
        columns[2].metric("Qubits", result.k)
        if result.k == 4:
            st.code("1000   0100   0010   0001", language=None)
            st.caption("Each valid bitstring selects exactly one of the four candidate routes.")

        st.warning(
            "The road-network nodes and edges are not individually encoded into these qubits. "
            "The graph is processed classically first. The quantum problem operates on the "
            "reduced candidate-route selection problem."
        )
        st.markdown("**QUBO used by the backend**")
        st.latex(
            r"Q(\mathbf{x})=\sum_{i=0}^{K-1} R_i x_i"
            r"+P\left(\sum_{i=0}^{K-1}x_i-1\right)^2"
        )
        st.latex(
            r"R_i=\hat{t}_i+\lambda_u\frac{U_i-L_i}{2}"
            r"+\lambda_d\max(0,U_i-D)+\lambda_r C_i"
        )
        st.write(
            f"For this run, P = {result.qubo.penalty:.3f}; "
            f"λᵤ = {result.risk_weights.lambda_uncertainty:.1f}, "
            f"λd = {result.risk_weights.lambda_deadline:.1f}, and "
            f"λr = {result.risk_weights.lambda_reroute:.1f}. Static reroute cost Cᵢ is zero."
        )
        coefficient_table = pd.DataFrame(
            {
                "Route": result.route_table["route_id"],
                "Risk coefficient Rᵢ": result.route_table["risk_score"],
                "Linear QUBO coefficient": result.qubo.linear,
            }
        )
        st.dataframe(
            coefficient_table.style.format(
                {
                    "Risk coefficient Rᵢ": "{:.3f}",
                    "Linear QUBO coefficient": "{:.3f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )


def render_ai_explainer(result: PipelineResult) -> None:
    row = result.selected_row
    with st.expander("How did the AI predict this?"):
        st.markdown("**CLASSICAL AI**")
        st.write(
            "Traffic and network features → classical regression model → predicted travel time. "
            "Held-out calibration residuals → locally scaled split conformal calibration → "
            "prediction interval."
        )
        details = pd.DataFrame(
            {
                "Field": [
                    "Model used",
                    "Input features",
                    "Prediction",
                    "Calibration method",
                    "Calibration alpha",
                    "Calibration rows",
                    "Prediction interval",
                    "Artifact scope",
                ],
                "Value": [
                    result.model_name,
                    ", ".join(result.model_features),
                    _minutes(float(row["prediction"])),
                    result.calibration_method,
                    f"{result.calibration_alpha:.2f} ({1-result.calibration_alpha:.0%} nominal)",
                    f"{result.calibration_size:,}",
                    f"{float(row['lower_bound']):.2f}–{float(row['upper_bound']):.2f} min",
                    result.model_artifact_scope,
                ],
            }
        )
        st.dataframe(details, hide_index=True, width="stretch")
        if result.is_real_topology:
            st.warning(
                "This is frozen-model transfer to real road geometry with simulated traffic. "
                "The validated transfer result was unfavorable; this is not live-traffic validation."
            )


def render_dynamic_result(result: DynamicDemoResult) -> None:
    st.markdown('<div class="q-section-label">DISRUPTION RESULT</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    columns[0].metric("Event", result.event_label)
    columns[1].metric("Rerouting occurred", "YES" if result.rerouting_occurred else "NO")
    columns[2].metric("Reroute cost", f"{result.reroute_cost_minutes:.2f} min-eq")
    margin = result.updated_deadline_margin
    columns[3].metric(
        "Updated deadline margin",
        "Unavailable" if margin is None else f"{margin:+.2f} min",
    )
    st.write(f"**Route before event:** {result.route_before_event or 'Unavailable'}")
    st.write(f"**Updated route:** {result.updated_route or 'Unavailable'}")
    if result.disruption_edge is not None:
        st.caption(
            f"Disrupted edge: {result.disruption_edge[0]} → {result.disruption_edge[1]}"
        )
    decisions = result.simulation.decisions.copy()
    if not decisions.empty:
        visible = [
            column
            for column in [
                "decision_index",
                "event",
                "current_node",
                "selected_route_id",
                "rerouted",
                "predicted_remaining_time",
                "elapsed_time",
            ]
            if column in decisions.columns
        ]
        st.dataframe(decisions.loc[:, visible], hide_index=True, width="stretch")
    st.caption(
        "This is a controlled simulation using the existing edge-by-edge rerouting backend, not "
        "an operational or live-traffic result."
    )
