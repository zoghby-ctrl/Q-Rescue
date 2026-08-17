"""Reusable Streamlit presentation components."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from ui.backend import METHOD_LABELS, PipelineResult


def render_header() -> None:
    st.markdown(
        """
        <header class="q-header">
          <div>
            <h1 class="q-title">Q-RESCUE</h1>
            <div class="q-subtitle">Uncertainty-Aware Hybrid Quantum Optimization for Emergency Routing</div>
          </div>
          <div class="q-status"><span class="q-status-dot"></span>Validated Research MVP</div>
        </header>
        <div class="q-identity-strip" aria-label="System composition">
          <span class="q-identity-token">CLASSICAL AI</span>
          <span>+</span>
          <span class="q-identity-token">UNCERTAINTY</span>
          <span>+</span>
          <span class="q-identity-token">QUANTUM OPTIMIZATION</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline(
    method: str,
    *,
    has_result: bool,
    has_deadline_risk: bool = False,
) -> None:
    quantum = method in {"standard_x", "feasible_xy"}
    optimization_label = "QAOA" if quantum else "CLASSICAL SOLVER"
    uses_risk = method != "expected_time"
    stages = [
        ("ROAD NETWORK", "active classical"),
        ("AI TRAVEL-TIME PREDICTION", "active classical"),
        ("CONFORMAL UNCERTAINTY", "active classical" if uses_risk else ""),
        (
            "DEADLINE RISK",
            "active risk" if uses_risk and has_deadline_risk else "active classical" if uses_risk else "",
        ),
        ("QUBO", "active classical" if uses_risk else ""),
        (optimization_label, "active quantum" if quantum else "active classical"),
        ("ROUTE DECISION", "active classical" if has_result else ""),
    ]
    stage_html = "".join(
        f'<div class="q-stage {css_class}">{html.escape(label)}</div>'
        for label, css_class in stages
    )
    explanation = (
        "The road graph and candidate-route generation are classical. Quantum encoding begins "
        "after K candidate routes have been reduced to binary route-selection variables."
    )
    st.markdown(
        f'<div class="q-section-label">WORKING PIPELINE · {html.escape(METHOD_LABELS[method])} '
        f'<span class="q-help" role="img" aria-label="Pipeline encoding explanation" '
        f'title="{html.escape(explanation)}">?</span></div>'
        f'<div class="q-pipeline" role="list" aria-label="Q-Rescue processing pipeline">{stage_html}</div>',
        unsafe_allow_html=True,
    )
    st.caption(explanation)


def render_route_table(result: PipelineResult) -> None:
    st.markdown('<div class="q-section-label">CANDIDATE ROUTES</div>', unsafe_allow_html=True)
    table = result.route_table.copy()
    display = pd.DataFrame(
        {
            "Route ID": table["route_id"],
            "Distance (km)": table["distance_km"],
            "Predicted (min)": table["prediction"],
            "Lower (min)": table["lower_bound"],
            "Upper (min)": table["upper_bound"],
            "Width (min)": table["uncertainty_width"],
            "Deadline margin (min)": table["deadline_margin"],
            "Risk score": table["risk_score"],
            "Selected": table["selected"].map({True: "YES", False: "—"}),
            "Status": table["status"],
        }
    )

    def style_row(row: pd.Series) -> list[str]:
        if row["Selected"] == "YES":
            return [
                "background-color: #102b34; color: #f2f5f7; border-top: 1px solid #38d8ff;"
            ] * len(row)
        if "Lowest risk" in str(row["Status"]):
            return [
                "background-color: #172329; color: #f2f5f7; border-top: 1px solid #397386;"
            ] * len(row)
        if "Fastest expected" in str(row["Status"]):
            return [
                "background-color: #191d22; color: #f2f5f7; border-top: 1px solid #52606d;"
            ] * len(row)
        return ["color: #d8dee5;"] * len(row)

    styled = (
        display.style.apply(style_row, axis=1)
        .format(
            {
                "Distance (km)": "{:.2f}",
                "Predicted (min)": "{:.2f}",
                "Lower (min)": "{:.2f}",
                "Upper (min)": "{:.2f}",
                "Width (min)": "{:.2f}",
                "Deadline margin (min)": "{:+.2f}",
                "Risk score": "{:.2f}",
            }
        )
    )
    st.dataframe(styled, hide_index=True, width="stretch", height=38 * len(display) + 42)
    st.caption(
        "Rows retain disagreements: fastest expected, lowest risk, and the selected method's "
        "recommendation are labeled independently."
    )


def render_comparison_table(result: PipelineResult) -> None:
    st.markdown('<div class="q-section-label">CLASSICAL VS QUANTUM · SAME SCENARIO</div>', unsafe_allow_html=True)
    table = result.comparison_table.copy()
    display = pd.DataFrame(
        {
            "Method": table["method"],
            "Selected route": table["route_id"],
            "Travel time (min)": table["predicted_time"],
            "Deadline margin (min)": table["deadline_margin"],
            "Risk score": table["risk_score"],
            "Valid probability": table["valid_probability"],
            "Optimal probability": table["optimal_probability"],
            "Active method": table["is_selected_method"].map({True: "YES", False: "—"}),
        }
    )

    def probability(value: object) -> str:
        if pd.isna(value):
            return "N/A · classical"
        return f"{float(value):.1%}"

    display["Valid probability"] = display["Valid probability"].map(probability)
    display["Optimal probability"] = display["Optimal probability"].map(probability)
    st.dataframe(
        display.style.format(
            {
                "Travel time (min)": "{:.2f}",
                "Deadline margin (min)": "{:+.2f}",
                "Risk score": "{:.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Valid-solution probability and optimal-route probability are distinct. Classical rows "
        "are deterministic selections, so quantum probability fields are not applicable."
    )


def render_empty_state(message: str) -> None:
    st.markdown(
        f'<div class="q-empty">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )
