"""Q-Rescue presentation-ready Streamlit MVP.

Run from the repository root with::

    streamlit run app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from src.candidate_routes import generate_candidate_routes
from ui.backend import (
    METHOD_LABELS,
    SCENARIO_LABELS,
    DynamicDemoResult,
    ModelBundle,
    NetworkDefinition,
    PipelineConfigurationError,
    PipelineResult,
    available_k_values,
    load_model_bundles,
    load_network,
    load_network_catalog,
    run_dynamic_demo,
    run_pipeline,
)
from ui.components import (
    render_comparison_table,
    render_empty_state,
    render_header,
    render_pipeline,
    render_route_table,
)
from ui.network_view import build_network_figure
from ui.research_view import render_research_view
from ui.result_cards import (
    render_ai_explainer,
    render_decision_panel,
    render_dynamic_result,
    render_encoding_explainer,
    render_quantum_panel,
)
from ui.theme import apply_theme


st.set_page_config(
    page_title="Q-Rescue · Research MVP",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()


@st.cache_resource(show_spinner=False)
def _catalog() -> dict[str, NetworkDefinition]:
    return load_network_catalog()


@st.cache_resource(show_spinner=False)
def _models() -> dict[str, ModelBundle]:
    return load_model_bundles()


@st.cache_resource(show_spinner=False)
def _graph(network_key: str) -> nx.DiGraph:
    return load_network(network_key)


@st.cache_data(show_spinner=False)
def _execute_pipeline(
    network_key: str,
    origin: str,
    destination: str,
    scenario_name: str,
    deadline_minutes: float,
    k: int,
    selected_method: str,
    qaoa_depth: int,
) -> PipelineResult:
    graph = _graph(network_key)
    bundle = _models()["real" if network_key.startswith("real:") else "synthetic"]
    return run_pipeline(
        graph,
        bundle,
        origin=origin,
        destination=destination,
        scenario_name=scenario_name,
        deadline_minutes=deadline_minutes,
        k=k,
        selected_method=selected_method,
        qaoa_depth=qaoa_depth,
    )


def _clear_results() -> None:
    st.session_state["pipeline_result"] = None
    st.session_state["pipeline_error"] = None
    st.session_state["dynamic_result"] = None


def _load_demo() -> None:
    st.session_state.update(
        {
            "network_key": "synthetic:grid",
            "custom_od": False,
            "scenario_name": "rain",
            "deadline_minutes": 8.0,
            "candidate_k": 4,
            "routing_method": "feasible_xy",
            "qaoa_depth": 1,
            "pipeline_result": None,
            "pipeline_error": None,
            "dynamic_result": None,
        }
    )


def _initialize_state() -> None:
    defaults = {
        "network_key": "synthetic:grid",
        "custom_od": False,
        "scenario_name": "rain",
        "deadline_minutes": 8.0,
        "candidate_k": 4,
        "routing_method": "feasible_xy",
        "qaoa_depth": 1,
        "pipeline_result": None,
        "pipeline_error": None,
        "dynamic_result": None,
        "dynamic_event": "accident",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _network_label(key: str) -> str:
    definition = _catalog()[key]
    prefix = "Real topology case study" if definition.is_real else "Synthetic"
    return f"{prefix} · {definition.label}"


_initialize_state()
render_header()
console_tab, research_tab = st.tabs(["DECISION CONSOLE", "RESEARCH RESULTS"])


with console_tab:
    current_method = str(st.session_state["routing_method"])
    current_result = st.session_state.get("pipeline_result")
    has_result = isinstance(current_result, PipelineResult)
    has_deadline_risk = bool(
        has_result
        and float(current_result.selected_row["upper_bound"])
        > float(current_result.deadline_minutes)
    )
    render_pipeline(
        current_method,
        has_result=has_result,
        has_deadline_risk=has_deadline_risk,
    )

    left, center, right = st.columns([0.92, 1.72, 1.04], gap="medium")

    with left:
        st.markdown('<div class="q-section-label">SCENARIO CONTROLS</div>', unsafe_allow_html=True)
        st.button(
            "LOAD DEMO SCENARIO",
            on_click=_load_demo,
            width="stretch",
            help="Loads the deterministic grid / rain / K=4 / feasible-XY presentation case.",
        )

        network_keys = list(_catalog())
        selected_network_key = st.selectbox(
            "Network",
            network_keys,
            key="network_key",
            format_func=_network_label,
            on_change=_clear_results,
        )
        definition = _catalog()[selected_network_key]
        graph = _graph(selected_network_key)

        if definition.is_real:
            st.warning("Real road topology + simulated traffic. No live traffic is used.")
            st.caption(
                f"{definition.region_label} · cached OpenStreetMap geometry · frozen-model transfer"
            )

        st.checkbox(
            "Choose a custom origin / destination",
            key="custom_od",
            on_change=_clear_results,
        )
        nodes = sorted(graph.nodes, key=lambda node: (str(type(node)), str(node)))
        default_origin = str(graph.graph["origin"])
        default_destination = str(graph.graph["destination"])
        if st.session_state["custom_od"]:
            if st.session_state.get("origin_node") not in nodes:
                st.session_state["origin_node"] = default_origin
            if st.session_state.get("destination_node") not in nodes:
                st.session_state["destination_node"] = default_destination
            origin = st.selectbox(
                "Origin",
                nodes,
                key="origin_node",
                format_func=str,
                on_change=_clear_results,
            )
            destination = st.selectbox(
                "Destination",
                nodes,
                key="destination_node",
                format_func=str,
                on_change=_clear_results,
            )
        else:
            origin = default_origin
            destination = default_destination
            st.caption(f"Benchmark O/D · {origin} → {destination}")

        if origin == destination:
            k_options: tuple[int, ...] = ()
            st.error("Origin and destination must be different.")
        else:
            try:
                k_options = available_k_values(graph, origin, destination)
            except (ValueError, nx.NetworkXNoPath):
                k_options = ()
            if not k_options:
                st.error("This O/D pair does not support the validated minimum K=4.")

        if k_options and st.session_state.get("candidate_k") not in k_options:
            st.session_state["candidate_k"] = k_options[0]

        with st.form("qrescue_controls", border=True):
            scenario_name = st.selectbox(
                "Traffic scenario",
                list(SCENARIO_LABELS),
                key="scenario_name",
                format_func=lambda name: SCENARIO_LABELS[name],
            )
            deadline_minutes = st.slider(
                "Emergency deadline (minutes)",
                min_value=2.0,
                max_value=60.0,
                step=0.5,
                key="deadline_minutes",
            )
            candidate_k = st.selectbox(
                "Candidate routes K",
                list(k_options) if k_options else [4],
                key="candidate_k",
                disabled=not bool(k_options),
                format_func=lambda value: f"{value} · ADVANCED / HEAVIER" if value == 8 else str(value),
                help=(
                    "Only K values supported by the current graph and O/D pair are enabled. "
                    "K=8 is an advanced, heavier ideal-simulator configuration."
                ),
            )
            selected_method = st.selectbox(
                "Routing method",
                list(METHOD_LABELS),
                key="routing_method",
                format_func=lambda method: METHOD_LABELS[method],
            )
            qaoa_depth = st.selectbox(
                "QAOA depth",
                [1, 2, 3],
                key="qaoa_depth",
                format_func=lambda depth: f"p = {depth} · ADVANCED / HEAVIER" if depth == 3 else f"p = {depth}",
                help=(
                    "Used by quantum methods and by the same-scenario comparison panel. "
                    "p=3 is an advanced, heavier simulation setting."
                ),
            )
            run_clicked = st.form_submit_button(
                "RUN Q-RESCUE",
                type="primary",
                width="stretch",
                disabled=not bool(k_options),
            )

        if run_clicked:
            st.session_state["pipeline_error"] = None
            st.session_state["dynamic_result"] = None
            try:
                with st.spinner(
                    "Running candidate generation, AI prediction, conformal calibration, "
                    "risk scoring, exact solve, and both QAOA mixers…"
                ):
                    st.session_state["pipeline_result"] = _execute_pipeline(
                        selected_network_key,
                        str(origin),
                        str(destination),
                        scenario_name,
                        float(deadline_minutes),
                        int(candidate_k),
                        selected_method,
                        int(qaoa_depth),
                    )
                st.rerun()
            except (PipelineConfigurationError, ValueError, nx.NetworkXNoPath) as error:
                st.session_state["pipeline_result"] = None
                st.session_state["pipeline_error"] = str(error)

        if st.session_state.get("pipeline_error"):
            st.error(str(st.session_state["pipeline_error"]))

    result = st.session_state.get("pipeline_result")

    with center:
        st.markdown('<div class="q-section-label">NETWORK VIEW</div>', unsafe_allow_html=True)
        if isinstance(result, PipelineResult):
            figure = build_network_figure(
                result.graph,
                result.routes,
                selected_route_id=result.selected_route_id,
                origin=result.origin,
                destination=result.destination,
                disrupted_edge=result.disruption_edge,
            )
        else:
            try:
                preview_routes = generate_candidate_routes(
                    graph,
                    origin,
                    destination,
                    int(st.session_state.get("candidate_k", 4)),
                    require_k=False,
                )
            except (ValueError, nx.NetworkXNoPath):
                preview_routes = []
            figure = build_network_figure(
                graph,
                preview_routes,
                origin=origin,
                destination=destination,
            )
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        if definition.is_real:
            st.caption("Map data © OpenStreetMap contributors · ODbL 1.0 · cached topology")
        if not isinstance(result, PipelineResult):
            st.caption("Candidate paths are previewed in muted gray. Run Q-Rescue to select a route.")

    with right:
        st.markdown('<div class="q-section-label">DECISION & METRICS</div>', unsafe_allow_html=True)
        if isinstance(result, PipelineResult):
            render_decision_panel(result)
            render_quantum_panel(result)
            st.caption(
                f"Completed in {result.runtime_seconds:.2f} s · cached by the complete scenario configuration"
            )
        else:
            render_empty_state(
                "Configure a supported scenario and run Q-Rescue. The recommendation, deadline "
                "status, uncertainty, and quantum diagnostics will appear here."
            )

    if isinstance(result, PipelineResult):
        st.divider()
        render_route_table(result)
        st.divider()
        render_comparison_table(result)
        explanation_left, explanation_right = st.columns(2, gap="medium")
        with explanation_left:
            render_encoding_explainer(result)
        with explanation_right:
            render_ai_explainer(result)

        st.divider()
        with st.expander("SIMULATE DISRUPTION", expanded=False):
            st.caption(
                "Optional controlled demonstration using the existing edge-by-edge dynamic rerouting backend."
            )
            dynamic_event = st.selectbox(
                "Disruption",
                ["accident", "congestion", "closure", "rain"],
                key="dynamic_event",
                format_func=lambda value: {
                    "accident": "Accident",
                    "congestion": "Congestion",
                    "closure": "Road closure",
                    "rain": "Rain",
                }[value],
            )
            if st.button("SIMULATE DISRUPTION", type="primary"):
                bundle = _models()["real" if result.is_real_topology else "synthetic"]
                try:
                    with st.spinner("Running controlled edge-by-edge rerouting simulation…"):
                        st.session_state["dynamic_result"] = run_dynamic_demo(
                            result,
                            bundle,
                            event_name=dynamic_event,
                        )
                except (PipelineConfigurationError, ValueError, nx.NetworkXNoPath) as error:
                    st.error(str(error))
            dynamic_result = st.session_state.get("dynamic_result")
            if isinstance(dynamic_result, DynamicDemoResult):
                render_dynamic_result(dynamic_result)


with research_tab:
    render_research_view()
