"""Artifact-backed research-results and honesty views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st

from ui.theme import CYAN, EMBER, GRAPHITE, MUTED, TEXT


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ResearchFinding:
    key: str
    label: str
    value: float | None
    sample_size: int | None
    source: str
    note: str


def _read_csv(relative_path: str) -> pd.DataFrame:
    path = REPOSITORY_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _finding(
    *,
    key: str,
    label: str,
    source: str,
    note: str,
    extractor: Callable[[], tuple[float, int | None]],
) -> ResearchFinding:
    try:
        value, sample_size = extractor()
        if not np.isfinite(value):
            raise ValueError("non-finite finding")
        return ResearchFinding(key, label, float(value), sample_size, source, note)
    except (FileNotFoundError, KeyError, ValueError, IndexError, pd.errors.ParserError):
        return ResearchFinding(key, label, None, None, source, "Result unavailable")


def load_research_findings() -> dict[str, ResearchFinding]:
    """Read headline results from validated saved artifacts, never literals."""

    cross_path = "results/research/step9_cross_network/conformal_metrics_overall.csv"
    lono_path = "results/research/step9_1_leave_one_network_out/conformal_metrics_aggregate.csv"
    real_point_path = "results/research/step8_5_real_topology/model_metrics_overall.csv"
    real_coverage_path = "results/research/step8_5_real_topology/conformal_coverage_overall.csv"
    qaoa_path = "results/research/step9_4_robustness_ablation/qaoa_aggregate.csv"
    dynamic_path = "results/research/step8_dynamic_rerouting/dynamic_policy_summary.csv"
    reroute_path = "results/research/step9_4_robustness_ablation/dynamic_ablation_summary.csv"

    def cross(group: str) -> tuple[float, int]:
        table = _read_csv(cross_path)
        selected = table.loc[
            (table["evaluation_group"] == group)
            & (table["method"] == "local")
            & np.isclose(table["configured_alpha"], 0.10)
        ]
        if len(selected) != 1:
            raise ValueError("ambiguous cross-network result")
        row = selected.iloc[0]
        return float(row["empirical_coverage"]), int(row["n_samples"])

    def lono() -> tuple[float, int]:
        table = _read_csv(lono_path)
        selected = table.loc[table["method"] == "local"]
        if len(selected) != 1:
            raise ValueError("ambiguous LONO result")
        row = selected.iloc[0]
        return float(row["empirical_coverage"]), int(row["n_samples"])

    def real_point() -> tuple[float, int]:
        row = _read_csv(real_point_path).iloc[0]
        return float(row["r2"]), int(row["n_samples"])

    def real_coverage() -> tuple[float, int]:
        row = _read_csv(real_coverage_path).iloc[0]
        return float(row["empirical_coverage"]), int(row["n_samples"])

    def qaoa(method: str) -> tuple[float, int]:
        table = _read_csv(qaoa_path)
        selected = table.loc[
            (table["scope_id"] == "single_grid_network_candidate_scaling")
            & (table["K"] == 8)
            & (table["depth"] == 3)
            & (table["method"] == method)
        ]
        if len(selected) != 1:
            raise ValueError("ambiguous QAOA result")
        row = selected.iloc[0]
        return float(row["mean_valid_probability"]), int(row["run_count"])

    def dynamic(policy: str) -> tuple[float, int]:
        table = _read_csv(dynamic_path)
        selected = table.loc[table["policy"] == policy]
        if len(selected) != 1:
            raise ValueError("ambiguous dynamic result")
        row = selected.iloc[0]
        return float(row["deadline_violation_rate"]), int(row["run_count"])

    def reroute(condition: str, column: str) -> tuple[float, int]:
        table = _read_csv(reroute_path)
        selected = table.loc[
            (table["condition"] == condition)
            & (table["aggregation_scope"] == "overall")
        ]
        if len(selected) != 1:
            raise ValueError("ambiguous reroute-penalty result")
        row = selected.iloc[0]
        return float(row[column]), int(row["run_count"])

    specifications = [
        ("cross_id", "Cross-network local coverage · TEST-ID", cross_path, "α = 0.10", lambda: cross("TEST-ID")),
        ("traffic_shift", "Traffic-shift local coverage", cross_path, "OOD-traffic · α = 0.10", lambda: cross("OOD-traffic")),
        ("combined_shift", "Combined-shift local coverage", cross_path, "OOD-combined · α = 0.10", lambda: cross("OOD-combined")),
        ("lono", "LONO pooled local coverage", lono_path, "20 folds pooled · α = 0.10", lono),
        ("real_r2", "Real-topology transfer R²", real_point_path, "Real geometry + controlled simulated traffic", real_point),
        ("real_coverage", "Real-topology transfer coverage", real_coverage_path, "α = 0.10", real_coverage),
        ("xy_k8_p3", "Feasible XY validity · K=8, p=3", qaoa_path, "Mean across saved runs", lambda: qaoa("feasible_xy")),
        ("x_k8_p3", "Standard X validity · K=8, p=3", qaoa_path, "Mean across saved runs", lambda: qaoa("standard_x")),
        ("dynamic_expected", "Expected-time deadline violations", dynamic_path, "Controlled dynamic simulation", lambda: dynamic("expected_time_rerouting")),
        ("dynamic_qaoa", "Risk-aware QAOA deadline violations", dynamic_path, "Controlled dynamic simulation", lambda: dynamic("uncertainty_aware_qaoa")),
        ("reroute_full", "Mean reroutes · with penalty", reroute_path, "Full condition", lambda: reroute("full", "mean_reroutes")),
        ("reroute_without", "Mean reroutes · without penalty", reroute_path, "Paired ablation", lambda: reroute("without_reroute_penalty", "mean_reroutes")),
        ("reroute_violation_full", "Deadline violations · with penalty", reroute_path, "Full condition", lambda: reroute("full", "realized_deadline_violation_rate")),
        ("reroute_violation_without", "Deadline violations · without penalty", reroute_path, "Paired ablation", lambda: reroute("without_reroute_penalty", "realized_deadline_violation_rate")),
    ]
    return {
        key: _finding(
            key=key,
            label=label,
            source=source,
            note=note,
            extractor=extractor,
        )
        for key, label, source, note, extractor in specifications
    }


def _bar_figure(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    value_format: str,
    percentage: bool = False,
    reference: float | None = None,
    colors: list[str] | None = None,
) -> Figure:
    figure, axis = plt.subplots(figsize=(8.3, max(2.7, 0.53 * len(labels) + 1.35)), dpi=125)
    figure.patch.set_facecolor(GRAPHITE)
    axis.set_facecolor(GRAPHITE)
    positions = np.arange(len(labels))
    chosen_colors = colors or [CYAN] * len(labels)
    bars = axis.barh(positions, values, color=chosen_colors, alpha=0.88)
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    maximum = 1.0 if percentage else max(values + [0.01]) * 1.18
    axis.set_xlim(0.0, maximum)
    axis.set_title(title, loc="left", color=TEXT, fontsize=10, weight="bold")
    axis.tick_params(colors=TEXT, labelsize=8)
    axis.grid(axis="x", color="#303842", alpha=0.38, linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if reference is not None:
        axis.axvline(reference, color="#e2e7eb", linewidth=1.0, linestyle=(0, (3, 3)))
        axis.text(reference, -0.65, "Nominal 90%", color=MUTED, fontsize=7, ha="center")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            min(value + maximum * 0.015, maximum * 0.94),
            bar.get_y() + bar.get_height() / 2,
            format(value, value_format),
            color=TEXT,
            va="center",
            fontsize=8,
            weight="bold",
        )
    figure.tight_layout(pad=0.8)
    return figure


def _available(findings: dict[str, ResearchFinding], keys: list[str]) -> bool:
    return all(findings[key].value is not None for key in keys)


def _metric_card(column, finding: ResearchFinding, *, percentage: bool = False) -> None:
    if finding.value is None:
        value = "Result unavailable"
    elif percentage:
        value = f"{finding.value:.1%}"
    else:
        value = f"{finding.value:.3f}"
    column.metric(finding.label, value)
    if finding.sample_size is not None:
        column.caption(f"n = {finding.sample_size:,}")


def _claims_sections() -> tuple[list[str], list[str]]:
    path = REPOSITORY_ROOT / "paper/CLAIMS.md"
    if not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8")

    supported: list[str] = []
    if "## Supported findings" in text:
        section = text.split("## Supported findings", 1)[1].split("## ", 1)[0]
        for line in section.splitlines():
            if not line.startswith("|") or "---" in line or "Claim |" in line:
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and cells[0]:
                supported.append(cells[0])

    unsupported: list[str] = []
    if "## Unsupported claims" in text:
        section = text.split("## Unsupported claims", 1)[1].split("## ", 1)[0]
        for line in section.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                unsupported.append(stripped[2:].rstrip(";."))
    return supported, unsupported


def render_research_view() -> None:
    findings = load_research_findings()
    st.markdown("## Research results")
    st.caption(
        "Values below are read from validated result artifacts at app runtime. Missing sources are "
        "reported as unavailable; no headline number is synthesized in the UI."
    )

    columns = st.columns(4)
    _metric_card(columns[0], findings["cross_id"], percentage=True)
    _metric_card(columns[1], findings["traffic_shift"], percentage=True)
    _metric_card(columns[2], findings["combined_shift"], percentage=True)
    _metric_card(columns[3], findings["lono"], percentage=True)

    coverage_keys = ["cross_id", "traffic_shift", "combined_shift", "lono", "real_coverage"]
    if _available(findings, coverage_keys):
        figure = _bar_figure(
            ["Cross-network ID", "Traffic shift", "Combined shift", "LONO pooled", "Real topology"],
            [float(findings[key].value) for key in coverage_keys],
            title="Empirical conformal coverage across validated evaluation scopes",
            value_format=".1%",
            percentage=True,
            reference=0.90,
            colors=[CYAN, EMBER, EMBER, CYAN, EMBER],
        )
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.caption(
            "Locally scaled conformal intervals, α = 0.10. Evaluation scopes are not interchangeable; "
            "the bars expose distribution-shift and transfer failures."
        )
    else:
        st.info("Coverage plot unavailable because one or more validated artifacts are missing.")

    left, right = st.columns(2)
    with left:
        st.markdown("### Real-topology transfer")
        metric_columns = st.columns(2)
        _metric_card(metric_columns[0], findings["real_r2"])
        _metric_card(metric_columns[1], findings["real_coverage"], percentage=True)
        st.caption("Real road topology with controlled simulated traffic—not observed live traffic.")

    with right:
        st.markdown("### K=8, p=3 feasibility")
        keys = ["xy_k8_p3", "x_k8_p3"]
        if _available(findings, keys):
            figure = _bar_figure(
                ["Feasible XY", "Standard X"],
                [float(findings[key].value) for key in keys],
                title="Mean valid-solution probability",
                value_format=".1%",
                percentage=True,
                colors=[CYAN, "#66717c"],
            )
            st.pyplot(figure, width="stretch")
            plt.close(figure)
        else:
            st.info("QAOA feasibility result unavailable.")

    left, right = st.columns(2)
    with left:
        st.markdown("### Dynamic deadline violations")
        keys = ["dynamic_expected", "dynamic_qaoa"]
        if _available(findings, keys):
            figure = _bar_figure(
                ["Expected-time rerouting", "Risk-aware QAOA"],
                [float(findings[key].value) for key in keys],
                title="Controlled-simulation deadline violation rate",
                value_format=".1%",
                percentage=True,
                colors=[CYAN, EMBER],
            )
            st.pyplot(figure, width="stretch")
            plt.close(figure)
            st.caption("35 paired scenarios per policy. Lower is better; QAOA performed worse here.")
        else:
            st.info("Dynamic policy result unavailable.")

    with right:
        st.markdown("### Reroute-penalty ablation")
        keys = ["reroute_full", "reroute_without"]
        if _available(findings, keys):
            figure = _bar_figure(
                ["With reroute penalty", "Without reroute penalty"],
                [float(findings[key].value) for key in keys],
                title="Mean reroutes per controlled run",
                value_format=".2f",
                colors=[CYAN, EMBER],
            )
            st.pyplot(figure, width="stretch")
            plt.close(figure)
            full = findings["reroute_violation_full"].value
            without = findings["reroute_violation_without"].value
            if full is not None and without is not None:
                st.caption(
                    f"Deadline violations were unchanged: {full:.2%} with the penalty and "
                    f"{without:.2%} without it."
                )
        else:
            st.info("Reroute-penalty result unavailable.")

    supported, unsupported = _claims_sections()
    with st.expander("WHAT Q-RESCUE CURRENTLY PROVES"):
        if supported:
            for claim in supported:
                st.markdown(f"- {claim}")
        else:
            st.write("Claims ledger unavailable.")
        st.caption("Source: paper/CLAIMS.md. Claims remain scoped to the validated experiments.")

    with st.expander("WHAT Q-RESCUE DOES NOT CLAIM"):
        required = [
            "no demonstrated quantum advantage",
            "no quantum speedup claim",
            "no real live-traffic validation",
            "no operational medical deployment claim",
            "no proof that QAOA beats classical routing",
            "no hardware result yet",
        ]
        for item in required:
            st.markdown(f"- {item}")
        if unsupported:
            st.markdown("**Additional wording prohibited by the claims ledger:**")
            for claim in unsupported:
                st.markdown(f"- {claim}")

    with st.expander("Artifact provenance"):
        provenance = pd.DataFrame(
            [
                {
                    "Finding": finding.label,
                    "Source artifact": finding.source,
                    "Status": "Loaded" if finding.value is not None else "Result unavailable",
                    "Sample size": finding.sample_size,
                }
                for finding in findings.values()
            ]
        )
        st.dataframe(provenance, hide_index=True, width="stretch")
