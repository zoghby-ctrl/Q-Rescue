"""Build paper figures only from validated, already-saved Q-Rescue tables."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2F6B9A"
ORANGE = "#D97732"
GOLD = "#C6A13B"
OLIVE = "#778A3D"
PINK = "#B45A7A"
INK = "#263238"
GREY = "#AAB3B8"
LIGHT = "#E8EDF0"
TOPOLOGY_COLORS = {
    "grid": BLUE,
    "radial": GOLD,
    "irregular_geometric": ORANGE,
    "sparse_mixed": OLIVE,
    "dense_mixed": PINK,
}

SOURCE_ARTIFACTS = [
    "results/research/step9_cross_network/conformal_metrics_overall.csv",
    "results/research/step9_1_leave_one_network_out/point_metrics_by_fold.csv",
    "results/research/step9_1_leave_one_network_out/conformal_metrics_by_fold.csv",
    "results/research/step8_5_real_topology/conformal_coverage_by_network.csv",
    "results/research/step7_5_qaoa/qaoa_aggregate.csv",
    "results/research/step8_dynamic_rerouting/dynamic_policy_summary.csv",
    "results/research/step8_dynamic_rerouting/dynamic_runs.csv",
    "results/research/step9_4_robustness_ablation/qubo_penalty_aggregate.csv",
    "results/research/step9_4_robustness_ablation/dynamic_ablation_summary.csv",
    "results/research/step9_3_classical_baselines/baseline_summary.csv",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": LIGHT,
            "grid.linewidth": 0.8,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cross_network_coverage() -> None:
    path = ROOT / "results/research/step9_cross_network/conformal_metrics_overall.csv"
    data = pd.read_csv(path)
    data = data[(data["method"] == "local") & np.isclose(data["configured_alpha"], 0.10)]
    order = ["TEST-ID", "OOD-topology", "OOD-traffic", "OOD-combined"]
    data = data.set_index("evaluation_group").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(data["evaluation_group"], data["empirical_coverage"], color=BLUE, edgecolor=INK)
    ax.axhline(0.90, color=INK, linestyle="--", linewidth=1.4, label="90% nominal")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Empirical coverage")
    fig.suptitle("Local conformal coverage under network and traffic shift", y=0.98, fontsize=14)
    n_per_group = int(data["n_samples"].min())
    fig.text(0.5, 0.91, f"Alpha=0.10; {n_per_group:,} simulated trips per evaluation group", ha="center")
    fig.subplots_adjust(top=0.82)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower left")
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in data["empirical_coverage"]], padding=3)
    save(fig, "fig01_cross_network_coverage.png")


def lono_transfer() -> None:
    metrics = pd.read_csv(ROOT / "results/research/step9_1_leave_one_network_out/point_metrics_by_fold.csv")
    conformal = pd.read_csv(ROOT / "results/research/step9_1_leave_one_network_out/conformal_metrics_by_fold.csv")
    local = conformal[conformal["method"] == "local"]
    data = metrics.merge(local[["fold_id", "empirical_coverage"]], on="fold_id", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for topology, group in data.groupby("heldout_topology"):
        ax.scatter(group["rmse"], group["empirical_coverage"], s=58,
                   color=TOPOLOGY_COLORS[topology], edgecolor=INK, label=topology.replace("_", " "))
    ax.axhline(0.90, color=INK, linestyle="--", linewidth=1.2)
    worst = data.nsmallest(2, "empirical_coverage")
    for row in worst.itertuples():
        ax.annotate(row.heldout_network_id, (row.rmse, row.empirical_coverage),
                    xytext=(6, -14), textcoords="offset points", fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0.65, 1.02)
    ax.set_xlabel("Held-out RMSE (minutes)")
    ax.set_ylabel("Local conformal coverage")
    fig.suptitle("Leave-one-network-out prediction error and coverage", y=0.98, fontsize=14)
    fig.text(0.5, 0.91, "20 folds; 216 held-out trips per network; alpha=0.10", ha="center")
    fig.subplots_adjust(top=0.82)
    ax.grid()
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, "fig02_lono_transfer.png")


def real_topology_transfer() -> None:
    data = pd.read_csv(ROOT / "results/research/step8_5_real_topology/conformal_coverage_by_network.csv")
    name_col = "network_id"
    coverage_col = "empirical_coverage"
    data = data.sort_values(name_col)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(data[name_col].str.replace("_", " "), data[coverage_col], color=ORANGE, edgecolor=INK)
    ax.axhline(0.90, color=INK, linestyle="--", linewidth=1.4, label="90% nominal")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Empirical coverage")
    fig.suptitle("Frozen conformal transfer to cached real-road topologies", y=0.98, fontsize=14)
    pooled_coverage = float(
        np.average(data[coverage_col], weights=data["n_samples"])
    )
    per_topology_counts = sorted(data["n_samples"].astype(int).unique())
    count_text = (
        f"{per_topology_counts[0]} trips per topology"
        if len(per_topology_counts) == 1
        else f"{int(data['n_samples'].sum())} trips across topologies"
    )
    fig.text(
        0.5,
        0.91,
        f"Controlled simulated traffic; {count_text}; pooled coverage {pooled_coverage:.2%}",
        ha="center",
    )
    fig.subplots_adjust(top=0.82, bottom=0.22)
    ax.grid(axis="y")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=12)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in data[coverage_col]], padding=3)
    save(fig, "fig03_real_topology_transfer.png")


def qaoa_method_depth() -> None:
    data = pd.read_csv(ROOT / "results/research/step7_5_qaoa/qaoa_aggregate.csv")
    data = data[(data["study"] == "topology_primary") & (data["K"] == 4)]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharex=True)
    for method, color, label in [("standard_x", GREY, "standard X"), ("feasible_xy", BLUE, "feasible XY")]:
        group = data[data["method"] == method].sort_values("depth")
        axes[0].plot(group["depth"], group["mean_valid_probability"], marker="o", color=color, label=label)
        axes[1].plot(group["depth"], group["mean_optimal_probability"], marker="o", color=color, label=label)
    axes[0].set_title("Valid one-hot probability")
    axes[1].set_title("Exact-route probability")
    for ax in axes:
        ax.set_ylim(0, 1.03)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("QAOA depth p")
        ax.grid()
    axes[0].set_ylabel("Mean analytic probability")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Standard-X and feasible-XY QAOA at K=4", y=1.03, fontsize=14)
    cell_count = int(data["run_count"].min())
    topology_count = int(data["topology_count"].max()) if "topology_count" in data else 5
    fig.text(
        0.5,
        -0.01,
        f"{topology_count} topology instances; {cell_count} runs per method/depth; ideal statevector",
        ha="center",
    )
    save(fig, "fig04_qaoa_method_depth.png")


def qaoa_k_scaling() -> None:
    data = pd.read_csv(ROOT / "results/research/step7_5_qaoa/qaoa_aggregate.csv")
    data = data[data["depth"] == 1].copy()
    rows = []
    for k in [4, 6, 8]:
        study = "topology_primary" if k == 4 else "candidate_scaling"
        rows.append(data[(data["study"] == study) & (data["K"] == k)])
    data = pd.concat(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharex=True)
    for method, color, label in [("standard_x", GREY, "standard X"), ("feasible_xy", BLUE, "feasible XY")]:
        group = data[data["method"] == method].sort_values("K")
        axes[0].plot(group["K"], group["mean_valid_probability"], marker="o", color=color, label=label)
        axes[1].plot(group["K"], group["mean_optimal_probability"], marker="o", color=color, label=label)
    axes[0].set_title("Valid one-hot probability")
    axes[1].set_title("Exact-route probability")
    for ax in axes:
        ax.set_ylim(0, 1.03)
        ax.set_xticks([4, 6, 8])
        ax.set_xlabel("Candidate count K / qubits")
        ax.grid()
    axes[0].set_ylabel("Mean analytic probability at p=1")
    axes[1].legend(frameon=False)
    fig.suptitle("Candidate-count scaling on validated ideal-simulator runs", y=1.03, fontsize=14)
    counts = data.groupby("K")["run_count"].sum().astype(int)
    count_text = "; ".join(f"K={int(k)} n={int(n)}" for k, n in counts.items())
    fig.text(0.5, -0.01, f"Unequal validated replication: {count_text}", ha="center")
    save(fig, "fig05_qaoa_k_scaling.png")


def dynamic_policy() -> None:
    data = pd.read_csv(ROOT / "results/research/step8_dynamic_rerouting/dynamic_policy_summary.csv")
    order = ["expected_time_rerouting", "uncertainty_aware_classical", "uncertainty_aware_qaoa", "never_reroute"]
    label_map = {"expected_time_rerouting": "expected time", "uncertainty_aware_classical": "risk classical",
                 "uncertainty_aware_qaoa": "risk QAOA", "never_reroute": "never reroute"}
    data = data.set_index("policy").loc[order].reset_index()
    labels = [label_map[v] for v in data["policy"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    b1 = axes[0].bar(labels, data["deadline_violation_rate"], color=ORANGE, edgecolor=INK)
    b2 = axes[1].bar(labels, data["mean_reroutes"], color=BLUE, edgecolor=INK)
    axes[0].set_title("Deadline violation rate")
    axes[1].set_title("Mean reroutes")
    axes[0].set_ylim(0, 0.20)
    axes[1].set_ylim(0, max(data["mean_reroutes"]) * 1.25)
    axes[0].bar_label(b1, labels=[f"{v:.1%}" for v in data["deadline_violation_rate"]], padding=3)
    axes[1].bar_label(b2, labels=[f"{v:.2f}" for v in data["mean_reroutes"]], padding=3)
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y")
    fig.suptitle("Dynamic routing policy outcomes", y=1.03, fontsize=14)
    run_count = int(data["run_count"].sum())
    dynamic_runs = pd.read_csv(
        ROOT / "results/research/step8_dynamic_rerouting/dynamic_runs.csv"
    )
    network_count = int(dynamic_runs["network_id"].nunique())
    fig.text(
        0.5,
        -0.02,
        f"{run_count} controlled simulations across {network_count} networks and {len(data)} policies",
        ha="center",
    )
    save(fig, "fig06_dynamic_policy.png")


def robustness_boundaries() -> None:
    penalty = pd.read_csv(ROOT / "results/research/step9_4_robustness_ablation/qubo_penalty_aggregate.csv")
    penalty = penalty[penalty["penalty_definition"] == "multiplier_of_instance_minimum_valid_cost"].copy()
    penalty["multiplier"] = penalty["penalty_setting"].str.extract(r"([0-9.]+)x")[0].astype(float)
    dynamic = pd.read_csv(ROOT / "results/research/step9_4_robustness_ablation/dynamic_ablation_summary.csv")
    dynamic = dynamic[dynamic["aggregation_scope"] == "overall"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))
    boundary = penalty.groupby("multiplier", as_index=False)["mean_ground_state_valid_fraction"].mean()
    axes[0].plot(boundary["multiplier"], boundary["mean_ground_state_valid_fraction"], marker="o", color=BLUE,
                 label="K=4/6/8 (identical)")
    axes[0].axvline(1.0, color=INK, linestyle="--", linewidth=1.2)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("Penalty / minimum valid route cost")
    axes[0].set_ylabel("Mean valid fraction among ground states")
    axes[0].set_title("Exactly-one QUBO penalty boundary")
    axes[0].grid()
    axes[0].legend(frameon=False)
    dynamic = dynamic.set_index("condition").loc[["full", "without_reroute_penalty"]].reset_index()
    bars = axes[1].bar(["full", "without reroute\npenalty"], dynamic["mean_reroutes"], color=[BLUE, ORANGE], edgecolor=INK)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Mean reroutes")
    axes[1].set_title("Reroute-penalty ablation")
    axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in dynamic["mean_reroutes"]], padding=3)
    axes[1].grid(axis="y")
    fig.suptitle("Validated robustness failure boundaries", y=1.03, fontsize=14)
    penalty_instances = int(penalty["instance_count"].sum() / penalty["multiplier"].nunique())
    paired_cases = int(dynamic["run_count"].min())
    fig.text(
        0.5,
        -0.02,
        f"Penalty: {penalty_instances} network/K instances; rerouting: {paired_cases} cases per condition",
        ha="center",
    )
    save(fig, "fig07_robustness_boundaries.png")


def classical_baselines() -> None:
    data = pd.read_csv(ROOT / "results/research/step9_3_classical_baselines/baseline_summary.csv")
    labels = {
        "shortest_distance_candidate": "candidate distance",
        "dijkstra_shortest_distance_path": "graph distance",
        "shortest_expected_time_candidate": "candidate expected",
        "dijkstra_expected_time_path": "graph base-time",
        "exact_risk_argmin": "exact risk",
        "minimum_upper_bound_heuristic": "minimum upper bound",
    }
    data["label"] = data["method"].map(labels)
    data = data.sort_values("exact_risk_optimal_rate")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    b1 = axes[0].barh(data["label"], data["exact_risk_optimal_rate"], color=BLUE, edgecolor=INK)
    b2 = axes[1].barh(data["label"], data["mean_risk_regret"], color=ORANGE, edgecolor=INK)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Exact-risk optimum rate")
    axes[1].set_xlabel("Mean risk regret (minute-equivalent)")
    axes[0].set_title("Agreement with exact candidate optimum")
    axes[1].set_title("Regret relative to exact candidate optimum")
    axes[0].bar_label(b1, labels=[f"{v:.0%}" for v in data["exact_risk_optimal_rate"]], padding=3)
    axes[1].bar_label(b2, labels=[f"{v:.2f}" for v in data["mean_risk_regret"]], padding=3)
    for ax in axes:
        ax.grid(axis="x")
    fig.subplots_adjust(left=0.19, right=0.98, wspace=0.72, top=0.80)
    decision_count = int(data["decision_count"].min())
    fig.suptitle(
        f"Classical baseline quality on {decision_count} tractable candidate decisions",
        y=1.02,
        fontsize=14,
    )
    save(fig, "fig08_classical_baselines.png")


def main() -> None:
    style()
    cross_network_coverage()
    lono_transfer()
    real_topology_transfer()
    qaoa_method_depth()
    qaoa_k_scaling()
    dynamic_policy()
    robustness_boundaries()
    classical_baselines()
    outputs = [OUT / f"fig{number:02d}_{stem}.png" for number, stem in [
        (1, "cross_network_coverage"),
        (2, "lono_transfer"),
        (3, "real_topology_transfer"),
        (4, "qaoa_method_depth"),
        (5, "qaoa_k_scaling"),
        (6, "dynamic_policy"),
        (7, "robustness_boundaries"),
        (8, "classical_baselines"),
    ]]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": _sha256(Path(__file__)),
        "source_sha256": {
            name: _sha256(ROOT / name) for name in SOURCE_ARTIFACTS
        },
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
        "manual_numeric_inputs": False,
    }
    (OUT / "step9_5_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote 8 figures to {OUT}")


if __name__ == "__main__":
    main()
