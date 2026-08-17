"""Build publication figures exclusively from frozen machine-readable artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
RED = "#B6403A"
INK = "#263238"
GREY = "#8C989E"
LIGHT = "#E8EDF0"
METHOD_COLORS = {"standard_x": GREY, "feasible_xy": BLUE}
TOPOLOGY_COLORS = {
    "grid": BLUE,
    "radial": GOLD,
    "irregular_geometric": ORANGE,
    "sparse_mixed": OLIVE,
    "dense_mixed": PINK,
}

SOURCE_ARTIFACTS = [
    "results/research/step9_6_statistics/descriptive_primary.csv",
    "results/research/step9_2_topology_uncertainty_qaoa/qaoa_runs.csv",
    "results/research/step9_7_adaptive_conformal/coverage_overall.csv",
    "results/research/step9_8_noise/noise_runs.csv",
    "results/research/targeted_k16_scaling/qaoa_runs.csv",
    "results/research/standard_x_penalty_sensitivity/penalty_runs.csv",
    "results/research/standard_x_penalty_sensitivity/test_summary.csv",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
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
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bootstrap_mean(values: np.ndarray, *, seed: int, resamples: int = 10_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def feasibility_optimality_depth() -> None:
    data = pd.read_csv(
        ROOT / "results/research/step9_6_statistics/descriptive_primary.csv"
    )
    metric_specs = [
        ("feasibility_probability", r"$P_{feasible}$", (0.0, 1.04)),
        ("optimal_state_probability", r"$P_{opt}$", (0.0, 0.75)),
        (
            "optimal_probability_given_feasible",
            r"$P_{opt\mid feasible}$",
            (0.0, 0.75),
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), sharex=True)
    for axis, (metric, title, limits) in zip(axes, metric_specs, strict=True):
        metric_data = data[data["metric"] == metric]
        for method in ["standard_x", "feasible_xy"]:
            for k_value, linestyle in [(4, "-"), (8, "--")]:
                group = metric_data[
                    (metric_data["method"] == method) & (metric_data["K"] == k_value)
                ].sort_values("depth")
                mean = group["cluster_equal_weight_mean"].to_numpy(float)
                lower = group["cluster_bootstrap_ci95_lower"].to_numpy(float)
                upper = group["cluster_bootstrap_ci95_upper"].to_numpy(float)
                label = f"{method.replace('_', ' ')}, K={k_value}"
                axis.plot(
                    group["depth"],
                    mean,
                    marker="o",
                    linestyle=linestyle,
                    linewidth=1.8,
                    color=METHOD_COLORS[method],
                    label=label,
                )
                axis.fill_between(
                    group["depth"], lower, upper, color=METHOD_COLORS[method], alpha=0.10
                )
        axis.set_title(title)
        axis.set_ylim(*limits)
        axis.set_xticks(sorted(metric_data["depth"].unique()))
        axis.set_xlabel("QAOA depth p")
        axis.grid()
    axes[0].set_ylabel("Network-equal-weight mean probability")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Feasibility does not imply optimal-state concentration", fontsize=14, y=1.02)
    fig.text(
        0.5,
        -0.10,
        f"{int(data['independent_cluster_count'].max())} independent networks; "
        "shaded bands are 95% network-cluster bootstrap intervals",
        ha="center",
    )
    fig.subplots_adjust(bottom=0.23, wspace=0.24)
    save(fig, "fig09_feasibility_optimality_depth")


def entropy_uncertainty_topology() -> None:
    runs = pd.read_csv(
        ROOT / "results/research/step9_2_topology_uncertainty_qaoa/qaoa_runs.csv"
    )
    runs = runs[
        (runs["method"] == "feasible_xy") & (runs["K"] == 8) & (runs["depth"] == 3)
    ]
    order = ["low", "medium", "high"]
    positions = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    seed = 9100
    for topology in TOPOLOGY_COLORS:
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for uncertainty in order:
            values = runs[
                (runs["topology"] == topology) & (runs["uncertainty"] == uncertainty)
            ]["normalized_feasible_entropy"].to_numpy(float)
            mean, lower, upper = _bootstrap_mean(values, seed=seed)
            seed += 1
            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)
        ax.errorbar(
            positions,
            means,
            yerr=[np.asarray(means) - lowers, np.asarray(uppers) - means],
            marker="o",
            capsize=3,
            linewidth=1.7,
            color=TOPOLOGY_COLORS[topology],
            label=topology.replace("_", " "),
        )
    ax.set_xticks(positions, order)
    ax.set_ylim(0.65, 0.95)
    ax.set_xlabel("Controlled predictive-uncertainty regime")
    ax.set_ylabel("Normalized feasible-state entropy")
    ax.grid()
    ax.legend(ncol=2, loc="lower left")
    fig.suptitle("Entropy varies more by topology than by uncertainty regime", fontsize=14, y=0.99)
    per_cell = int(runs.groupby(["topology", "uncertainty"]).size().min())
    fig.text(0.5, 0.91, f"K=8, p=3; {per_cell} matched network seeds per topology/regime", ha="center")
    fig.subplots_adjust(top=0.82)
    save(fig, "fig10_entropy_uncertainty_topology")


def entropy_risk_regret() -> None:
    runs = pd.read_csv(
        ROOT / "results/research/step9_2_topology_uncertainty_qaoa/qaoa_runs.csv"
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for method in ["standard_x", "feasible_xy"]:
        group = runs[runs["method"] == method]
        ax.scatter(
            group["normalized_feasible_entropy"],
            group["risk_regret"],
            s=16,
            alpha=0.20,
            color=METHOD_COLORS[method],
            edgecolors="none",
            label=method.replace("_", " "),
        )
        binned = group.assign(
            entropy_bin=pd.qcut(
                group["normalized_feasible_entropy"], q=8, duplicates="drop"
            )
        ).groupby("entropy_bin", observed=True).agg(
            entropy=("normalized_feasible_entropy", "mean"),
            regret=("risk_regret", "mean"),
        )
        ax.plot(
            binned["entropy"],
            binned["regret"],
            color=METHOD_COLORS[method],
            linewidth=2.2,
        )
    ax.set_xlabel("Normalized feasible-state entropy")
    ax.set_ylabel("Selected-route encoded risk regret")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=-0.2)
    ax.grid()
    ax.legend()
    fig.suptitle("Diffuse feasible distributions co-occur with encoded risk regret", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.92,
        f"Points are {len(runs):,} controlled trials; solid curves are within-mixer octile means",
        ha="center",
    )
    fig.subplots_adjust(top=0.84)
    save(fig, "fig11_entropy_risk_regret")


def operational_violation_popt() -> None:
    runs = pd.read_csv(
        ROOT / "results/research/step9_2_topology_uncertainty_qaoa/qaoa_runs.csv"
    )
    fig, ax = plt.subplots(figsize=(8.9, 5.4))
    rng = np.random.default_rng(9130)
    for method in ["standard_x", "feasible_xy"]:
        group = runs[runs["method"] == method].copy()
        y = group["selected_realized_deadline_violation"].astype(float).to_numpy()
        jitter = rng.uniform(-0.035, 0.035, size=len(group))
        ax.scatter(
            group["optimal_state_probability"],
            y + jitter,
            s=11,
            alpha=0.13,
            color=METHOD_COLORS[method],
            edgecolors="none",
        )
        group["probability_bin"] = pd.qcut(
            group["optimal_state_probability"], q=7, duplicates="drop"
        )
        binned = group.groupby("probability_bin", observed=True).agg(
            probability=("optimal_state_probability", "mean"),
            violation=("selected_realized_deadline_violation", "mean"),
            n=("run_id", "size"),
        )
        ax.plot(
            binned["probability"],
            binned["violation"],
            marker="o",
            linewidth=2.2,
            color=METHOD_COLORS[method],
            label=f"{method.replace('_', ' ')} bin mean",
        )
    ax.set_xlabel(r"Optimal-state probability $P_{opt}$")
    ax.set_ylabel("Realized one-decision deadline violation")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 1], ["met", "violated"])
    ax.grid()
    ax.legend(loc="upper right")
    fig.suptitle(r"Higher $P_{opt}$ co-occurs with fewer violations but does not determine utility", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.92,
        r"Feasible-XY has $P_{feasible}=1$ in every trial; curves show within-mixer septile means",
        ha="center",
    )
    fig.subplots_adjust(top=0.84)
    save(fig, "fig12_operational_violation_popt")


def adaptive_conformal() -> None:
    data = pd.read_csv(
        ROOT / "results/research/step9_7_adaptive_conformal/coverage_overall.csv"
    )
    groups = ["TEST-ID", "OOD-topology", "OOD-traffic", "OOD-combined"]
    methods = ["absolute_split", "local_scaled_split", "adaptive_local_aci"]
    labels = ["absolute split", "local split", "adaptive local ACI"]
    colors = [GREY, ORANGE, BLUE]
    x = np.arange(len(groups))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.9), sharex=True)
    for index, (method, label, color) in enumerate(zip(methods, labels, colors, strict=True)):
        subset = data[data["method"] == method].set_index("evaluation_group").loc[groups]
        position = x + (index - 1) * width
        axes[0].bar(
            position,
            subset["empirical_coverage"],
            width=width,
            color=color,
            edgecolor=INK,
            linewidth=0.6,
            label=label,
        )
        axes[1].bar(
            position,
            subset["average_interval_width"],
            width=width,
            color=color,
            edgecolor=INK,
            linewidth=0.6,
        )
    axes[0].axhline(0.90, color=INK, linestyle="--", linewidth=1.2, label="90% target")
    axes[0].set_ylim(0.4, 1.01)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Coverage")
    axes[1].set_ylabel("Mean interval width (minutes)")
    axes[1].set_title("Width cost")
    for axis in axes:
        axis.set_xticks(x, [value.replace("OOD-", "OOD\n") for value in groups])
        axis.grid(axis="y")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Adaptive calibration partially recovers shift coverage at a width cost", fontsize=14, y=1.02)
    observations = int(data["n_samples"].min())
    fig.text(0.5, -0.10, f"{observations:,} sequential observations per evaluation group and method", ha="center")
    fig.subplots_adjust(bottom=0.25, wspace=0.25)
    save(fig, "fig13_adaptive_conformal")


def noise_robustness() -> None:
    runs = pd.read_csv(ROOT / "results/research/step9_8_noise/noise_runs.csv")
    conditions = ["ideal_analytic", "finite_shot_noiseless", "noisy_shot"]
    condition_labels = ["ideal", "4,096-shot", "noisy 4,096-shot"]
    metrics = [
        ("feasibility_probability", r"$P_{feasible}$"),
        ("optimal_state_probability", r"$P_{opt}$"),
        ("normalized_feasible_entropy", "Normalized entropy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharex=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for method in ["standard_x", "feasible_xy"]:
            for depth, linestyle in [(1, "-"), (3, "--")]:
                group = (
                    runs[(runs["method"] == method) & (runs["depth"] == depth)]
                    .groupby("condition")[metric]
                    .mean()
                    .reindex(conditions)
                )
                axis.plot(
                    np.arange(len(conditions)),
                    group,
                    marker="o",
                    linestyle=linestyle,
                    linewidth=1.8,
                    color=METHOD_COLORS[method],
                    label=f"{method.replace('_', ' ')}, p={depth}",
                )
        axis.set_title(title)
        axis.set_xticks(np.arange(len(conditions)), condition_labels, rotation=12)
        axis.set_ylim(-0.02, 1.03)
        axis.grid()
    axes[0].set_ylabel("Mean probability / normalized entropy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Local noise erodes the ideal feasible-XY advantage", fontsize=14, y=1.02)
    fig.text(
        0.5,
        -0.12,
        f"K=4; {runs['problem_id'].nunique()} matched problems; declared phenomenological channel model, not hardware",
        ha="center",
    )
    fig.subplots_adjust(bottom=0.28, wspace=0.24)
    save(fig, "fig14_noise_robustness")


def conceptual_pipeline() -> None:
    labels = [
        "Road graph\n+ traffic state",
        "K candidate\nroutes",
        "Prediction +\nuncertainty",
        "Risk vector +\nK-variable QUBO",
        "QAOA\ndistribution",
        "Decoded route +\noperational outcome",
    ]
    colors = [LIGHT, "#DCE9F2", "#F6E8D8", "#E8E2F2", "#DDEBE7", "#F1DFDF"]
    fig, ax = plt.subplots(figsize=(14.0, 3.2))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    x_positions = np.linspace(0.3, 12.8, len(labels))
    for index, (x_value, label, color) in enumerate(
        zip(x_positions, labels, colors, strict=True)
    ):
        box = FancyBboxPatch(
            (x_value, 1.05),
            2.0,
            1.15,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.1,
            edgecolor=INK,
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x_value + 1.0, 1.63, label, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1], 1.63),
                xytext=(x_value + 2.0, 1.63),
                arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.4},
            )
    ax.text(
        7.6,
        0.44,
        r"Keep separate: validity $P_{feasible}$   |   concentration $P_{opt}$ / entropy   |   downstream utility",
        ha="center",
        va="center",
        fontsize=11,
        color=RED,
    )
    fig.suptitle("Q-Rescue diagnostic pipeline", fontsize=14, y=0.96)
    save(fig, "fig15_conceptual_pipeline")


def targeted_k_scaling() -> None:
    primary = pd.read_csv(
        ROOT / "results/research/step9_2_topology_uncertainty_qaoa/qaoa_runs.csv"
    )
    k16 = pd.read_csv(ROOT / "results/research/targeted_k16_scaling/qaoa_runs.csv")
    topologies = sorted(k16["topology"].unique())
    seed_ids = sorted(k16["matched_seed_id"].unique())
    uncertainty = sorted(k16["uncertainty"].unique())
    primary = primary[
        (primary["topology"].isin(topologies))
        & (primary["matched_seed_id"].isin(seed_ids))
        & (primary["uncertainty"].isin(uncertainty))
        & (primary["depth"].isin([1, 3]))
    ]
    data = pd.concat([primary, k16], ignore_index=True)
    metrics = [
        ("feasibility_probability", r"$P_{feasible}$"),
        ("optimal_state_probability", r"$P_{opt}$"),
        ("normalized_feasible_entropy", "Normalized entropy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for method in ["standard_x", "feasible_xy"]:
            for depth, linestyle in [(1, "-"), (3, "--")]:
                group = (
                    data[(data["method"] == method) & (data["depth"] == depth)]
                    .groupby("K")[metric]
                    .mean()
                    .sort_index()
                )
                axis.plot(
                    group.index,
                    group.values,
                    marker="o",
                    linestyle=linestyle,
                    color=METHOD_COLORS[method],
                    label=f"{method.replace('_', ' ')}, p={depth}",
                )
        axis.set_title(title)
        axis.set_xticks([4, 8, 16])
        axis.set_xlabel("Candidate count K / qubits")
        axis.set_ylim(-0.02, 1.03)
        axis.grid()
    axes[0].set_ylabel("Matched-subset mean")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Indicative matched-prefix K scaling", fontsize=14, y=1.02)
    fig.text(
        0.5,
        -0.13,
        f"{k16['topology'].nunique()} topologies × {k16['matched_seed_id'].nunique()} seeds × "
        f"{k16['uncertainty'].nunique()} uncertainty regimes; K=16 remains classically trivial",
        ha="center",
    )
    fig.subplots_adjust(bottom=0.28, wspace=0.24)
    save(fig, "fig16_targeted_k_scaling")


def penalty_sensitivity() -> None:
    runs = pd.read_csv(
        ROOT / "results/research/standard_x_penalty_sensitivity/penalty_runs.csv"
    )
    test = runs[runs["data_split"] == "test"]
    order = (
        test.groupby("penalty_condition")["penalty_ratio_to_exact_threshold"]
        .median()
        .sort_values()
    )
    summary = pd.read_csv(
        ROOT / "results/research/standard_x_penalty_sensitivity/test_summary.csv"
    )
    metrics = [
        ("feasibility_probability", r"$P_{feasible}$"),
        ("optimal_state_probability", r"$P_{opt}$"),
        ("normalized_feasible_entropy", "Normalized entropy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.35), sharex=True)
    x = order.to_numpy(float)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = summary[summary["metric"] == metric].set_index("penalty_condition").loc[order.index]
        mean = values["mean"].to_numpy(float)
        lower = values["ci95_lower"].to_numpy(float)
        upper = values["ci95_upper"].to_numpy(float)
        axis.errorbar(
            x,
            mean,
            yerr=[mean - lower, upper - mean],
            marker="o",
            capsize=3,
            color=BLUE,
            linewidth=1.8,
        )
        axis.axvline(1.0, color=RED, linestyle="--", linewidth=1.1)
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel(r"Penalty / exact threshold $P / \min(r)$")
        axis.set_ylim(-0.02, 1.03)
        axis.grid()
    axes[0].set_ylabel("Held-out test mean")
    fig.suptitle("Standard-X behavior depends on penalty scale, not only validity", fontsize=14, y=1.02)
    fig.text(
        0.5,
        -0.05,
        f"K=8, p=3; {test['network_id'].nunique()} held-out networks; "
        "bars are 95% network bootstrap intervals",
        ha="center",
    )
    fig.subplots_adjust(bottom=0.20, wspace=0.24)
    save(fig, "fig17_penalty_sensitivity")


def main() -> None:
    style()
    builders = [
        feasibility_optimality_depth,
        entropy_uncertainty_topology,
        entropy_risk_regret,
        operational_violation_popt,
        adaptive_conformal,
        noise_robustness,
        conceptual_pipeline,
        targeted_k_scaling,
        penalty_sensitivity,
    ]
    for builder in builders:
        builder()
    outputs = sorted(
        path
        for path in OUT.iterdir()
        if path.is_file()
        and path.suffix in {".png", ".pdf"}
        and path.stem.startswith(tuple(f"fig{number:02d}" for number in range(9, 18)))
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": _sha256(Path(__file__)),
        "source_sha256": {
            name: _sha256(ROOT / name) for name in SOURCE_ARTIFACTS
        },
        "artifact_sha256": {
            path.name: _sha256(path) for path in outputs
        },
        "manual_numeric_inputs": False,
    }
    (OUT / "science_freeze_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(builders)} science-freeze figures as PNG and PDF to {OUT}")


if __name__ == "__main__":
    main()
