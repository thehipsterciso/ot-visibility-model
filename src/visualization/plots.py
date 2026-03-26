"""
visualization/plots.py — Per-hypothesis chart functions.
Each function returns a matplotlib Figure ready for embedding.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

from src.hypotheses.runner import HypothesisResult
from src.simulation.engine import SimulationResults

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

PALETTE = {
    "checkpoint": "#e05c2a",
    "inventory": "#2a7ae0",
    "neutral": "#555555",
    "background": "#f9f9f7",
    "grid": "#e0e0e0",
    "supported": "#2e7d32",
    "failed": "#c62828",
    "inconclusive": "#f57c00",
}


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(PALETTE["background"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])


def _verdict_color(verdict_str: str) -> str:
    return {
        "SUPPORTED": PALETTE["supported"],
        "FAILED": PALETTE["failed"],
        "INCONCLUSIVE": PALETTE["inconclusive"],
    }.get(verdict_str, PALETTE["neutral"])


# ---------------------------------------------------------------------------
# H1 — Side-by-side KDE: checkpoint vs inventory-informed net impact
# ---------------------------------------------------------------------------

def plot_h1(
    result: HypothesisResult,
    checkpoint_results: SimulationResults,
    inventory_results: SimulationResults,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    cp_impacts = [i.net_financial_impact / 1_000 for i in checkpoint_results.incidents]
    inv_impacts = [i.net_financial_impact / 1_000 for i in inventory_results.incidents]

    def _kde_line(data, color, label, ax):
        kde = stats.gaussian_kde(data, bw_method=0.3)
        xs = np.linspace(min(data), max(data), 300)
        ys = kde(xs)
        ax.fill_between(xs, ys, alpha=0.25, color=color)
        ax.plot(xs, ys, color=color, linewidth=2, label=label)
        ax.axvline(np.mean(data), color=color, linestyle="--", linewidth=1.2, alpha=0.8)

    _kde_line(cp_impacts, PALETTE["checkpoint"], f"Checkpoint-Optimized (μ=${np.mean(cp_impacts):,.0f}K)", ax)
    _kde_line(inv_impacts, PALETTE["inventory"], f"Inventory-Informed (μ=${np.mean(inv_impacts):,.0f}K)", ax)

    _style_ax(ax, "H1 — Bort Hypothesis: Loss Distribution Comparison", "Net Financial Impact ($K)", "Density")
    ax.legend(fontsize=8)
    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# H2 — Dual-axis line: detection rate + response accuracy vs completeness
# ---------------------------------------------------------------------------

def plot_h2(
    result: HypothesisResult,
    sweep_results: list[SimulationResults],
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["background"])
    ax2 = ax1.twinx()

    levels = [r.inventory_completeness * 100 for r in sweep_results]
    detection = [r.detection_rate() * 100 for r in sweep_results]
    accuracy = [r.mean_response_accuracy() * 100 for r in sweep_results]

    ax1.plot(levels, detection, "o-", color=PALETTE["checkpoint"], linewidth=2, label="Detection Rate (%)")
    ax2.plot(levels, accuracy, "s--", color=PALETTE["inventory"], linewidth=2, label="Response Accuracy (%)")

    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    _style_ax(ax1, "H2 — Foundation Hypothesis: Security Outcomes vs Inventory Coverage",
              "Inventory Completeness (%)", "Detection Rate (%)")
    ax2.set_ylabel("Response Accuracy (%)", fontsize=9, color=PALETTE["inventory"])
    ax2.tick_params(colors=PALETTE["inventory"], labelsize=8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# H3 — Scatter: detection lead time vs net financial impact
# ---------------------------------------------------------------------------

def plot_h3(
    result: HypothesisResult,
    sweep_results: list[SimulationResults],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    lead_times, impacts = [], []
    for r in sweep_results:
        for inc in r.incidents:
            if inc.detected and inc.detection_lead_time_hours > 0:
                lead_times.append(inc.detection_lead_time_hours)
                impacts.append(inc.net_financial_impact / 1_000)

    # Sample for readability
    if len(lead_times) > 2000:
        idx = np.random.default_rng(42).choice(len(lead_times), 2000, replace=False)
        lead_times = [lead_times[i] for i in idx]
        impacts = [impacts[i] for i in idx]

    ax.scatter(lead_times, impacts, alpha=0.15, s=8, color=PALETTE["neutral"], zorder=2)

    if len(lead_times) >= 2:
        m, b, r, p, _ = stats.linregress(lead_times, impacts)
        xs = np.linspace(min(lead_times), max(lead_times), 200)
        ax.plot(xs, m * xs + b, color=PALETTE["checkpoint"], linewidth=2,
                label=f"OLS fit (r={r:.3f}, p={p:.4f})")

    _style_ax(ax, "H3 — Actionability Gap: Detection Lead Time vs Financial Impact",
              "Detection Lead Time (hours)", "Net Financial Impact ($K)")
    ax.legend(fontsize=8)
    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# H4 — Scatter + linear/quadratic fit: attacker advantage vs completeness
# ---------------------------------------------------------------------------

def plot_h4(
    result: HypothesisResult,
    sweep_results: list[SimulationResults],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    levels = np.array([r.inventory_completeness for r in sweep_results])
    advantages = np.array([r.mean_attacker_advantage() for r in sweep_results])

    ax.scatter(levels * 100, advantages, s=80, color=PALETTE["checkpoint"], zorder=5, label="Mean attacker advantage")

    xs = np.linspace(levels.min(), levels.max(), 200)
    lin = np.polyval(np.polyfit(levels, advantages, 1), xs)
    quad = np.polyval(np.polyfit(levels, advantages, 2), xs)
    ax.plot(xs * 100, lin, "--", color=PALETTE["neutral"], linewidth=1.5, label="Linear fit")
    ax.plot(xs * 100, quad, "-", color=PALETTE["inventory"], linewidth=2, label="Quadratic fit")

    ax.axhline(1.0, color=PALETTE["grid"], linewidth=1, linestyle=":")
    ax.text(levels.min() * 100 + 1, 1.03, "parity (1.0x)", fontsize=7, color=PALETTE["neutral"])

    _style_ax(ax, "H4 — Blast Radius: Attacker Advantage vs Inventory Coverage",
              "Inventory Completeness (%)", "Attacker Advantage Ratio (OT only)")
    ax.legend(fontsize=8)
    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# H5 — Grouped bar: TA-3 detection rate by strategy
# ---------------------------------------------------------------------------

def plot_h5(
    result: HypothesisResult,
    checkpoint_ta3_results: SimulationResults,
    inventory_ta3_results: SimulationResults,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    cp_det = checkpoint_ta3_results.detection_rate() * 100
    inv_det = inventory_ta3_results.detection_rate() * 100

    bars = ax.bar(
        ["Checkpoint-Only", "Inventory-Informed"],
        [cp_det, inv_det],
        color=[PALETTE["checkpoint"], PALETTE["inventory"]],
        width=0.5, zorder=3,
    )
    for bar, val in zip(bars, [cp_det, inv_det]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 110)
    _style_ax(ax, "H5 — Insider/Vendor: TA-3 Detection Rate by Control Strategy",
              "Control Strategy", "TA-3 Detection Rate (%)")
    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# H6 — Bar: blind segmentation gap count vs completeness
# ---------------------------------------------------------------------------

def plot_h6(
    result: HypothesisResult,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    sm = result.supporting_metrics
    levels = [lvl * 100 for lvl in sm.get("completeness_levels", [])]
    gaps = sm.get("blind_gap_counts", [])

    if levels and gaps:
        colors = [PALETTE["checkpoint"] if g > 0 else PALETTE["supported"] for g in gaps]
        ax.bar(levels, gaps, width=6, color=colors, zorder=3)
        ax.plot(levels, gaps, "o-", color=PALETTE["neutral"], linewidth=1.5, zorder=4)

    _style_ax(ax, "H6 — Segmentation Quality: Blind Crown Jewel Attack Paths vs Inventory Coverage",
              "Inventory Completeness (%)", "Blind Attack Paths to Crown Jewels")
    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# H7 — Line + sigmoid overlay: response accuracy vs completeness
# ---------------------------------------------------------------------------

def plot_h7(
    result: HypothesisResult,
    sweep_results: list[SimulationResults],
) -> plt.Figure:
    import math
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    levels = [r.inventory_completeness for r in sweep_results]
    accuracies = [r.mean_response_accuracy() * 100 for r in sweep_results]

    ax.plot([l * 100 for l in levels], accuracies, "o-",
            color=PALETTE["inventory"], linewidth=2, zorder=4, label="Response accuracy")

    inflection = result.supporting_metrics.get("inflection_point", 0.65)
    xs = np.linspace(0, 1, 200)
    # Sigmoid centered at inflection
    sigmoid = [20 + 75 / (1 + math.exp(-8 * (x - inflection))) for x in xs]
    ax.plot(xs * 100, sigmoid, "--", color=PALETTE["checkpoint"], linewidth=1.5,
            label="Sigmoid model", alpha=0.8)
    ax.axvline(inflection * 100, color=PALETTE["inconclusive"], linewidth=1.5,
               linestyle=":", label=f"Inflection ~{inflection:.0%}")

    _style_ax(ax, "H7 — Response Inflection: Accuracy vs Inventory Coverage",
              "Inventory Completeness (%)", "Mean Response Accuracy (%)")
    ax.legend(fontsize=8)
    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# H8 — Error band: mean impact ± 1 std by completeness
# ---------------------------------------------------------------------------

def plot_h8(
    result: HypothesisResult,
    sweep_results: list[SimulationResults],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    levels = [r.inventory_completeness * 100 for r in sweep_results]
    means, stds = [], []
    for r in sweep_results:
        impacts = [i.net_financial_impact / 1_000 for i in r.incidents]
        means.append(float(np.mean(impacts)))
        stds.append(float(np.std(impacts)))

    levels_arr = np.array(levels)
    means_arr = np.array(means)
    stds_arr = np.array(stds)

    ax.fill_between(levels_arr, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.25, color=PALETTE["checkpoint"], label="±1 std")
    ax.plot(levels_arr, means_arr, "o-", color=PALETTE["checkpoint"], linewidth=2, label="Mean impact")
    ax.plot(levels_arr, means_arr - stds_arr, "--", color=PALETTE["checkpoint"], linewidth=0.8, alpha=0.5)
    ax.plot(levels_arr, means_arr + stds_arr, "--", color=PALETTE["checkpoint"], linewidth=0.8, alpha=0.5)

    _style_ax(ax, "H8 — Risk Quantification Reliability: Impact Distribution by Coverage",
              "Inventory Completeness (%)", "Net Financial Impact ($K)")
    ax.legend(fontsize=8)
    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# H9 — Stacked bar: compliance gap count + dollar exposure by completeness
# ---------------------------------------------------------------------------

def plot_h9(
    result: HypothesisResult,
) -> plt.Figure:
    sm = result.supporting_metrics
    levels = [lvl * 100 for lvl in sm.get("completeness_levels", [])]
    annualized = [v / 1_000 for v in sm.get("annualized_regulatory_exposure_by_level", [])]
    nist_counts = sm.get("nist_am_controls_impacted_by_level", [])

    if not levels:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["background"])
    ax2 = ax1.twinx()

    width = 5
    ax1.bar(levels, annualized, width=width, color=PALETTE["checkpoint"], alpha=0.8,
            label="Annualized reg. exposure ($K)", zorder=3)
    ax2.plot(levels, nist_counts, "o--", color=PALETTE["inventory"], linewidth=2,
             label="NIST ID.AM controls impacted", zorder=4)

    _style_ax(ax1, "H9 — Compliance Exposure: Regulatory Risk vs Inventory Coverage",
              "Inventory Completeness (%)", "Annualized Regulatory Exposure ($K)")
    ax2.set_ylabel("NIST CSF ID.AM Controls Impacted", fontsize=9, color=PALETTE["inventory"])
    ax2.tick_params(colors=PALETTE["inventory"], labelsize=8)
    ax2.set_ylim(0, len(sm.get("nist_am_controls", [""] * 5)) + 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# H10 — Waterfall: gap loss / program cost / net ROI
# ---------------------------------------------------------------------------

def plot_h10(
    result: HypothesisResult,
) -> plt.Figure:
    sm = result.supporting_metrics
    gap_loss = sm.get("three_year_gap_loss", 0) / 1_000
    program_cost = sm.get("program_cost", 0) / 1_000
    net_roi = sm.get("net_roi", 0) / 1_000

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    labels = ["3-Year Gap Loss", "Program Cost", "Net ROI"]
    values = [gap_loss, -program_cost, net_roi]
    colors = [
        PALETTE["checkpoint"],
        PALETTE["neutral"],
        PALETTE["supported"] if net_roi >= 0 else PALETTE["failed"],
    ]

    running = 0.0
    bottoms = []
    for val in values:
        bottoms.append(running if val >= 0 else running + val)
        running += val

    for i, (label, val, color, bottom) in enumerate(zip(labels, values, colors, bottoms)):
        ax.bar(label, abs(val), bottom=bottom, color=color, width=0.5, zorder=3)
        ax.text(i, bottom + abs(val) / 2, f"${val:+,.0f}K",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.axhline(0, color=PALETTE["neutral"], linewidth=1)
    _style_ax(ax, "H10 — Cost Parity: 3-Year Inventory Program ROI",
              "", "Value ($K)")
    _add_verdict_badge(fig, result.verdict.value)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Shared verdict badge
# ---------------------------------------------------------------------------

def _add_verdict_badge(fig: plt.Figure, verdict: str) -> None:
    color = _verdict_color(verdict)
    fig.text(0.98, 0.98, verdict, ha="right", va="top", fontsize=9, fontweight="bold",
             color="white", bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85))
