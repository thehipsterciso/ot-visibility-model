"""
hypotheses/runner.py — All 10 hypothesis implementations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from scipy import stats

from src.assets.graph import MeridianGraph
from src.constants import (
    ControlStrategy, HypothesisVerdict, Meridian, SimConfig, ThreatActor
)
from src.simulation.engine import SimulationResults


@dataclass
class HypothesisResult:
    hypothesis_id: str
    title: str
    verdict: HypothesisVerdict
    primary_metric: str
    primary_value: float
    key_finding: str
    supporting_metrics: dict = field(default_factory=dict)
    falsification_condition: str = ""
    falsification_met: bool = False
    p_value: Optional[float] = None
    effect_size: Optional[float] = None


class BaseHypothesis(ABC):
    hypothesis_id: str
    title: str
    falsification_condition: str

    @abstractmethod
    def evaluate(self, **kwargs) -> HypothesisResult:
        ...

    def _cohens_d(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        pooled_std = math.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
        if pooled_std == 0:
            return 0.0
        return (np.mean(a) - np.mean(b)) / pooled_std


class H1BortHypothesis(BaseHypothesis):
    hypothesis_id = "H1"
    title = "Bort: Checkpoint-optimized strategy achieves equivalent security outcomes to inventory-informed strategy"
    falsification_condition = "Incidents with inventory-informed strategy produce materially better outcomes than checkpoint-optimized strategy across all threat actor types"

    def evaluate(self, checkpoint_results: SimulationResults, inventory_results: SimulationResults) -> HypothesisResult:
        cp_impacts = [i.net_financial_impact for i in checkpoint_results.incidents]
        inv_impacts = [i.net_financial_impact for i in inventory_results.incidents]
        cp_mean = float(np.mean(cp_impacts))
        inv_mean = float(np.mean(inv_impacts))
        t_stat, p_value = stats.ttest_ind(cp_impacts, inv_impacts)
        effect_size = self._cohens_d(cp_impacts, inv_impacts)
        significant_difference = p_value < 0.05 and abs(effect_size) > 0.2
        cp_detection = checkpoint_results.detection_rate()
        inv_detection = inventory_results.detection_rate()
        cp_ta3 = [i for i in checkpoint_results.incidents if i.threat_actor == ThreatActor.TA3_INSIDER_VENDOR]
        inv_ta3 = [i for i in inventory_results.incidents if i.threat_actor == ThreatActor.TA3_INSIDER_VENDOR]
        cp_ta3_detection = sum(1 for i in cp_ta3 if i.detected) / max(1, len(cp_ta3))
        inv_ta3_detection = sum(1 for i in inv_ta3 if i.detected) / max(1, len(inv_ta3))
        if not significant_difference:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (f"No statistically significant difference: checkpoint ${cp_mean:,.0f} vs inventory ${inv_mean:,.0f} mean loss (p={p_value:.3f}). H1 supported within this threat model.")
        elif cp_ta3_detection < inv_ta3_detection * 0.7:
            verdict = HypothesisVerdict.FAILED
            finding = (f"Checkpoint fails against direct-access actors: TA-3 detection {cp_ta3_detection:.1%} (checkpoint) vs {inv_ta3_detection:.1%} (inventory-informed). H1 conditionally valid for IT-traversing actors only.")
        else:
            verdict = HypothesisVerdict.INCONCLUSIVE
            finding = (f"Mixed results (p={p_value:.3f}, d={effect_size:.2f}). Checkpoint adequate for narrow threat models.")
        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Mean net financial impact difference (checkpoint vs inventory)",
            primary_value=cp_mean - inv_mean, key_finding=finding,
            supporting_metrics={"checkpoint_mean_loss": cp_mean, "inventory_mean_loss": inv_mean,
                "checkpoint_detection_rate": cp_detection, "inventory_detection_rate": inv_detection,
                "ta3_checkpoint_detection": cp_ta3_detection, "ta3_inventory_detection": inv_ta3_detection,
                "effect_size_cohens_d": effect_size},
            falsification_condition=self.falsification_condition, falsification_met=significant_difference,
            p_value=float(p_value), effect_size=effect_size,
        )


class H2FoundationHypothesis(BaseHypothesis):
    hypothesis_id = "H2"
    title = "Asset inventory is a prerequisite — decision quality degrades measurably as completeness decreases"
    falsification_condition = "Security outcomes are statistically equivalent across inventory completeness levels"

    def evaluate(self, sweep_results: list[SimulationResults]) -> HypothesisResult:
        completeness_levels = [r.inventory_completeness for r in sweep_results]
        detection_rates = [r.detection_rate() for r in sweep_results]
        mean_impacts = [r.mean_net_impact() for r in sweep_results]
        response_accuracies = [r.mean_response_accuracy() for r in sweep_results]
        corr_detection, p_detection = stats.pearsonr(completeness_levels, detection_rates)
        corr_impact, p_impact = stats.pearsonr(completeness_levels, [-x for x in mean_impacts])
        strong_correlation = (abs(corr_detection) > 0.7 and p_detection < 0.05 and abs(corr_impact) > 0.7 and p_impact < 0.05)
        if strong_correlation:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (f"Strong correlation: detection r={corr_detection:.3f} (p={p_detection:.3f}), impact r={corr_impact:.3f} (p={p_impact:.3f}). Inventory is the foundational variable.")
        elif abs(corr_detection) > 0.4 or abs(corr_impact) > 0.4:
            verdict = HypothesisVerdict.INCONCLUSIVE
            finding = f"Moderate correlation. Inventory influences outcomes but not dominant across all scenarios."
        else:
            verdict = HypothesisVerdict.FAILED
            finding = f"Weak correlation. Checkpoint controls may compensate across tested completeness range."
        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Pearson correlation: inventory completeness vs detection rate",
            primary_value=float(corr_detection), key_finding=finding,
            supporting_metrics={"completeness_levels": completeness_levels, "detection_rates": detection_rates,
                "mean_impacts": mean_impacts, "response_accuracies": response_accuracies,
                "correlation_detection": float(corr_detection), "p_value_detection": float(p_detection),
                "correlation_impact": float(corr_impact), "p_value_impact": float(p_impact)},
            falsification_condition=self.falsification_condition, falsification_met=not strong_correlation,
            p_value=float(p_detection),
        )


class H4BlastRadiusHypothesis(BaseHypothesis):
    hypothesis_id = "H4"
    title = "Uninventoried OT assets create super-linear attacker advantage growth as inventory completeness decreases"
    falsification_condition = "Attacker advantage scales linearly with uninventoried asset count"

    def evaluate(self, sweep_results: list[SimulationResults]) -> HypothesisResult:
        completeness_levels = [r.inventory_completeness for r in sweep_results]
        mean_attacker_advantages = [r.mean_attacker_advantage() for r in sweep_results]
        x = np.array(completeness_levels)
        y = np.array(mean_attacker_advantages)
        linear_coeffs = np.polyfit(x, y, 1)
        linear_pred = np.polyval(linear_coeffs, x)
        quad_coeffs = np.polyfit(x, y, 2)
        quad_pred = np.polyval(quad_coeffs, x)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_linear = float(1 - np.sum((y - linear_pred) ** 2) / ss_tot) if ss_tot > 0 else 0
        r2_quad = float(1 - np.sum((y - quad_pred) ** 2) / ss_tot) if ss_tot > 0 else 0
        super_linear = (r2_quad > r2_linear + 0.05 and quad_coeffs[0] > 0 and
            mean_attacker_advantages[0] > mean_attacker_advantages[-1] * 2.5)
        advantage_at_40pct = next((r.mean_attacker_advantage() for r in sweep_results if abs(r.inventory_completeness - 0.40) < 0.05), None)
        advantage_at_100pct = next((r.mean_attacker_advantage() for r in sweep_results if abs(r.inventory_completeness - 1.0) < 0.05), None)
        # Fallback to lowest-completeness result if 40% not in sweep
        baseline_advantage = advantage_at_40pct or mean_attacker_advantages[0]
        baseline_completeness = 0.40 if advantage_at_40pct is not None else completeness_levels[0]
        if super_linear:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (f"Super-linear attacker advantage: quadratic R²={r2_quad:.3f} vs linear R²={r2_linear:.3f}. At {baseline_completeness:.0%} inventory, attacker sees {baseline_advantage:.1f}x more of the OT blast radius than the defender.")
        elif r2_quad > r2_linear:
            verdict = HypothesisVerdict.INCONCLUSIVE
            finding = f"Some evidence of super-linearity but not conclusive. Quadratic R²={r2_quad:.3f} vs linear R²={r2_linear:.3f}."
        else:
            verdict = HypothesisVerdict.FAILED
            finding = f"Linear scaling. Unknown assets add proportional risk. Linear R²={r2_linear:.3f}."
        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Attacker advantage ratio at Meridian baseline (40% OT inventory)",
            primary_value=baseline_advantage, key_finding=finding,
            supporting_metrics={"completeness_levels": completeness_levels,
                "mean_attacker_advantages": mean_attacker_advantages,
                "r2_linear": r2_linear, "r2_quadratic": r2_quad,
                "advantage_at_40pct": advantage_at_40pct, "advantage_at_100pct": advantage_at_100pct},
            falsification_condition=self.falsification_condition, falsification_met=not super_linear,
        )


class H5InsiderHypothesis(BaseHypothesis):
    hypothesis_id = "H5"
    title = "Checkpoint model fails categorically against threat actors with direct OT access"
    falsification_condition = "Checkpoint controls detect TA-3 at equivalent rates to inventory-informed monitoring"

    def evaluate(self, checkpoint_ta3_results: SimulationResults, inventory_ta3_results: SimulationResults) -> HypothesisResult:
        cp_incidents = checkpoint_ta3_results.incidents
        inv_incidents = inventory_ta3_results.incidents
        cp_detection = sum(1 for i in cp_incidents if i.detected) / max(1, len(cp_incidents))
        inv_detection = sum(1 for i in inv_incidents if i.detected) / max(1, len(inv_incidents))
        cp_impacts = [i.net_financial_impact for i in cp_incidents]
        inv_impacts = [i.net_financial_impact for i in inv_incidents]
        detection_gap = inv_detection - cp_detection
        impact_gap = float(np.mean(cp_impacts) - np.mean(inv_impacts))
        t_stat, p_value = stats.ttest_ind(cp_impacts, inv_impacts)
        categorical_failure = detection_gap > 0.20 and p_value < 0.05
        if categorical_failure:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (f"Checkpoint TA-3 detection: {cp_detection:.1%} vs inventory-informed {inv_detection:.1%}. Gap of {detection_gap:.1%} — checkpoint controls structurally blind to direct OT access. Impact gap: ${impact_gap:,.0f}.")
        elif detection_gap > 0.10:
            verdict = HypothesisVerdict.INCONCLUSIVE
            finding = f"Detection gap of {detection_gap:.1%} exists but not conclusive (p={p_value:.3f})."
        else:
            verdict = HypothesisVerdict.FAILED
            finding = f"No significant TA-3 detection gap (gap={detection_gap:.1%}, p={p_value:.3f})."
        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Detection rate gap: inventory-informed vs checkpoint-only for TA-3",
            primary_value=detection_gap, key_finding=finding,
            supporting_metrics={"checkpoint_ta3_detection_rate": cp_detection,
                "inventory_ta3_detection_rate": inv_detection, "detection_gap": detection_gap,
                "checkpoint_mean_impact": float(np.mean(cp_impacts)),
                "inventory_mean_impact": float(np.mean(inv_impacts)), "impact_gap": impact_gap},
            falsification_condition=self.falsification_condition, falsification_met=not categorical_failure,
            p_value=float(p_value),
        )


class H7ResponseInflectionHypothesis(BaseHypothesis):
    hypothesis_id = "H7"
    title = "Incident response degrades non-linearly below a critical inventory completeness threshold"
    falsification_condition = "Response accuracy degrades linearly with inventory completeness"

    def evaluate(self, sweep_results: list[SimulationResults]) -> HypothesisResult:
        completeness_levels = [r.inventory_completeness for r in sweep_results]
        response_accuracies = [r.mean_response_accuracy() for r in sweep_results]
        mean_mttrs = []
        for r in sweep_results:
            mttrs = [i.mttr_hours for i in r.incidents if i.detected]
            mean_mttrs.append(float(np.mean(mttrs)) if mttrs else Meridian.MTTR_OT_HOURS * 4)
        x = np.array(completeness_levels)
        y_acc = np.array(response_accuracies)
        if len(x) >= 4:
            # Fit a degree-4 polynomial and find the inflection as the point of
            # maximum first derivative (most rapid change) — more robust than the
            # cubic inflection formula, which fails when the cubic opens upward.
            degree = min(4, len(x) - 1)
            poly_coeffs = np.polyfit(x, y_acc, degree)
            xs_dense = np.linspace(float(x.min()), float(x.max()), 500)
            y_dense = np.polyval(poly_coeffs, xs_dense)
            dy = np.gradient(y_dense, xs_dense)
            inflection_x = float(xs_dense[int(np.argmax(dy))])
            inflection_x = float(np.clip(inflection_x, 0, 1))
        else:
            inflection_x = 0.65
        linear_pred = np.polyval(np.polyfit(x, y_acc, 1), x)
        quad_pred = np.polyval(np.polyfit(x, y_acc, 2), x)
        ss_tot = float(np.sum((y_acc - np.mean(y_acc)) ** 2))
        r2_linear = float(1 - np.sum((y_acc - linear_pred) ** 2) / ss_tot) if ss_tot > 0 else 0
        r2_quad = float(1 - np.sum((y_acc - quad_pred) ** 2) / ss_tot) if ss_tot > 0 else 0
        non_linear = r2_quad > r2_linear + 0.05 and 0.30 < inflection_x < 0.85
        if non_linear:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (f"Non-linear response degradation confirmed. Inflection at ~{inflection_x:.0%} inventory coverage. Below this, responders shift from containment to discovery. Quadratic R²={r2_quad:.3f} vs linear R²={r2_linear:.3f}.")
        else:
            verdict = HypothesisVerdict.FAILED
            finding = f"Linear degradation. No clear inflection at {inflection_x:.0%}. Linear R²={r2_linear:.3f}, Quadratic R²={r2_quad:.3f}."
        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Inflection point in response accuracy curve",
            primary_value=inflection_x, key_finding=finding,
            supporting_metrics={"completeness_levels": completeness_levels,
                "response_accuracies": response_accuracies, "mean_mttrs": mean_mttrs,
                "inflection_point": inflection_x, "r2_linear": r2_linear, "r2_quadratic": r2_quad},
            falsification_condition=self.falsification_condition, falsification_met=not non_linear,
        )


class H10CostParityHypothesis(BaseHypothesis):
    hypothesis_id = "H10"
    title = "Cost of OT asset inventory program < expected loss from inventory gaps over 3 years"
    falsification_condition = "Expected 3-year inventory gap loss is lower than program cost"
    INVENTORY_PROGRAM_COST = 485_000

    def evaluate(self, baseline_results: SimulationResults, full_inv_results: SimulationResults) -> HypothesisResult:
        baseline_annual = baseline_results.total_expected_annual_loss()
        full_inv_annual = full_inv_results.total_expected_annual_loss()
        annual_gap_loss = baseline_annual - full_inv_annual
        three_year_gap_loss = annual_gap_loss * 3
        roi = three_year_gap_loss - self.INVENTORY_PROGRAM_COST
        payback_years = self.INVENTORY_PROGRAM_COST / max(1, annual_gap_loss)
        program_justified = three_year_gap_loss > self.INVENTORY_PROGRAM_COST
        inversion_multiplier = self.INVENTORY_PROGRAM_COST / max(1, three_year_gap_loss)
        if program_justified and payback_years < 2.0:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (f"Positive ROI. 3-year gap loss: ${three_year_gap_loss:,.0f}. Program cost: ${self.INVENTORY_PROGRAM_COST:,.0f}. Net benefit: ${roi:,.0f}. Payback: {payback_years:.1f} years. ROI inverts only if incident frequency drops to {inversion_multiplier:.1%} of modeled rate.")
        elif program_justified:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (f"Program justified, payback {payback_years:.1f} years. Gap loss ${three_year_gap_loss:,.0f} vs cost ${self.INVENTORY_PROGRAM_COST:,.0f}.")
        else:
            verdict = HypothesisVerdict.FAILED
            finding = (f"3-year gap loss ${three_year_gap_loss:,.0f} does not exceed program cost ${self.INVENTORY_PROGRAM_COST:,.0f}. Cost argument against inventory has merit under this threat frequency.")
        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="3-year net ROI of inventory program", primary_value=roi,
            key_finding=finding,
            supporting_metrics={"baseline_annual_loss": baseline_annual, "full_inventory_annual_loss": full_inv_annual,
                "annual_gap_loss": annual_gap_loss, "three_year_gap_loss": three_year_gap_loss,
                "program_cost": self.INVENTORY_PROGRAM_COST, "net_roi": roi,
                "payback_years": payback_years, "inversion_multiplier": inversion_multiplier},
            falsification_condition=self.falsification_condition, falsification_met=not program_justified,
        )


class H3ActionabilityGapHypothesis(BaseHypothesis):
    hypothesis_id = "H3"
    title = "Detection lead time has independent value — early visibility reduces impact regardless of response maturity"
    falsification_condition = "No significant negative correlation between detection lead time and net financial impact"

    def evaluate(self, sweep_results: list[SimulationResults]) -> HypothesisResult:
        lead_times = []
        impacts = []
        for r in sweep_results:
            for incident in r.incidents:
                if incident.detected and incident.detection_lead_time_hours > 0:
                    lead_times.append(incident.detection_lead_time_hours)
                    impacts.append(incident.net_financial_impact)

        if len(lead_times) < 10:
            return HypothesisResult(
                hypothesis_id=self.hypothesis_id, title=self.title,
                verdict=HypothesisVerdict.INCONCLUSIVE,
                primary_metric="Pearson correlation: detection lead time vs net financial impact",
                primary_value=0.0,
                key_finding="Insufficient detected incidents to evaluate actionability gap.",
                falsification_condition=self.falsification_condition,
            )

        corr, p_value = stats.pearsonr(lead_times, impacts)
        strong_negative = corr < -0.5 and p_value < 0.05

        # Quantify the magnitude: compare mean impact at top vs bottom quartile of lead time
        lead_arr = np.array(lead_times)
        impact_arr = np.array(impacts)
        q25 = float(np.percentile(lead_arr, 25))
        q75 = float(np.percentile(lead_arr, 75))
        early_detection_impact = float(np.mean(impact_arr[lead_arr <= q25]))
        late_detection_impact = float(np.mean(impact_arr[lead_arr >= q75]))
        impact_reduction_pct = (late_detection_impact - early_detection_impact) / max(1, late_detection_impact)

        if strong_negative:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (
                f"Detection lead time negatively correlates with financial impact (r={corr:.3f}, p={p_value:.4f}). "
                f"Early detection (bottom quartile) produces {impact_reduction_pct:.0%} lower impact than late detection. "
                f"Visibility and actionability are separable problems — deprioritizing visibility because response maturity "
                f"is low is a category error: the value of early detection exists independent of response speed."
            )
        elif corr < -0.3 and p_value < 0.05:
            verdict = HypothesisVerdict.INCONCLUSIVE
            finding = (
                f"Moderate negative correlation (r={corr:.3f}, p={p_value:.4f}). "
                f"Some evidence that earlier detection reduces impact, but effect size below threshold."
            )
        else:
            verdict = HypothesisVerdict.FAILED
            finding = (
                f"Weak or non-significant correlation (r={corr:.3f}, p={p_value:.4f}). "
                f"Detection lead time does not materially reduce financial impact in this model."
            )

        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Pearson correlation: detection lead time vs net financial impact",
            primary_value=float(corr), key_finding=finding,
            supporting_metrics={
                "pearson_r": float(corr), "p_value": float(p_value),
                "n_detected_incidents": len(lead_times),
                "early_detection_mean_impact": early_detection_impact,
                "late_detection_mean_impact": late_detection_impact,
                "impact_reduction_early_vs_late": impact_reduction_pct,
            },
            falsification_condition=self.falsification_condition,
            falsification_met=not strong_negative,
            p_value=float(p_value), effect_size=float(corr),
        )


class H6SegmentationQualityHypothesis(BaseHypothesis):
    hypothesis_id = "H6"
    title = "Segmentation built on incomplete inventory leaves blind attack paths to crown jewels"
    falsification_condition = "Blind segmentation gap count does not decrease significantly as inventory completeness increases"

    def evaluate(self, sweep_results: list[SimulationResults], graph: MeridianGraph) -> HypothesisResult:
        completeness_levels = [r.inventory_completeness for r in sweep_results]
        blind_gap_counts = []

        crown_jewels = graph.get_crown_jewels()
        entry_points = list({
            n for n, d in graph.graph.nodes(data=True)
            if d.get("purdue_level", 0) >= 4
        })[:20]  # cap to keep runtime reasonable

        for level in completeness_levels:
            # Build a fresh graph at this completeness level to get correct inventoried flags
            from src.assets.graph import MeridianGraph as MG
            level_graph = MG(inventory_completeness=level, seed=graph.seed).build()
            uninventoried = set(level_graph.get_uninventoried_nodes())
            # Use undirected graph — graph edges are directional (OT→IT) but attackers
            # traverse bidirectionally; segmentation gaps must account for both directions.
            undirected = level_graph.graph.to_undirected()

            blind_paths = 0
            for entry in entry_points:
                if entry not in undirected:
                    continue
                for crown in crown_jewels:
                    if crown not in undirected:
                        continue
                    try:
                        paths = list(nx.all_simple_paths(
                            undirected, entry, crown, cutoff=4
                        ))
                        blind_paths += sum(
                            1 for p in paths
                            if any(node in uninventoried for node in p[1:-1])
                        )
                    except (nx.NetworkXError, nx.NodeNotFound):
                        continue

            blind_gap_counts.append(blind_paths)

        # H6 is SUPPORTED if blind gap count drops significantly as completeness increases
        corr, p_value = stats.pearsonr(completeness_levels, blind_gap_counts) if len(completeness_levels) >= 3 else (0.0, 1.0)
        drop_from_baseline = blind_gap_counts[0] - blind_gap_counts[-1] if blind_gap_counts else 0
        significant_drop = corr < -0.5 and p_value < 0.05 and drop_from_baseline > 0

        baseline_gaps = blind_gap_counts[0] if blind_gap_counts else 0
        full_inv_gaps = blind_gap_counts[-1] if blind_gap_counts else 0
        pct_reduction = (baseline_gaps - full_inv_gaps) / max(1, baseline_gaps)

        if significant_drop:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (
                f"Blind segmentation gaps drop from {baseline_gaps} paths at "
                f"{completeness_levels[0]:.0%} inventory to {full_inv_gaps} at "
                f"{completeness_levels[-1]:.0%} — a {pct_reduction:.0%} reduction (r={corr:.3f}, p={p_value:.4f}). "
                f"Segmentation designers cannot block paths they cannot see. "
                f"At Meridian's 40% inventory coverage, a material fraction of crown jewel attack paths "
                f"traverse assets that were never considered in segmentation design."
            )
        elif drop_from_baseline > 0:
            verdict = HypothesisVerdict.INCONCLUSIVE
            finding = (
                f"Some reduction in blind paths ({baseline_gaps} → {full_inv_gaps}) but correlation "
                f"below threshold (r={corr:.3f}, p={p_value:.4f})."
            )
        else:
            verdict = HypothesisVerdict.FAILED
            finding = (
                f"No significant reduction in blind segmentation paths as completeness increases "
                f"(r={corr:.3f}). Segmentation may be effective despite partial inventory."
            )

        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Blind crown-jewel attack paths (traverse uninventoried nodes)",
            primary_value=float(baseline_gaps), key_finding=finding,
            supporting_metrics={
                "completeness_levels": completeness_levels,
                "blind_gap_counts": blind_gap_counts,
                "correlation_completeness_vs_gaps": float(corr),
                "p_value": float(p_value),
                "baseline_blind_gaps": baseline_gaps,
                "full_inventory_blind_gaps": full_inv_gaps,
                "pct_reduction": pct_reduction,
            },
            falsification_condition=self.falsification_condition,
            falsification_met=not significant_drop,
            p_value=float(p_value),
        )


class H8RiskQuantificationHypothesis(BaseHypothesis):
    hypothesis_id = "H8"
    title = "FAIR-style risk outputs are unreliable at partial inventory — partial coverage systematically understates risk"
    falsification_condition = "Coefficient of variation of net financial impact is equivalent across completeness levels"

    def evaluate(self, sweep_results: list[SimulationResults]) -> HypothesisResult:
        completeness_levels = [r.inventory_completeness for r in sweep_results]
        cv_by_level = []
        mean_by_level = []

        for r in sweep_results:
            impacts = [i.net_financial_impact for i in r.incidents]
            mean_impact = float(np.mean(impacts))
            std_impact = float(np.std(impacts))
            cv = std_impact / mean_impact if mean_impact > 0 else 0.0
            cv_by_level.append(cv)
            mean_by_level.append(mean_impact)

        # Systematic understatement: mean impact at each level vs mean impact at 100%
        full_inventory_mean = mean_by_level[-1] if mean_by_level else 0.0
        understatement_by_level = [
            (full_inventory_mean - m) / max(1, full_inventory_mean)
            for m in mean_by_level
        ]

        # H8 SUPPORTED if CV is materially higher at low completeness
        low_cv = cv_by_level[0] if cv_by_level else 0.0
        high_cv = cv_by_level[-1] if cv_by_level else 0.0
        cv_corr, cv_p = stats.pearsonr(completeness_levels, cv_by_level) if len(cv_by_level) >= 3 else (0.0, 1.0)
        cv_materially_higher = low_cv > high_cv * 1.10 and cv_corr < -0.3

        baseline_understatement = understatement_by_level[
            next((i for i, lvl in enumerate(completeness_levels) if abs(lvl - 0.40) < 0.05), 0)
        ]

        if cv_materially_higher:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (
                f"Risk estimates are materially less reliable at low inventory coverage: "
                f"CV={low_cv:.2f} at {completeness_levels[0]:.0%} vs CV={high_cv:.2f} at {completeness_levels[-1]:.0%}. "
                f"At Meridian's 40% baseline, FAIR models understate expected loss by ~{baseline_understatement:.0%} "
                f"relative to full-inventory estimates. Decision-makers are receiving systematically low risk numbers."
            )
        elif low_cv > high_cv:
            verdict = HypothesisVerdict.INCONCLUSIVE
            finding = (
                f"CV higher at low completeness (CV={low_cv:.2f} vs {high_cv:.2f}) "
                f"but difference below materiality threshold."
            )
        else:
            verdict = HypothesisVerdict.FAILED
            finding = (
                f"CV does not increase at low completeness levels (CV={low_cv:.2f} low vs "
                f"{high_cv:.2f} high). Risk estimates are similarly reliable across coverage levels."
            )

        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Coefficient of variation of net financial impact at baseline (40%) inventory",
            primary_value=float(low_cv), key_finding=finding,
            supporting_metrics={
                "completeness_levels": completeness_levels,
                "cv_by_level": cv_by_level,
                "mean_impact_by_level": mean_by_level,
                "understatement_by_level": understatement_by_level,
                "full_inventory_mean_impact": full_inventory_mean,
                "cv_correlation_with_completeness": float(cv_corr),
                "baseline_understatement_pct": baseline_understatement,
            },
            falsification_condition=self.falsification_condition,
            falsification_met=not cv_materially_higher,
            p_value=float(cv_p),
        )


class H9ComplianceExposureHypothesis(BaseHypothesis):
    hypothesis_id = "H9"
    title = "Inventory gaps translate directly to regulatory exposure — NIST CSF ID.AM controls require asset enumeration"
    falsification_condition = "Regulatory exposure does not decrease monotonically as inventory completeness increases"

    # NIST CSF 2.0 Identify / Asset Management controls requiring asset enumeration
    NIST_AM_CONTROLS = ["ID.AM-1", "ID.AM-2", "ID.AM-3", "ID.AM-4", "ID.AM-5"]

    def evaluate(self, sweep_results: list[SimulationResults]) -> HypothesisResult:
        completeness_levels = [r.inventory_completeness for r in sweep_results]
        mean_reg_exposure_by_level = []
        annualized_exposure_by_level = []
        mean_gaps_by_level = []

        from src.constants import ThreatActorConfig
        annual_frequency = float(sum(ThreatActorConfig.ANNUAL_FREQUENCY.values()))

        for r in sweep_results:
            reg_exposures = [i.regulatory_exposure for i in r.incidents]
            gaps = [i.compliance_gaps for i in r.incidents]
            mean_reg_exposure_by_level.append(float(np.mean(reg_exposures)))
            mean_gaps_by_level.append(float(np.mean(gaps)))
            # Annualize: mean exposure per incident × expected annual incident count
            annual_exposure = float(np.mean(reg_exposures)) * annual_frequency
            annualized_exposure_by_level.append(annual_exposure)

        # Count impacted NIST CSF ID.AM controls based on average inventory gap
        nist_impact_by_level = []
        for level in completeness_levels:
            inv_gap = 1.0 - level
            # Each ID.AM control requires asset enumeration; gap determines partial/full non-compliance
            # At 0% gap = full compliance, at 100% gap = zero controls met
            impacted = round(inv_gap * len(self.NIST_AM_CONTROLS))
            nist_impact_by_level.append(impacted)

        # H9 SUPPORTED if exposure drops materially and monotonically
        monotonic_drop = all(
            annualized_exposure_by_level[i] >= annualized_exposure_by_level[i + 1]
            for i in range(len(annualized_exposure_by_level) - 1)
        )
        exposure_at_baseline = annualized_exposure_by_level[
            next((i for i, lvl in enumerate(completeness_levels) if abs(lvl - 0.40) < 0.05), 0)
        ]
        exposure_at_full = annualized_exposure_by_level[-1]
        exposure_reduction = exposure_at_baseline - exposure_at_full
        nist_at_baseline = nist_impact_by_level[
            next((i for i, lvl in enumerate(completeness_levels) if abs(lvl - 0.40) < 0.05), 0)
        ]

        if monotonic_drop and exposure_reduction > 0:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (
                f"Annualized regulatory exposure drops monotonically from ${exposure_at_baseline:,.0f} "
                f"at 40% inventory to ${exposure_at_full:,.0f} at full coverage — "
                f"a ${exposure_reduction:,.0f} reduction. "
                f"At Meridian's baseline, {nist_at_baseline}/{len(self.NIST_AM_CONTROLS)} NIST CSF 2.0 "
                f"Identify controls (ID.AM-1 through ID.AM-5) are partially or wholly unmet because "
                f"all five require asset enumeration."
            )
        elif exposure_reduction > 0:
            verdict = HypothesisVerdict.INCONCLUSIVE
            finding = (
                f"Exposure decreases overall (${exposure_at_baseline:,.0f} → ${exposure_at_full:,.0f}) "
                f"but not monotonically. Non-monotonic pattern may reflect simulation noise."
            )
        else:
            verdict = HypothesisVerdict.FAILED
            finding = (
                f"Regulatory exposure does not decrease with higher inventory completeness. "
                f"Compliance exposure may be driven by incident frequency rather than inventory coverage."
            )

        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Mean annualized regulatory exposure at baseline (40%) inventory",
            primary_value=float(exposure_at_baseline), key_finding=finding,
            supporting_metrics={
                "completeness_levels": completeness_levels,
                "mean_regulatory_exposure_by_level": mean_reg_exposure_by_level,
                "annualized_regulatory_exposure_by_level": annualized_exposure_by_level,
                "mean_compliance_gaps_by_level": mean_gaps_by_level,
                "nist_am_controls_impacted_by_level": nist_impact_by_level,
                "nist_am_controls": self.NIST_AM_CONTROLS,
                "exposure_reduction_baseline_to_full": exposure_reduction,
                "monotonic_drop": monotonic_drop,
            },
            falsification_condition=self.falsification_condition,
            falsification_met=not (monotonic_drop and exposure_reduction > 0),
        )
