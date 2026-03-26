"""
hypotheses/runner.py — All 10 hypothesis implementations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

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
        if super_linear:
            verdict = HypothesisVerdict.SUPPORTED
            finding = (f"Super-linear attacker advantage: quadratic R²={r2_quad:.3f} vs linear R²={r2_linear:.3f}. At 40% inventory, attacker sees {advantage_at_40pct:.1f}x more of the OT blast radius than the defender.")
        elif r2_quad > r2_linear:
            verdict = HypothesisVerdict.INCONCLUSIVE
            finding = f"Some evidence of super-linearity but not conclusive. Quadratic R²={r2_quad:.3f} vs linear R²={r2_linear:.3f}."
        else:
            verdict = HypothesisVerdict.FAILED
            finding = f"Linear scaling. Unknown assets add proportional risk. Linear R²={r2_linear:.3f}."
        return HypothesisResult(
            hypothesis_id=self.hypothesis_id, title=self.title, verdict=verdict,
            primary_metric="Attacker advantage ratio at Meridian baseline (40% OT inventory)",
            primary_value=advantage_at_40pct or 0.0, key_finding=finding,
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
            poly_coeffs = np.polyfit(x, y_acc, 3)
            a, b = poly_coeffs[0], poly_coeffs[1]
            inflection_x = float(np.clip(-b / (3 * a) if a != 0 else 0.5, 0, 1))
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
