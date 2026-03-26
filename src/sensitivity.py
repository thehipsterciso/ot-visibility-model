"""
sensitivity.py — Parameter sensitivity analysis for the OT Visibility Model.

Addresses the critique: "your results depend on your parameter assumptions."

ParameterSensitivityAnalyzer perturbs the three most influential parameters
(MTTD_OT_HOURS, ANNUAL_FREQUENCY, REVENUE_PER_HOUR) by ±20% and ±40% and
re-runs the hypothesis sweep at each perturbation level. Hypotheses whose
verdict is stable across all perturbations are "robust"; hypotheses that flip
or degrade to INCONCLUSIVE under perturbation are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.constants import ControlStrategy, SimConfig
from src.hypotheses.runner import (
    H1BortHypothesis,
    H2FoundationHypothesis,
    H3ActionabilityGapHypothesis,
    H4BlastRadiusHypothesis,
    H5InsiderHypothesis,
    H6SegmentationQualityHypothesis,
    H7ResponseInflectionHypothesis,
    H8RiskQuantificationHypothesis,
    H9ComplianceExposureHypothesis,
    H10CostParityHypothesis,
    HypothesisResult,
)
from src.simulation.engine import SimulationEngine, perturb_parameters
from src.assets.graph import MeridianGraph
from src.constants import ThreatActor


@dataclass
class PerturbationLevel:
    name: str                  # e.g. "mttd_+20pct"
    mttd_mult: float = 1.0
    revenue_mult: float = 1.0
    frequency_mult: float = 1.0


# The perturbation grid: ±20% and ±40% on the three key parameters
PERTURBATION_LEVELS: list[PerturbationLevel] = [
    PerturbationLevel("baseline"),
    PerturbationLevel("mttd_+20pct", mttd_mult=1.20),
    PerturbationLevel("mttd_-20pct", mttd_mult=0.80),
    PerturbationLevel("mttd_+40pct", mttd_mult=1.40),
    PerturbationLevel("mttd_-40pct", mttd_mult=0.60),
    PerturbationLevel("revenue_+20pct", revenue_mult=1.20),
    PerturbationLevel("revenue_-20pct", revenue_mult=0.80),
    PerturbationLevel("revenue_+40pct", revenue_mult=1.40),
    PerturbationLevel("revenue_-40pct", revenue_mult=0.60),
    PerturbationLevel("frequency_+20pct", frequency_mult=1.20),
    PerturbationLevel("frequency_-20pct", frequency_mult=0.80),
    PerturbationLevel("frequency_+40pct", frequency_mult=1.40),
    PerturbationLevel("frequency_-40pct", frequency_mult=0.60),
]


@dataclass
class SensitivityResult:
    """Results for a single hypothesis across all perturbation levels."""
    hypothesis_id: str
    title: str
    baseline_verdict: str
    verdicts_by_perturbation: dict[str, str] = field(default_factory=dict)
    primary_values_by_perturbation: dict[str, float] = field(default_factory=dict)

    def is_stable(self) -> bool:
        """True if verdict is identical across all perturbation levels."""
        return all(v == self.baseline_verdict for v in self.verdicts_by_perturbation.values())

    def instability_count(self) -> int:
        return sum(1 for v in self.verdicts_by_perturbation.values() if v != self.baseline_verdict)

    def primary_value_cv(self) -> float:
        """Coefficient of variation of the primary metric across perturbations."""
        vals = list(self.primary_values_by_perturbation.values())
        if not vals:
            return 0.0
        mean = float(np.mean(vals))
        if mean == 0:
            return 0.0
        return float(np.std(vals)) / abs(mean)


@dataclass
class SensitivityReport:
    """Aggregated sensitivity analysis across all 10 hypotheses."""
    results: list[SensitivityResult] = field(default_factory=list)
    perturbation_levels_tested: int = 0

    def stable_verdicts(self) -> list[SensitivityResult]:
        """Hypotheses whose verdict holds across all perturbations."""
        return [r for r in self.results if r.is_stable()]

    def unstable_verdicts(self) -> list[SensitivityResult]:
        """Hypotheses that change verdict under at least one perturbation."""
        return [r for r in self.results if not r.is_stable()]

    def summary_table(self) -> list[dict]:
        rows = []
        for r in self.results:
            rows.append({
                "hypothesis_id": r.hypothesis_id,
                "baseline_verdict": r.baseline_verdict,
                "stable": r.is_stable(),
                "instability_count": r.instability_count(),
                "primary_value_cv": round(r.primary_value_cv(), 3),
            })
        return rows


class ParameterSensitivityAnalyzer:
    """
    Runs the full hypothesis suite at each perturbation level and reports
    which verdicts are robust vs parameter-sensitive.

    Designed for defensive use: run this before presenting results to an
    adversarial audience. Any UNSTABLE verdict should either be explained
    (genuine sensitivity) or investigated (model design issue).
    """

    def __init__(
        self,
        sweep_levels: Optional[list[float]] = None,
        iterations: int = 2000,
        seed: int = SimConfig.RANDOM_SEED,
    ):
        self.sweep_levels = sweep_levels or [0.0, 0.25, 0.40, 0.60, 0.75, 0.90, 1.0]
        self.iterations = iterations
        self.seed = seed

    def _run_hypothesis_suite(self) -> list[HypothesisResult]:
        """Run all 10 hypotheses at the current parameter state (caller applies perturbation)."""
        sweep_results = [
            SimulationEngine(
                inventory_completeness=level,
                control_strategy=ControlStrategy.CHECKPOINT_ONLY,
                iterations=self.iterations,
                seed=self.seed,
                scenario_id=f"SA_{int(level * 100)}",
            ).run()
            for level in self.sweep_levels
        ]

        s0 = SimulationEngine(
            inventory_completeness=0.40,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            iterations=self.iterations,
            seed=self.seed,
            scenario_id="SA_S0",
        ).run()
        s1 = SimulationEngine(
            inventory_completeness=0.40,
            control_strategy=ControlStrategy.CHECKPOINT_OPTIMIZED,
            iterations=self.iterations,
            seed=self.seed,
            scenario_id="SA_S1",
        ).run()
        s2 = SimulationEngine(
            inventory_completeness=1.00,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            iterations=self.iterations,
            seed=self.seed,
            scenario_id="SA_S2",
        ).run()
        s5 = SimulationEngine(
            inventory_completeness=0.40,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0},
            iterations=self.iterations,
            seed=self.seed,
            scenario_id="SA_S5",
        ).run()
        h5_inv = SimulationEngine(
            inventory_completeness=1.00,
            control_strategy=ControlStrategy.INVENTORY_INFORMED,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0},
            iterations=self.iterations,
            seed=self.seed,
            scenario_id="SA_H5inv",
        ).run()
        baseline_graph = MeridianGraph(inventory_completeness=0.40, seed=self.seed).build()

        return [
            H1BortHypothesis().evaluate(checkpoint_results=s1, inventory_results=s2),
            H2FoundationHypothesis().evaluate(sweep_results=sweep_results),
            H3ActionabilityGapHypothesis().evaluate(sweep_results=sweep_results),
            H4BlastRadiusHypothesis().evaluate(sweep_results=sweep_results),
            H5InsiderHypothesis().evaluate(checkpoint_ta3_results=s5, inventory_ta3_results=h5_inv),
            H6SegmentationQualityHypothesis().evaluate(sweep_results=sweep_results, graph=baseline_graph),
            H7ResponseInflectionHypothesis().evaluate(sweep_results=sweep_results),
            H8RiskQuantificationHypothesis().evaluate(sweep_results=sweep_results),
            H9ComplianceExposureHypothesis().evaluate(sweep_results=sweep_results),
            H10CostParityHypothesis().evaluate(baseline_results=s0, full_inv_results=s2),
        ]

    def run(
        self,
        perturbation_levels: Optional[list[PerturbationLevel]] = None,
        progress_callback=None,
    ) -> SensitivityReport:
        """
        Run the hypothesis suite at each perturbation level.

        Args:
            perturbation_levels: List of PerturbationLevel to test. Defaults to
                PERTURBATION_LEVELS (±20%, ±40% on MTTD, revenue, frequency).
            progress_callback: Optional callable(label: str) called before each level.
        """
        levels = perturbation_levels or PERTURBATION_LEVELS
        report = SensitivityReport(perturbation_levels_tested=len(levels))

        # Run baseline first to get hypothesis IDs and titles
        if progress_callback:
            progress_callback("baseline")
        baseline_results = self._run_hypothesis_suite()

        # Initialise SensitivityResult for each hypothesis
        sensitivity_by_id: dict[str, SensitivityResult] = {}
        for r in baseline_results:
            sensitivity_by_id[r.hypothesis_id] = SensitivityResult(
                hypothesis_id=r.hypothesis_id,
                title=r.title,
                baseline_verdict=r.verdict.value,
                verdicts_by_perturbation={"baseline": r.verdict.value},
                primary_values_by_perturbation={"baseline": r.primary_value or 0.0},
            )

        # Run each non-baseline perturbation level
        for level in levels:
            if level.name == "baseline":
                continue
            if progress_callback:
                progress_callback(level.name)
            with perturb_parameters(
                mttd_mult=level.mttd_mult,
                revenue_mult=level.revenue_mult,
                frequency_mult=level.frequency_mult,
            ):
                perturbed_results = self._run_hypothesis_suite()

            for r in perturbed_results:
                if r.hypothesis_id in sensitivity_by_id:
                    sensitivity_by_id[r.hypothesis_id].verdicts_by_perturbation[level.name] = r.verdict.value
                    sensitivity_by_id[r.hypothesis_id].primary_values_by_perturbation[level.name] = r.primary_value or 0.0

        report.results = list(sensitivity_by_id.values())
        return report
