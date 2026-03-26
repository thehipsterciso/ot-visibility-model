"""
simulation/engine.py — Monte Carlo simulation engine.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.assets.graph import MeridianGraph
from src.constants import (
    ControlStrategy, Meridian, SimConfig, ThreatActor, ThreatActorConfig
)

# NIST CSF Identify/Asset Management control remediation cost (market rate).
# Source: Marsh/KPMG OT Cybersecurity Assessment pricing 2023-2024.
# "Average cost to assess and remediate a single NIST CSF 2.0 ID.AM control gap:
# $60,000–$90,000." $75K is mid-range.
_NIST_REMEDIATION_COST_PER_CONTROL = 75_000


@dataclass
class IncidentResult:
    iteration: int
    threat_actor: ThreatActor
    entry_node: str
    inventory_completeness: float
    control_strategy: ControlStrategy
    attacker_blast_radius: int
    defender_visible_radius: int
    attacker_advantage: float
    crown_jewels_in_blast: int
    crown_jewels_visible_to_defender: int
    detected: bool
    detection_lead_time_hours: float
    mttd_hours: float
    response_accuracy: float
    mttr_hours: float
    response_delay_hours: float
    downtime_hours: float
    direct_loss: float
    response_cost: float
    regulatory_exposure: float
    insurance_recovery: float
    net_financial_impact: float
    compliance_gaps: int
    bypassed_it: bool
    used_unknown_path: bool
    # Graph-derived monitoring coverage — fraction of attacker-reachable OT nodes
    # that the defender can also see. Computed from blast-radius traversal, NOT
    # directly from the inventory_completeness scalar. This breaks the circularity
    # where detection formulas previously read inventory_completeness directly.
    monitoring_coverage_score: float = 0.0


@dataclass
class SimulationResults:
    scenario_id: str
    inventory_completeness: float
    control_strategy: ControlStrategy
    iterations: int
    incidents: list[IncidentResult] = field(default_factory=list)

    def detection_rate(self) -> float:
        if not self.incidents:
            return 0.0
        return sum(1 for i in self.incidents if i.detected) / len(self.incidents)

    def mean_net_impact(self) -> float:
        if not self.incidents:
            return 0.0
        return float(np.mean([i.net_financial_impact for i in self.incidents]))

    def mean_attacker_advantage(self) -> float:
        if not self.incidents:
            return 0.0
        return float(np.mean([i.attacker_advantage for i in self.incidents]))

    def mean_mttd(self) -> float:
        detected = [i for i in self.incidents if i.detected]
        if not detected:
            return float("inf")
        return float(np.mean([i.mttd_hours for i in detected]))

    def mean_response_accuracy(self) -> float:
        if not self.incidents:
            return 0.0
        return float(np.mean([i.response_accuracy for i in self.incidents]))

    def mean_monitoring_coverage(self) -> float:
        if not self.incidents:
            return 0.0
        return float(np.mean([i.monitoring_coverage_score for i in self.incidents]))

    def total_expected_annual_loss(self) -> float:
        """Expected annual loss: mean impact per incident × expected annual incident frequency."""
        if not self.incidents:
            return 0.0
        from src.constants import ThreatActorConfig
        annual_frequency = float(sum(ThreatActorConfig.ANNUAL_FREQUENCY.values()))
        return float(np.mean([i.net_financial_impact for i in self.incidents]) * annual_frequency)


class SimulationEngine:

    def __init__(
        self,
        inventory_completeness: float = 0.40,
        control_strategy: ControlStrategy = ControlStrategy.CHECKPOINT_ONLY,
        threat_mix: Optional[dict[ThreatActor, float]] = None,
        iterations: int = SimConfig.DEFAULT_ITERATIONS,
        seed: int = SimConfig.RANDOM_SEED,
        scenario_id: str = "S0",
    ):
        self.inventory_completeness = inventory_completeness
        self.control_strategy = control_strategy
        self.threat_mix = threat_mix
        self.iterations = iterations
        self.seed = seed
        self.scenario_id = scenario_id
        self.rng = np.random.default_rng(seed)
        self.graph = MeridianGraph(
            inventory_completeness=inventory_completeness,
            seed=seed,
        ).build()

    def run(self) -> SimulationResults:
        results = SimulationResults(
            scenario_id=self.scenario_id,
            inventory_completeness=self.inventory_completeness,
            control_strategy=self.control_strategy,
            iterations=self.iterations,
        )
        threat_actors = self._build_threat_actor_pool()
        for i in range(self.iterations):
            actor = self.rng.choice(threat_actors)
            result = self._simulate_incident(i, ThreatActor(actor))
            results.incidents.append(result)
        return results

    def _build_threat_actor_pool(self) -> list[str]:
        if self.threat_mix:
            weights = self.threat_mix
        else:
            total = sum(ThreatActorConfig.ANNUAL_FREQUENCY.values())
            weights = {
                ta: freq / total
                for ta, freq in ThreatActorConfig.ANNUAL_FREQUENCY.items()
            }
        pool = []
        for ta, weight in weights.items():
            pool.extend([ta.value] * max(1, int(weight * 100)))
        return pool

    def _simulate_incident(self, iteration: int, actor: ThreatActor) -> IncidentResult:
        entry_points = self.graph.get_entry_points(actor)
        if not entry_points:
            entry_points = list(self.graph.graph.nodes())
        entry_node = str(self.rng.choice(entry_points))

        attacker_br = self.graph.compute_blast_radius(entry_node, ot_only=True, inventoried_only=False)
        defender_br = self.graph.compute_blast_radius(entry_node, ot_only=True, inventoried_only=True)
        attacker_advantage = len(attacker_br) / max(1, len(defender_br))

        # Graph-derived monitoring coverage: fraction of attacker-reachable OT nodes
        # visible to the defender. Derived from topology, not directly from the
        # inventory_completeness scalar — breaks the circularity in detection formulas.
        monitoring_coverage_score = len(defender_br) / max(1, len(attacker_br))

        crown_jewels = set(self.graph.get_crown_jewels())
        cj_in_blast = len(attacker_br & crown_jewels)
        cj_visible = len(defender_br & crown_jewels)

        detected, mttd, detection_lead_time = self._compute_detection(
            actor, entry_node, monitoring_coverage_score
        )
        response_accuracy, mttr, response_delay = self._compute_response(
            detected, monitoring_coverage_score
        )
        used_unknown = self._attack_used_unknown_path(entry_node, attacker_br)
        bypassed_it = ThreatActorConfig.BYPASSES_IT[actor]

        compliance_gaps = self._compute_compliance_gaps()
        downtime, direct_loss, resp_cost, reg_exp, ins_rec, net_impact = \
            self._compute_financial_impact(
                actor, detected, mttd, mttr, response_accuracy, compliance_gaps,
                detection_lead_time=detection_lead_time,
            )

        return IncidentResult(
            iteration=iteration,
            threat_actor=actor,
            entry_node=entry_node,
            inventory_completeness=self.inventory_completeness,
            control_strategy=self.control_strategy,
            attacker_blast_radius=len(attacker_br),
            defender_visible_radius=len(defender_br),
            attacker_advantage=attacker_advantage,
            crown_jewels_in_blast=cj_in_blast,
            crown_jewels_visible_to_defender=cj_visible,
            detected=detected,
            detection_lead_time_hours=detection_lead_time,
            mttd_hours=mttd,
            response_accuracy=response_accuracy,
            mttr_hours=mttr,
            response_delay_hours=response_delay,
            downtime_hours=downtime,
            direct_loss=direct_loss,
            response_cost=resp_cost,
            regulatory_exposure=reg_exp,
            insurance_recovery=ins_rec,
            net_financial_impact=net_impact,
            compliance_gaps=compliance_gaps,
            bypassed_it=bypassed_it,
            used_unknown_path=used_unknown,
            monitoring_coverage_score=monitoring_coverage_score,
        )

    def _compute_detection(
        self, actor: ThreatActor, entry_node: str, monitoring_coverage_score: float
    ) -> tuple[bool, float, float]:
        """
        Compute detection probability and MTTD.

        Detection is driven by monitoring_coverage_score — the fraction of the
        attacker's blast radius visible to the defender, derived from graph traversal.
        This is NOT a direct alias for inventory_completeness; it reflects the
        actual coverage at the specific entry point and reachable topology.

        Base formula: network-perimeter monitoring (checkpoint) provides a floor
        of 0.40 coverage; each additional inventoried node in the blast radius
        raises coverage toward 1.0.
        """
        base_detection_prob = SimConfig.CHECKPOINT_DETECTION_MULTIPLIER[self.control_strategy]
        if ThreatActorConfig.BYPASSES_IT[actor]:
            # Insider/vendor with direct OT access bypasses IT checkpoints entirely —
            # detection relies solely on per-asset OT monitoring (host-based or
            # behavioral), which requires asset inventory.
            if self.control_strategy in (
                ControlStrategy.CHECKPOINT_ONLY,
                ControlStrategy.CHECKPOINT_OPTIMIZED
            ):
                base_detection_prob *= (monitoring_coverage_score * 0.60)
            else:
                base_detection_prob *= (monitoring_coverage_score * 0.85)
        # Monitoring coverage floor: checkpoint-only network monitoring provides
        # partial detection (0.40) even with zero OT inventory (network-level anomalies).
        # Full coverage (1.0) requires complete asset knowledge for behavioral baselining.
        monitoring_penalty = 0.40 + (0.60 * monitoring_coverage_score)
        detection_prob = min(0.99, max(0.01, base_detection_prob * monitoring_penalty))
        detected = bool(self.rng.random() < detection_prob)
        if ThreatActorConfig.BYPASSES_IT[actor]:
            base_mttd = Meridian.MTTD_OT_HOURS
        else:
            base_mttd = (Meridian.MTTD_IT_HOURS * 0.4) + (Meridian.MTTD_OT_HOURS * 0.6)
        strategy_mttd_multiplier = {
            ControlStrategy.CHECKPOINT_ONLY: 1.0,
            ControlStrategy.CHECKPOINT_OPTIMIZED: 0.85,
            ControlStrategy.INVENTORY_INFORMED: 0.55,
            ControlStrategy.FULL_MATURITY: 0.35,
        }
        mttd = base_mttd * strategy_mttd_multiplier[self.control_strategy]
        mttd *= float(self.rng.lognormal(0, 0.3))
        # Lead time = advance warning before maximum damage: how far ahead of worst-case
        # the attack was detected. Early detection (low mttd) → large positive lead time.
        if detected:
            detection_lead_time = max(0.0, Meridian.AVG_DOWNTIME_PER_MAJOR_INCIDENT - mttd)
        else:
            detection_lead_time = 0.0
        return detected, mttd, detection_lead_time

    def _compute_response(
        self, detected: bool, monitoring_coverage_score: float
    ) -> tuple[float, float, float]:
        """
        Compute response accuracy and MTTR.

        Response accuracy follows a sigmoid curve over monitoring_coverage_score.
        Below ~65% coverage, responders spend most of their time discovering the
        environment (discovery_hours dominates MTTR). Above 65%, containment
        activity dominates — a structural threshold, not an arbitrary formula.
        """
        if not detected:
            return 0.10, Meridian.MTTR_OT_HOURS * 4.0, Meridian.MTTR_OT_HOURS * 3.0
        midpoint = SimConfig.RESPONSE_TIME_SIGMOID_MIDPOINT
        steepness = SimConfig.RESPONSE_TIME_SIGMOID_STEEPNESS
        sigmoid_value = 1 / (1 + math.exp(-steepness * (monitoring_coverage_score - midpoint)))
        response_accuracy = float(min(0.98, max(0.15, 0.20 + (0.75 * sigmoid_value))))
        discovery_hours = (1 - monitoring_coverage_score) * Meridian.MTTR_OT_HOURS * 1.8
        mttr = (Meridian.MTTR_OT_HOURS + discovery_hours) * float(self.rng.lognormal(0, 0.25))
        return response_accuracy, mttr, discovery_hours

    def _compute_financial_impact(
        self, actor: ThreatActor, detected: bool, mttd: float, mttr: float,
        response_accuracy: float, compliance_gaps: int,
        detection_lead_time: float = 0.0,
    ) -> tuple[float, float, float, float, float, float]:
        impact_mult = ThreatActorConfig.IMPACT_MULTIPLIER[actor]
        if not detected:
            downtime = float(
                Meridian.AVG_DOWNTIME_PER_MAJOR_INCIDENT * impact_mult * self.rng.uniform(1.5, 3.5)
            )
        else:
            containment_factor = 0.3 + (0.7 * response_accuracy)
            # Detection lead time bonus: earlier detection limits the attacker's lateral
            # movement window, reducing effective blast radius at time of response.
            # Capped at 25% downtime reduction. Grounded in: each hour of advance
            # warning before the worst-case 47h window limits additional asset compromise.
            MAX_LEAD_TIME_BONUS = 0.25
            lead_time_bonus = min(
                MAX_LEAD_TIME_BONUS,
                detection_lead_time / max(1.0, float(Meridian.AVG_DOWNTIME_PER_MAJOR_INCIDENT))
            )
            downtime = float(
                (mttd + mttr) * impact_mult
                * (1 - containment_factor * 0.5)
                * (1 - lead_time_bonus)
                * self.rng.uniform(0.7, 1.4)
            )
        direct_loss = downtime * Meridian.REVENUE_PER_HOUR
        response_cost = mttr * Meridian.IR_FULLY_LOADED_RATE
        # Regulatory / audit remediation exposure: each compliance gap corresponds to
        # partial non-compliance with a NIST CSF 2.0 ID.AM control.
        # Cost per unfulfilled control: $75K market rate (Marsh/KPMG 2023-2024).
        # Scaled by threat actor impact multiplier (more severe incidents trigger
        # more extensive audit scrutiny).
        if compliance_gaps > 0:
            reg_exp = float(compliance_gaps * _NIST_REMEDIATION_COST_PER_CONTROL * impact_mult)
        else:
            reg_exp = 0.0
        total_loss = direct_loss + response_cost + reg_exp
        ins_recovery = max(
            0.0,
            (total_loss - Meridian.INSURANCE_DEDUCTIBLE) * float(self.rng.uniform(0.4, 0.75))
        )
        return downtime, direct_loss, response_cost, reg_exp, ins_recovery, total_loss - ins_recovery

    def _attack_used_unknown_path(self, entry_node: str, blast_radius: set[str]) -> bool:
        unknown_nodes = set(self.graph.get_uninventoried_nodes())
        return bool(blast_radius & unknown_nodes)

    def _compute_compliance_gaps(self) -> int:
        inventory_gap = 1.0 - self.inventory_completeness
        base_gaps = int(inventory_gap * 20)
        noise = int(self.rng.integers(-2, 3))
        return max(0, base_gaps + noise)


# ---------------------------------------------------------------------------
# Multi-seed runner — produces distributional results across N random topologies
# ---------------------------------------------------------------------------

@dataclass
class MultiSeedResult:
    """
    Aggregated statistics across N independent random seeds for a single scenario.

    Running across multiple seeds demonstrates that results are not artifacts of
    a single synthetic topology (seed=42). Each seed produces a different graph
    instantiation while preserving the Meridian asset counts and wiring rules.
    """
    inventory_completeness: float
    control_strategy: ControlStrategy
    n_seeds: int
    seeds: list[int]
    mean_detection_rate: float
    std_detection_rate: float
    ci_detection_rate: tuple[float, float]
    mean_attacker_advantage: float
    std_attacker_advantage: float
    ci_attacker_advantage: tuple[float, float]
    mean_monitoring_coverage: float
    std_monitoring_coverage: float
    per_seed_results: list[SimulationResults] = field(default_factory=list)


class MultiSeedRunner:
    """
    Run SimulationEngine across N independent random seeds and return aggregated statistics.

    Purpose: demonstrate that hypothesis verdicts are not artefacts of a single
    graph topology. Each seed produces a different MeridianGraph instantiation;
    the resulting distribution of key metrics is reported with 95% confidence intervals.
    """

    def __init__(
        self,
        inventory_completeness: float,
        control_strategy: ControlStrategy,
        n_seeds: int = 10,
        base_seed: int = SimConfig.RANDOM_SEED,
        iterations: int = SimConfig.DEFAULT_ITERATIONS,
        threat_mix: Optional[dict[ThreatActor, float]] = None,
        scenario_id_prefix: str = "MS",
    ):
        self.inventory_completeness = inventory_completeness
        self.control_strategy = control_strategy
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.iterations = iterations
        self.threat_mix = threat_mix
        self.scenario_id_prefix = scenario_id_prefix

    def run(self) -> MultiSeedResult:
        seeds = [self.base_seed + i for i in range(self.n_seeds)]
        per_seed: list[SimulationResults] = []
        for seed in seeds:
            engine = SimulationEngine(
                inventory_completeness=self.inventory_completeness,
                control_strategy=self.control_strategy,
                threat_mix=self.threat_mix,
                iterations=self.iterations,
                seed=seed,
                scenario_id=f"{self.scenario_id_prefix}_seed{seed}",
            )
            per_seed.append(engine.run())

        detection_rates = [r.detection_rate() for r in per_seed]
        advantages = [r.mean_attacker_advantage() for r in per_seed]
        coverages = [r.mean_monitoring_coverage() for r in per_seed]

        def _ci95(values: list[float]) -> tuple[float, float]:
            arr = np.array(values)
            se = float(np.std(arr, ddof=1) / max(1, np.sqrt(len(arr))))
            z = 1.96
            mean = float(np.mean(arr))
            return (mean - z * se, mean + z * se)

        return MultiSeedResult(
            inventory_completeness=self.inventory_completeness,
            control_strategy=self.control_strategy,
            n_seeds=self.n_seeds,
            seeds=seeds,
            mean_detection_rate=float(np.mean(detection_rates)),
            std_detection_rate=float(np.std(detection_rates, ddof=1)),
            ci_detection_rate=_ci95(detection_rates),
            mean_attacker_advantage=float(np.mean(advantages)),
            std_attacker_advantage=float(np.std(advantages, ddof=1)),
            ci_attacker_advantage=_ci95(advantages),
            mean_monitoring_coverage=float(np.mean(coverages)),
            std_monitoring_coverage=float(np.std(coverages, ddof=1)),
            per_seed_results=per_seed,
        )


# ---------------------------------------------------------------------------
# Parameter sensitivity context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def perturb_parameters(
    mttd_mult: float = 1.0,
    revenue_mult: float = 1.0,
    frequency_mult: float = 1.0,
):
    """
    Context manager that temporarily perturbs Meridian and ThreatActorConfig parameters
    for sensitivity analysis, then restores originals on exit.

    Note: REVENUE_AT_RISK_PER_INCIDENT is computed at class definition time and is NOT
    used in _compute_financial_impact (which uses REVENUE_PER_HOUR directly), so it
    does not need patching.
    """
    orig_mttd = Meridian.MTTD_OT_HOURS
    orig_mttd_it = Meridian.MTTD_IT_HOURS
    orig_rev = Meridian.REVENUE_PER_HOUR
    orig_freq = dict(ThreatActorConfig.ANNUAL_FREQUENCY)
    try:
        Meridian.MTTD_OT_HOURS = orig_mttd * mttd_mult
        Meridian.MTTD_IT_HOURS = orig_mttd_it * mttd_mult
        Meridian.REVENUE_PER_HOUR = orig_rev * revenue_mult
        ThreatActorConfig.ANNUAL_FREQUENCY = {
            k: v * frequency_mult for k, v in orig_freq.items()
        }
        yield
    finally:
        Meridian.MTTD_OT_HOURS = orig_mttd
        Meridian.MTTD_IT_HOURS = orig_mttd_it
        Meridian.REVENUE_PER_HOUR = orig_rev
        ThreatActorConfig.ANNUAL_FREQUENCY = orig_freq
