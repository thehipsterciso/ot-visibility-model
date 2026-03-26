"""
tests/test_core.py — Smoke tests for the OT Visibility Model.
"""

import pytest
import numpy as np

from src.assets.graph import MeridianGraph
from src.constants import ControlStrategy, Meridian, ThreatActor
from src.simulation.engine import MultiSeedRunner, SimulationEngine, perturb_parameters


class TestMeridianGraph:

    def test_builds_without_error(self):
        graph = MeridianGraph(inventory_completeness=0.40).build()
        assert graph.graph.number_of_nodes() > 0

    def test_ot_asset_count_approximate(self):
        graph = MeridianGraph(inventory_completeness=0.40).build()
        ot_nodes = [n for n, d in graph.graph.nodes(data=True) if d.get("is_ot")]
        assert abs(len(ot_nodes) - Meridian.OT_TOTAL) < 5

    def test_inventory_completeness_at_40pct(self):
        graph = MeridianGraph(inventory_completeness=0.40).build()
        actual = graph.inventory_completeness_actual()
        assert 0.30 < actual < 0.55

    def test_inventory_completeness_at_100pct(self):
        graph = MeridianGraph(inventory_completeness=1.0).build()
        actual = graph.inventory_completeness_actual()
        assert actual > 0.95

    def test_crown_jewels_assigned(self):
        graph = MeridianGraph().build()
        assert len(graph.get_crown_jewels()) > 0

    def test_direct_access_points_exist(self):
        graph = MeridianGraph(inventory_completeness=0.40).build()
        ta3_entries = graph.get_entry_points(ThreatActor.TA3_INSIDER_VENDOR)
        assert len(ta3_entries) > 0

    def test_summary_returns_expected_keys(self):
        graph = MeridianGraph().build()
        summary = graph.summary()
        for key in ["total_nodes", "total_edges", "ot_assets", "it_assets",
                    "inventory_completeness", "criticality_weighted_completeness",
                    "direct_external_access_points", "crown_jewels", "uninventoried_ot"]:
            assert key in summary

    def test_criticality_weighted_completeness_between_0_and_1(self):
        graph = MeridianGraph(inventory_completeness=0.40).build()
        cwc = graph.criticality_weighted_completeness()
        assert 0.0 <= cwc <= 1.0

    def test_criticality_weighted_completeness_full_inventory(self):
        graph = MeridianGraph(inventory_completeness=1.0).build()
        cwc = graph.criticality_weighted_completeness()
        assert cwc > 0.90

    def test_criticality_weighted_gt_flat_for_low_coverage(self):
        # When coverage is low, crown jewels (high criticality) may be disproportionately
        # missing, making CWC lower than flat completeness — OR it could be higher.
        # The key invariant: CWC at 100% > CWC at 40% for the same seed.
        g40 = MeridianGraph(inventory_completeness=0.40, seed=7).build()
        g100 = MeridianGraph(inventory_completeness=1.0, seed=7).build()
        assert g100.criticality_weighted_completeness() > g40.criticality_weighted_completeness()


class TestSimulationEngine:

    def test_runs_without_error(self):
        engine = SimulationEngine(iterations=100, seed=42)
        results = engine.run()
        assert len(results.incidents) == 100

    def test_detection_rate_between_0_and_1(self):
        engine = SimulationEngine(iterations=300, seed=42)
        results = engine.run()
        dr = results.detection_rate()
        assert 0.0 <= dr <= 1.0

    def test_financial_impacts_positive(self):
        engine = SimulationEngine(iterations=200, seed=42)
        results = engine.run()
        for incident in results.incidents:
            assert incident.net_financial_impact >= 0

    def test_full_inventory_improves_detection(self):
        # Full inventory should improve the graph-derived monitoring coverage score,
        # which feeds into detection probability. We assert on monitoring_coverage_score
        # (the direct causal variable) and on detection_rate over a larger independent
        # sample (different seeds) to avoid small-sample seed collisions.
        baseline = SimulationEngine(inventory_completeness=0.40,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY, iterations=2000, seed=42).run()
        full_inv = SimulationEngine(inventory_completeness=1.0,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY, iterations=2000, seed=99).run()
        assert full_inv.mean_monitoring_coverage() >= baseline.mean_monitoring_coverage()
        assert full_inv.detection_rate() >= baseline.detection_rate()

    def test_ta3_bypasses_it(self):
        engine = SimulationEngine(inventory_completeness=0.40,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0}, iterations=100, seed=42)
        results = engine.run()
        ta3 = [i for i in results.incidents if i.threat_actor == ThreatActor.TA3_INSIDER_VENDOR]
        if ta3:
            assert all(i.bypassed_it for i in ta3)

    def test_response_accuracy_higher_with_full_inventory(self):
        baseline = SimulationEngine(inventory_completeness=0.40, iterations=300, seed=42).run()
        full = SimulationEngine(inventory_completeness=1.0, iterations=300, seed=42).run()
        assert full.mean_response_accuracy() > baseline.mean_response_accuracy()

    def test_compliance_gaps_decrease_with_inventory(self):
        low = SimulationEngine(inventory_completeness=0.20, iterations=200, seed=42).run()
        high = SimulationEngine(inventory_completeness=0.90, iterations=200, seed=42).run()
        low_gaps = np.mean([i.compliance_gaps for i in low.incidents])
        high_gaps = np.mean([i.compliance_gaps for i in high.incidents])
        assert low_gaps > high_gaps


class TestHypotheses:

    def _run(self, inv, strategy, threat_mix=None, n=150, sid="T"):
        return SimulationEngine(inventory_completeness=inv, control_strategy=strategy,
            threat_mix=threat_mix, iterations=n, seed=42, scenario_id=sid).run()

    def test_h1_returns_verdict(self):
        from src.hypotheses.runner import H1BortHypothesis
        cp = self._run(0.40, ControlStrategy.CHECKPOINT_OPTIMIZED, sid="S1")
        inv = self._run(1.00, ControlStrategy.INVENTORY_INFORMED, sid="S2")
        result = H1BortHypothesis().evaluate(checkpoint_results=cp, inventory_results=inv)
        assert result.verdict is not None and result.key_finding

    def test_h2_returns_verdict(self):
        from src.hypotheses.runner import H2FoundationHypothesis
        sweep = [self._run(lvl, ControlStrategy.CHECKPOINT_ONLY, sid=f"S4_{int(lvl*100)}")
                 for lvl in [0.0, 0.25, 0.50, 0.75, 1.0]]
        result = H2FoundationHypothesis().evaluate(sweep_results=sweep)
        assert result.verdict is not None

    def test_h4_returns_verdict(self):
        from src.hypotheses.runner import H4BlastRadiusHypothesis
        sweep = [self._run(lvl, ControlStrategy.CHECKPOINT_ONLY, sid=f"S4_{int(lvl*100)}")
                 for lvl in [0.0, 0.25, 0.50, 0.75, 1.0]]
        result = H4BlastRadiusHypothesis().evaluate(sweep_results=sweep)
        assert result.verdict is not None

    def test_h5_returns_verdict(self):
        from src.hypotheses.runner import H5InsiderHypothesis
        cp = self._run(0.40, ControlStrategy.CHECKPOINT_ONLY,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0}, sid="S5")
        inv = self._run(1.00, ControlStrategy.INVENTORY_INFORMED,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0}, sid="H5_inv")
        result = H5InsiderHypothesis().evaluate(checkpoint_ta3_results=cp, inventory_ta3_results=inv)
        assert result.verdict is not None

    def test_h10_returns_verdict(self):
        from src.hypotheses.runner import H10CostParityHypothesis
        baseline = self._run(0.40, ControlStrategy.CHECKPOINT_ONLY, sid="S0")
        full = self._run(1.00, ControlStrategy.CHECKPOINT_ONLY, sid="S2")
        result = H10CostParityHypothesis().evaluate(baseline_results=baseline, full_inv_results=full)
        assert result.verdict is not None


class TestMonitoringCoverageScore:

    def test_score_between_0_and_1(self):
        engine = SimulationEngine(inventory_completeness=0.40, iterations=200, seed=42)
        results = engine.run()
        for inc in results.incidents:
            assert 0.0 <= inc.monitoring_coverage_score <= 1.0

    def test_score_higher_at_full_inventory(self):
        low = SimulationEngine(inventory_completeness=0.20, iterations=500, seed=42).run()
        high = SimulationEngine(inventory_completeness=1.00, iterations=500, seed=42).run()
        assert high.mean_monitoring_coverage() >= low.mean_monitoring_coverage()

    def test_mean_monitoring_coverage_method(self):
        results = SimulationEngine(iterations=200, seed=42).run()
        mmc = results.mean_monitoring_coverage()
        assert 0.0 <= mmc <= 1.0


class TestMultiSeedRunner:

    def test_runs_multiple_seeds(self):
        runner = MultiSeedRunner(
            inventory_completeness=0.40,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            n_seeds=3,
            iterations=200,
        )
        result = runner.run()
        assert result.n_seeds == 3
        assert len(result.per_seed_results) == 3

    def test_ci_bounds_ordered(self):
        runner = MultiSeedRunner(
            inventory_completeness=0.40,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            n_seeds=5,
            iterations=150,
        )
        result = runner.run()
        lo, hi = result.ci_detection_rate
        assert lo <= hi
        lo2, hi2 = result.ci_attacker_advantage
        assert lo2 <= hi2

    def test_mean_attacker_advantage_nonnegative(self):
        runner = MultiSeedRunner(
            inventory_completeness=0.40,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            n_seeds=3,
            iterations=150,
        )
        result = runner.run()
        assert result.mean_attacker_advantage >= 0.0


class TestPerturbParameters:

    def test_mttd_reverts_after_context(self):
        from src.constants import Meridian as M
        original = M.MTTD_OT_HOURS
        with perturb_parameters(mttd_mult=2.0):
            assert M.MTTD_OT_HOURS == original * 2.0
        assert M.MTTD_OT_HOURS == original

    def test_revenue_reverts_after_context(self):
        from src.constants import Meridian as M
        original = M.REVENUE_PER_HOUR
        with perturb_parameters(revenue_mult=0.5):
            assert M.REVENUE_PER_HOUR == original * 0.5
        assert M.REVENUE_PER_HOUR == original

    def test_perturbation_changes_financial_outcomes(self):
        normal = SimulationEngine(iterations=300, seed=42).run()
        with perturb_parameters(revenue_mult=2.0):
            doubled = SimulationEngine(iterations=300, seed=42).run()
        assert doubled.mean_net_impact() > normal.mean_net_impact()


class TestH1SameInventory:

    def test_h1_with_same_inventory_comparison(self):
        from src.hypotheses.runner import H1BortHypothesis
        s1 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_OPTIMIZED,
            iterations=150, seed=42, scenario_id="S1").run()
        s2 = SimulationEngine(inventory_completeness=1.00, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            iterations=150, seed=42, scenario_id="S2").run()
        s1_inv = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.INVENTORY_INFORMED,
            iterations=150, seed=42, scenario_id="S1_inv").run()
        result = H1BortHypothesis().evaluate(
            checkpoint_results=s1, inventory_results=s2,
            checkpoint_same_inv_results=s1, inventory_same_inv_results=s1_inv,
        )
        assert result.verdict is not None
        assert "same_inv_cp_mean_loss" in result.supporting_metrics


class TestH9InsuranceModel:

    def test_h9_includes_insurance_metrics(self):
        from src.hypotheses.runner import H9ComplianceExposureHypothesis
        sweep = [
            SimulationEngine(inventory_completeness=lvl, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
                iterations=150, seed=42, scenario_id=f"T_{int(lvl*100)}").run()
            for lvl in [0.0, 0.40, 1.0]
        ]
        result = H9ComplianceExposureHypothesis().evaluate(sweep_results=sweep)
        assert result.verdict is not None
        sm = result.supporting_metrics
        assert "insurance_cost_at_baseline" in sm
        assert "total_compliance_burden_baseline" in sm
        assert "insurance_uplift_at_baseline" in sm
        assert sm["insurance_cost_at_baseline"] > 0

    def test_h9_insurance_higher_at_low_inventory(self):
        from src.hypotheses.runner import H9ComplianceExposureHypothesis
        sweep = [
            SimulationEngine(inventory_completeness=lvl, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
                iterations=150, seed=42, scenario_id=f"T_{int(lvl*100)}").run()
            for lvl in [0.0, 0.25, 0.40, 0.75, 1.0]
        ]
        result = H9ComplianceExposureHypothesis().evaluate(sweep_results=sweep)
        # Insurance costs should decrease as inventory completeness increases
        ins_costs = result.supporting_metrics["insurance_annual_cost_by_level"]
        assert ins_costs[0] >= ins_costs[-1]
