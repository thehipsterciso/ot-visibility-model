"""
tests/test_core.py — Smoke tests for the OT Visibility Model.
"""

import pytest
import numpy as np

from src.assets.graph import MeridianGraph
from src.constants import ControlStrategy, Meridian, ThreatActor
from src.simulation.engine import SimulationEngine


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
                    "inventory_completeness", "direct_external_access_points",
                    "crown_jewels", "uninventoried_ot"]:
            assert key in summary


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
        baseline = SimulationEngine(inventory_completeness=0.40,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY, iterations=300, seed=42).run()
        full_inv = SimulationEngine(inventory_completeness=1.0,
            control_strategy=ControlStrategy.CHECKPOINT_ONLY, iterations=300, seed=42).run()
        assert full_inv.detection_rate() > baseline.detection_rate()

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
