"""
assets/graph.py — Synthetic asset graph for Meridian Precision Manufacturing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from src.constants import (
    AssetAge, AssetType, Meridian, PurdueLevel, SimConfig, ThreatActorConfig, ThreatActor
)


@dataclass
class Asset:
    asset_id: str
    asset_type: AssetType
    purdue_level: PurdueLevel
    age: AssetAge
    is_inventoried: bool
    criticality: float
    vulnerability_count: int
    site: int
    is_crown_jewel: bool = False
    has_direct_external_access: bool = False


@dataclass
class MeridianGraph:
    inventory_completeness: Optional[float] = None
    seed: int = SimConfig.RANDOM_SEED
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    assets: dict[str, Asset] = field(default_factory=dict)

    def build(self) -> "MeridianGraph":
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)
        self._generate_ot_assets(rng, np_rng)
        self._generate_it_assets(rng, np_rng)
        self._wire_communication_paths(rng)
        self._assign_crown_jewels()
        self._assign_direct_external_access(rng)
        return self

    def _generate_ot_assets(self, rng: random.Random, np_rng: np.random.Generator):
        purdue_map = {
            AssetType.SENSOR_RTU: PurdueLevel.LEVEL_0,
            AssetType.PLC: PurdueLevel.LEVEL_1,
            AssetType.SAFETY_SYSTEM: PurdueLevel.LEVEL_1,
            AssetType.HMI: PurdueLevel.LEVEL_2,
            AssetType.ENGINEERING_WORKSTATION: PurdueLevel.LEVEL_3,
            AssetType.HISTORIAN: PurdueLevel.LEVEL_3,
            AssetType.SCADA_SERVER: PurdueLevel.LEVEL_3,
            AssetType.INDUSTRIAL_SWITCH: PurdueLevel.LEVEL_2,
            AssetType.LEGACY_UNMANAGED: PurdueLevel.LEVEL_1,
        }
        idx = 0
        for asset_type, count in Meridian.OT_ASSET_COUNTS.items():
            baseline_inv = Meridian.BASELINE_INVENTORY_BY_TYPE[asset_type]
            inv_rate = self.inventory_completeness if self.inventory_completeness is not None \
                else baseline_inv
            for i in range(count):
                age = rng.choices(
                    list(Meridian.AGE_DISTRIBUTION.keys()),
                    weights=list(Meridian.AGE_DISTRIBUTION.values())
                )[0]
                mean_vulns = Meridian.VULNS_BY_AGE[age]
                vuln_count = max(0, int(np_rng.poisson(mean_vulns)))
                criticality = Meridian.CRITICALITY_BY_TYPE[asset_type]
                criticality += rng.uniform(-0.10, 0.10)
                criticality = max(0.0, min(1.0, criticality))
                site = rng.randint(1, Meridian.SITES)
                asset_id = f"OT-{asset_type.value.upper()[:4]}-{idx:04d}"
                asset = Asset(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    purdue_level=purdue_map[asset_type],
                    age=age,
                    is_inventoried=rng.random() < inv_rate,
                    criticality=criticality,
                    vulnerability_count=vuln_count,
                    site=site,
                )
                self.assets[asset_id] = asset
                self.graph.add_node(
                    asset_id,
                    asset_type=asset_type.value,
                    purdue_level=purdue_map[asset_type].value,
                    is_ot=True,
                    is_it=False,
                    is_inventoried=asset.is_inventoried,
                    criticality=criticality,
                    vulnerability_count=vuln_count,
                    site=site,
                    age=age.value,
                )
                idx += 1

    def _generate_it_assets(self, rng: random.Random, np_rng: np.random.Generator):
        it_counts = {
            AssetType.IT_WORKSTATION: 200,
            AssetType.IT_SERVER: 100,
            AssetType.IT_NETWORK: 40,
        }
        for asset_type, count in it_counts.items():
            for i in range(count):
                asset_id = f"IT-{asset_type.value.upper()[:4]}-{i:04d}"
                age = rng.choices(
                    list(Meridian.AGE_DISTRIBUTION.keys()),
                    weights=list(Meridian.AGE_DISTRIBUTION.values())
                )[0]
                vuln_count = max(0, int(np_rng.poisson(Meridian.VULNS_BY_AGE[age])))
                criticality = rng.uniform(0.3, 0.8)
                site = rng.randint(1, Meridian.SITES)
                asset = Asset(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    purdue_level=PurdueLevel.LEVEL_4,
                    age=age,
                    is_inventoried=True,
                    criticality=criticality,
                    vulnerability_count=vuln_count,
                    site=site,
                )
                self.assets[asset_id] = asset
                self.graph.add_node(
                    asset_id,
                    asset_type=asset_type.value,
                    purdue_level=PurdueLevel.LEVEL_4.value,
                    is_ot=False,
                    is_it=True,
                    is_inventoried=True,
                    criticality=criticality,
                    vulnerability_count=vuln_count,
                    site=site,
                    age=age.value,
                )

    def _wire_communication_paths(self, rng: random.Random):
        nodes_by_level: dict[int, list[str]] = {}
        for nid, data in self.graph.nodes(data=True):
            lvl = data["purdue_level"]
            nodes_by_level.setdefault(lvl, []).append(nid)
        for nid, data in self.graph.nodes(data=True):
            lvl = data["purdue_level"]
            upper_level = nodes_by_level.get(lvl + 1, [])
            if upper_level:
                supervisor = rng.choice(upper_level)
                self.graph.add_edge(nid, supervisor, monitored=False, authenticated=True)
        for level_nodes in nodes_by_level.values():
            if len(level_nodes) < 2:
                continue
            sample_size = min(len(level_nodes), 15)
            sample = rng.sample(level_nodes, sample_size)
            for i, src in enumerate(sample):
                for dst in sample[i+1:]:
                    if rng.random() < 0.20:
                        self.graph.add_edge(src, dst, monitored=False, authenticated=True)
        legacy_nodes = [
            nid for nid, d in self.graph.nodes(data=True)
            if d.get("age") == AssetAge.LEGACY.value or
               d.get("asset_type") == AssetType.LEGACY_UNMANAGED.value
        ]
        for nid in legacy_nodes:
            if rng.random() < 0.35:
                target_level_nodes = nodes_by_level.get(PurdueLevel.LEVEL_4.value, [])
                if target_level_nodes:
                    dst = rng.choice(target_level_nodes)
                    self.graph.add_edge(nid, dst, monitored=False, authenticated=False,
                                        unexpected=True)
        historian_nodes = [
            nid for nid, d in self.graph.nodes(data=True)
            if d.get("asset_type") == AssetType.HISTORIAN.value
        ]
        erp_nodes = [
            nid for nid, d in self.graph.nodes(data=True)
            if d.get("asset_type") == AssetType.IT_SERVER.value
        ]
        if historian_nodes and erp_nodes:
            for h in historian_nodes:
                erp = rng.choice(erp_nodes)
                self.graph.add_edge(h, erp, monitored=True, authenticated=True,
                                    is_historian_erp=True)

    def _assign_crown_jewels(self):
        crown_jewel_types = {
            AssetType.SAFETY_SYSTEM.value,
            AssetType.SCADA_SERVER.value,
            AssetType.HISTORIAN.value,
        }
        for nid, data in self.graph.nodes(data=True):
            if data.get("asset_type") in crown_jewel_types:
                self.graph.nodes[nid]["is_crown_jewel"] = True
                self.assets[nid].is_crown_jewel = True
            else:
                self.graph.nodes[nid]["is_crown_jewel"] = False

    def _assign_direct_external_access(self, rng: random.Random):
        ot_nodes = [nid for nid, d in self.graph.nodes(data=True) if d.get("is_ot")]
        for nid in ot_nodes:
            has_direct = rng.random() < 0.08
            self.graph.nodes[nid]["has_direct_external_access"] = has_direct
            self.assets[nid].has_direct_external_access = has_direct

    def get_entry_points(self, threat_actor: ThreatActor) -> list[str]:
        entry_level = ThreatActorConfig.ENTRY_LEVEL[threat_actor]
        if ThreatActorConfig.BYPASSES_IT[threat_actor]:
            return [
                nid for nid, d in self.graph.nodes(data=True)
                if d.get("has_direct_external_access", False)
            ]
        else:
            return [
                nid for nid, d in self.graph.nodes(data=True)
                if d.get("purdue_level", 0) == entry_level.value
            ]

    def compute_blast_radius(
        self,
        entry_node: str,
        hops: int = SimConfig.DEFAULT_BLAST_RADIUS_HOPS,
        inventoried_only: bool = False,
        ot_only: bool = False,
    ) -> set[str]:
        reachable = set()
        for node in nx.single_source_shortest_path_length(
            self.graph, entry_node, cutoff=hops
        ).keys():
            if node == entry_node:
                continue
            if ot_only and not self.graph.nodes[node].get("is_ot", False):
                continue
            if inventoried_only and not self.graph.nodes[node].get("is_inventoried", False):
                continue
            reachable.add(node)
        return reachable

    def get_crown_jewels(self) -> list[str]:
        return [nid for nid, d in self.graph.nodes(data=True) if d.get("is_crown_jewel", False)]

    def get_uninventoried_nodes(self) -> list[str]:
        return [
            nid for nid, d in self.graph.nodes(data=True)
            if not d.get("is_inventoried", True) and d.get("is_ot", False)
        ]

    def inventory_completeness_actual(self) -> float:
        ot_nodes = [nid for nid, d in self.graph.nodes(data=True) if d.get("is_ot")]
        if not ot_nodes:
            return 0.0
        inventoried = sum(
            1 for nid in ot_nodes if self.graph.nodes[nid].get("is_inventoried", False)
        )
        return inventoried / len(ot_nodes)

    def criticality_weighted_completeness(self) -> float:
        """
        Weighted inventory completeness where each OT asset's contribution is
        proportional to its criticality score. A high-criticality asset that is
        uninventoried degrades this metric more than a low-criticality sensor.

        This is the operationally relevant completeness measure — a site with
        100% SENSOR_RTU inventory but 0% SAFETY_SYSTEM inventory is not well-covered.
        """
        ot_nodes = [(nid, d) for nid, d in self.graph.nodes(data=True) if d.get("is_ot")]
        if not ot_nodes:
            return 0.0
        total_weight = sum(d.get("criticality", 0.5) for _, d in ot_nodes)
        if total_weight == 0:
            return 0.0
        inventoried_weight = sum(
            d.get("criticality", 0.5)
            for _, d in ot_nodes
            if d.get("is_inventoried", False)
        )
        return inventoried_weight / total_weight

    def summary(self) -> dict:
        ot_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("is_ot"))
        it_count = sum(1 for _, d in self.graph.nodes(data=True) if d.get("is_it"))
        direct_access = sum(
            1 for _, d in self.graph.nodes(data=True)
            if d.get("has_direct_external_access", False)
        )
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "ot_assets": ot_count,
            "it_assets": it_count,
            "inventory_completeness": round(self.inventory_completeness_actual(), 3),
            "criticality_weighted_completeness": round(self.criticality_weighted_completeness(), 3),
            "direct_external_access_points": direct_access,
            "crown_jewels": len(self.get_crown_jewels()),
            "uninventoried_ot": len(self.get_uninventoried_nodes()),
        }
