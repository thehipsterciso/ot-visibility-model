"""
constants.py — Shared enums, types, and Meridian configuration values.
All simulation parameters flow from here. Change here, changes everywhere.
"""

from enum import Enum
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AssetType(str, Enum):
    PLC = "plc"
    HMI = "hmi"
    ENGINEERING_WORKSTATION = "engineering_workstation"
    HISTORIAN = "historian"
    SCADA_SERVER = "scada_server"
    INDUSTRIAL_SWITCH = "industrial_switch"
    SENSOR_RTU = "sensor_rtu"
    SAFETY_SYSTEM = "safety_system"
    LEGACY_UNMANAGED = "legacy_unmanaged"
    IT_WORKSTATION = "it_workstation"
    IT_SERVER = "it_server"
    IT_NETWORK = "it_network"


class AssetAge(str, Enum):
    MODERN = "modern"
    MID_LIFE = "mid_life"
    LEGACY = "legacy"


class PurdueLevel(int, Enum):
    LEVEL_0 = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_35 = 35
    LEVEL_4 = 4
    LEVEL_5 = 5


class ThreatActor(str, Enum):
    TA1_NATION_STATE = "ta1_nation_state"
    TA2_RANSOMWARE = "ta2_ransomware"
    TA3_INSIDER_VENDOR = "ta3_insider_vendor"


class ControlStrategy(str, Enum):
    CHECKPOINT_ONLY = "checkpoint_only"
    CHECKPOINT_OPTIMIZED = "checkpoint_optimized"
    INVENTORY_INFORMED = "inventory_informed"
    FULL_MATURITY = "full_maturity"


class HypothesisVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"


class Meridian:
    NAME = "Meridian Precision Manufacturing"
    INDUSTRY = "Discrete Manufacturing"
    REVENUE_ANNUAL = 340_000_000
    EMPLOYEES = 1_200
    SITES = 3
    HOLD_YEARS = 3
    REVENUE_PER_HOUR = 38_750
    AVG_DOWNTIME_PER_MAJOR_INCIDENT = 47
    REVENUE_AT_RISK_PER_INCIDENT = REVENUE_PER_HOUR * AVG_DOWNTIME_PER_MAJOR_INCIDENT
    REGULATORY_EXPOSURE_MIN = 250_000
    REGULATORY_EXPOSURE_MAX = 2_000_000
    INSURANCE_DEDUCTIBLE = 500_000
    IR_FULLY_LOADED_RATE = 350
    MTTD_IT_HOURS = 6.2
    MTTD_OT_HOURS = 84.0
    MTTR_IT_HOURS = 4.1
    MTTR_OT_HOURS = 31.0
    OT_ASSET_COUNTS = {
        AssetType.PLC: 180,
        AssetType.HMI: 95,
        AssetType.ENGINEERING_WORKSTATION: 42,
        AssetType.HISTORIAN: 12,
        AssetType.SCADA_SERVER: 8,
        AssetType.INDUSTRIAL_SWITCH: 67,
        AssetType.SENSOR_RTU: 310,
        AssetType.SAFETY_SYSTEM: 22,
        AssetType.LEGACY_UNMANAGED: 84,
    }
    OT_TOTAL = sum(OT_ASSET_COUNTS.values())
    BASELINE_INVENTORY_BY_TYPE = {
        AssetType.PLC: 0.35,
        AssetType.HMI: 0.55,
        AssetType.ENGINEERING_WORKSTATION: 0.90,
        AssetType.HISTORIAN: 1.00,
        AssetType.SCADA_SERVER: 1.00,
        AssetType.INDUSTRIAL_SWITCH: 0.40,
        AssetType.SENSOR_RTU: 0.15,
        AssetType.SAFETY_SYSTEM: 0.70,
        AssetType.LEGACY_UNMANAGED: 0.00,
    }
    IT_TOTAL = 340
    AGE_DISTRIBUTION = {
        AssetAge.MODERN: 0.15,
        AssetAge.MID_LIFE: 0.60,
        AssetAge.LEGACY: 0.25,
    }
    VULNS_BY_AGE = {
        AssetAge.MODERN: 1.2,
        AssetAge.MID_LIFE: 3.8,
        AssetAge.LEGACY: 8.4,
    }
    CRITICALITY_BY_TYPE = {
        AssetType.PLC: 0.85,
        AssetType.HMI: 0.60,
        AssetType.ENGINEERING_WORKSTATION: 0.55,
        AssetType.HISTORIAN: 0.70,
        AssetType.SCADA_SERVER: 0.90,
        AssetType.INDUSTRIAL_SWITCH: 0.65,
        AssetType.SENSOR_RTU: 0.40,
        AssetType.SAFETY_SYSTEM: 0.95,
        AssetType.LEGACY_UNMANAGED: 0.50,
    }


class ThreatActorConfig:
    ANNUAL_FREQUENCY = {
        ThreatActor.TA1_NATION_STATE: 0.3,
        ThreatActor.TA2_RANSOMWARE: 2.1,
        ThreatActor.TA3_INSIDER_VENDOR: 0.8,
    }
    BYPASSES_IT = {
        ThreatActor.TA1_NATION_STATE: False,
        ThreatActor.TA2_RANSOMWARE: False,
        ThreatActor.TA3_INSIDER_VENDOR: True,
    }
    ENTRY_LEVEL = {
        ThreatActor.TA1_NATION_STATE: PurdueLevel.LEVEL_4,
        ThreatActor.TA2_RANSOMWARE: PurdueLevel.LEVEL_4,
        ThreatActor.TA3_INSIDER_VENDOR: PurdueLevel.LEVEL_2,
    }
    IMPACT_MULTIPLIER = {
        ThreatActor.TA1_NATION_STATE: 2.5,
        ThreatActor.TA2_RANSOMWARE: 1.8,
        ThreatActor.TA3_INSIDER_VENDOR: 1.3,
    }


class SimConfig:
    DEFAULT_ITERATIONS = 10_000
    DEFAULT_BLAST_RADIUS_HOPS = 3
    RANDOM_SEED = 42
    CONFIDENCE_INTERVAL = 0.95
    CHECKPOINT_DETECTION_MULTIPLIER = {
        ControlStrategy.CHECKPOINT_ONLY: 0.65,
        ControlStrategy.CHECKPOINT_OPTIMIZED: 0.78,
        ControlStrategy.INVENTORY_INFORMED: 0.85,
        ControlStrategy.FULL_MATURITY: 0.95,
    }
    RESPONSE_TIME_SIGMOID_MIDPOINT = 0.65
    RESPONSE_TIME_SIGMOID_STEEPNESS = 8.0
    SCENARIOS = {
        "S0": {"inventory": 0.40, "strategy": ControlStrategy.CHECKPOINT_ONLY,
               "threat_mix": None, "label": "Meridian Baseline"},
        "S1": {"inventory": 0.40, "strategy": ControlStrategy.CHECKPOINT_OPTIMIZED,
               "threat_mix": None, "label": "Bort Model"},
        "S2": {"inventory": 1.00, "strategy": ControlStrategy.CHECKPOINT_ONLY,
               "threat_mix": None, "label": "Inventory-First"},
        "S3": {"inventory": 1.00, "strategy": ControlStrategy.FULL_MATURITY,
               "threat_mix": None, "label": "Full Maturity"},
        "S4": {"inventory": "sweep", "strategy": ControlStrategy.CHECKPOINT_ONLY,
               "threat_mix": None, "label": "Inventory Sweep"},
        "S5": {"inventory": 0.40, "strategy": ControlStrategy.CHECKPOINT_ONLY,
               "threat_mix": {ThreatActor.TA3_INSIDER_VENDOR: 1.0},
               "label": "Insider/Vendor Only"},
        "S6": {"inventory": "variable", "strategy": "variable",
               "threat_mix": None, "label": "Cost Model"},
    }
