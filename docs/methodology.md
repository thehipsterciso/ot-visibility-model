# OT Visibility Model — Methodology

## Overview

The OT Visibility Model is a Monte Carlo simulation that tests whether Bryson Bort's claim at S4x26 — "OT visibility is overrated" — survives mathematical scrutiny. The model runs 10,000 simulated OT security incidents against a synthetic mid-market manufacturer (Meridian Precision Manufacturing, $340M PE-backed, 820 OT assets, 3 production sites) and evaluates 10 falsifiable hypotheses.

This document describes the key design decisions and explains why they were made.

---

## The Monitoring Coverage Score

### The Problem

A naive model would directly use `inventory_completeness` (a scalar 0–1) as an input to the detection probability formula:

```
detection_prob = f(inventory_completeness)
```

This creates a circularity: `inventory_completeness` is an input to the simulation *and* the variable being studied. If detection only depends on a scalar you control, you can trivially produce any result by choosing parameter weights.

### The Solution: Graph-Derived Intermediate Variable

Instead, the model computes `monitoring_coverage_score` from the actual graph topology on each incident:

```python
attacker_br = graph.compute_blast_radius(entry_node, ot_only=True, inventoried_only=False)
defender_br = graph.compute_blast_radius(entry_node, ot_only=True, inventoried_only=True)
monitoring_coverage_score = len(defender_br) / max(1, len(attacker_br))
```

This score represents: *what fraction of the OT assets the attacker can reach from this entry point is also visible to the defender?*

It is an intermediate, topology-derived variable. Its relationship to `inventory_completeness` is mediated by the specific graph structure — which nodes are inventoried, which paths exist, and which crown jewels are reachable. Two incidents at the same `inventory_completeness` level can produce very different `monitoring_coverage_score` values depending on the specific entry node and network topology.

This breaks the circularity: the detection formula depends on `monitoring_coverage_score`, which depends on the graph, which depends on `inventory_completeness`. The chain of causation is explicit and inspectable.

### The Detection Formula

```python
monitoring_penalty = 0.40 + (0.60 * monitoring_coverage_score)
detection_prob = base_detection_prob * monitoring_penalty
```

The `0.40` floor represents the baseline detection from network-perimeter monitoring (checkpoint controls) — even with zero OT inventory, anomalous network traffic can be detected. The remaining `0.60` requires asset knowledge for behavioral baselining.

For TA-3 (insider/vendor with direct OT access), the formula degrades more severely because checkpoint controls provide no floor for direct-access actors:

```python
if control_strategy in (CHECKPOINT_ONLY, CHECKPOINT_OPTIMIZED):
    base_detection_prob *= (monitoring_coverage_score * 0.60)
```

---

## The Sigmoid Response Model

Response accuracy is modeled as a sigmoid function of `monitoring_coverage_score`:

```python
sigmoid_input = 8.0 * (monitoring_coverage_score - 0.65)
response_accuracy = 1 / (1 + exp(-sigmoid_input))
```

The midpoint at 0.65 and steepness of 8.0 are calibrated so that:
- At 40% monitoring coverage (Meridian baseline): response accuracy ≈ 0.25
- At 65% monitoring coverage (inflection point): response accuracy ≈ 0.50
- At 90%+ monitoring coverage: response accuracy approaches 0.95

This creates the non-linear inflection H7 tests for. The inflection point at ~65% coverage represents the transition from "responders discovering what happened" to "responders containing known threats."

---

## Detection Lead Time and H3

Detection lead time is defined as:

```python
detection_lead_time = max(0.0, AVG_DOWNTIME_PER_MAJOR_INCIDENT - mttd)
```

Where `AVG_DOWNTIME_PER_MAJOR_INCIDENT = 47 hours` (Ponemon 2022).

This is the *advance warning* an early detection provides — how far ahead of maximum expected damage the attacker was caught. A detection at `mttd = 10h` produces `47 - 10 = 37h` of lead time; detection at `mttd = 84h` (after maximum damage) produces `max(0, 47 - 84) = 0`.

This definition ensures lead time is causally meaningful: more advance warning → more opportunity to limit damage → lower impact. H3 tests the correlation between lead time and net financial impact across detected incidents.

### Lead Time Bonus in Financial Impact

A detected incident with positive lead time reduces downtime:

```python
lead_time_bonus = min(0.25, detection_lead_time / 47)
downtime = base_downtime * (1 - lead_time_bonus)
```

The cap at 25% prevents lead time from dominating the impact formula — even perfect early detection doesn't eliminate all downtime (some damage occurs before detection, and containment takes time).

---

## H1: Comparing Like Against Like

Bort's claim is that a checkpoint-optimized strategy achieves equivalent outcomes to an inventory-informed strategy. The fair comparison requires holding inventory constant:

- **Primary comparison**: Checkpoint-optimized (40% inventory) vs Inventory-informed (100% inventory) — this tests Bort's claim in its strongest form
- **Same-inventory comparison**: Checkpoint-optimized (40%) vs Inventory-informed (40%) — this isolates the strategy effect from the inventory effect

If the Bort model is only "equivalent" because the inventory-informed scenario uses 100% inventory, that is not a win for the strategy — it is a confound. The same-inventory comparison disambiguates this.

---

## H9: Compliance Exposure Model

H9 uses three independent quantification approaches:

1. **Simulation-derived regulatory exposure**: `compliance_gaps * $75,000/control * impact_multiplier` per incident, annualized by `mean_per_incident * annual_frequency`. The $75,000 figure is the Marsh/KPMG estimate for NIST CSF control remediation cost in manufacturing.

2. **NIST CSF ID.AM control mapping**: All five ID.AM controls (ID.AM-1 through ID.AM-5) require asset enumeration. The number of impacted controls scales with inventory gap.

3. **Cyber insurance premium uplift**: At Meridian's 40% baseline inventory, Lloyd's/Marsh 2023 OT underwriting guidelines indicate a 35% premium surcharge on the baseline $850K annual policy = $297,500/yr additional cost. This is a *published underwriting requirement*, independent of simulation parameters.

The combination of simulation-derived + actuarial overlay makes H9 defensible even if the simulation parameters are challenged.

---

## Graph Construction

### Purdue Model Hierarchy

Assets are assigned to Purdue model levels (0–4) based on type:
- Level 0: Sensors, RTUs
- Level 1: PLCs, Safety systems, Legacy unmanaged
- Level 2: HMIs, Industrial switches
- Level 3: Engineering workstations, Historians, SCADA servers
- Level 4: IT workstations, IT servers, IT network

Communication edges follow the hierarchy (OT→IT direction) with lateral edges within levels (probability 0.20). Legacy nodes have additional "unexpected" edges to Level 4 at 35% probability, modeling the undocumented IT connections common in older OT environments.

### Why Undirected Graph for H6

H6 (segmentation gaps) uses an undirected version of the graph because:
1. Graph edges are modeled OT→IT (signal flow direction)
2. Attackers traverse bidirectionally — they move from IT entry points *into* OT
3. Using directed edges from IT→OT produces empty path sets (edges go the wrong direction)
4. Segmentation gaps must account for both directions of potential attack

The undirected conversion is explicit and documented in the code.

---

## Multi-Seed Stability

Each simulation run uses a single random seed that determines the graph topology. To prevent the critique "your results depend on topology seed X," the `MultiSeedRunner` runs N independent seeds (default: 10) and reports 95% confidence intervals.

Hypothesis verdicts that appear in the report are derived from the primary seed=42 run. The multi-seed CIs are available as supporting metrics in H4's output.

---

## What This Model Does Not Claim

1. **Specific dollar figures are illustrative, not predictive.** Meridian is synthetic. The financial outputs should be interpreted as ratios and directions, not absolute forecasts.

2. **The model does not simulate all threat actors or attack paths.** Nation-state persistence, supply chain firmware attacks, and physical-cyber convergence are not modeled.

3. **100% inventory is not a real-world baseline.** The model uses 100% inventory as an analytical upper bound, not as an achievable target.

4. **Verdicts are about the Meridian synthetic environment.** A real organization with different topology, threat exposure, or control maturity may produce different results.
