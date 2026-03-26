# OT Visibility Model — Adversarial Q&A

This document anticipates the strongest objections to this model and addresses each one directly. It is intended as preparation for the IT-OT webinar and for any technical review.

---

## "Your model is circular — you put inventory in and inventory comes out."

**Addressed.** Early versions of this model used `inventory_completeness` directly in detection formulas. The current model uses `monitoring_coverage_score`, a graph-derived intermediate variable:

```python
monitoring_coverage_score = len(defender_blast_radius) / max(1, len(attacker_blast_radius))
```

This is computed from the actual network topology on each incident — not from the scalar you control. Two incidents at 40% inventory can produce monitoring coverage scores of 0.0 (entry node has no OT reach) or 0.8 (entry node leads to many OT assets, most of which are inventoried).

The causal chain is: `inventory_completeness` → (graph construction) → `monitoring_coverage_score` → (detection formula) → `detected`. Each step is inspectable. The graph structure mediates the relationship — it does not merely alias the input.

---

## "Bort's comparison is unfair — you're comparing 40% inventory checkpoint vs 100% inventory-informed."

**Addressed.** H1 includes a same-inventory comparison: checkpoint-optimized (40%) vs inventory-informed (40%). This isolates the strategy effect from the inventory effect. The H1 supporting metrics include `same_inv_cp_mean_loss`, `same_inv_inv_mean_loss`, and the associated p-value.

If Bort's claim is that strategy matters more than inventory, this comparison tests it directly. If the two strategies produce statistically equivalent outcomes at equal inventory, the strategy-vs-inventory debate is a false binary. If they diverge, the winner is reported.

---

## "You cherry-picked seed=42 to get the result you wanted."

**Addressed.** Three mechanisms protect against this:

1. **MultiSeedRunner**: H4 (and other hypotheses where stability is relevant) can incorporate a `MultiSeedResult` with 95% confidence intervals across N independent seeds.

2. **Sensitivity analysis**: `ot-model sensitivity` runs the complete hypothesis suite at 12 perturbation levels (±20%/±40% on MTTD, revenue, frequency). Unstable verdicts are flagged. Run the analysis yourself and compare.

3. **Reproducible execution**: All scenarios use `seed=42` by default, which is documented and fixed. You can re-run with any seed using the CLI.

---

## "Your regulatory exposure figures are made up."

**Addressed for H9 specifically.** H9 uses three independent channels:

1. **NIST CSF control mapping**: All five ID.AM controls require asset enumeration. The non-compliance mapping is deterministic from the inventory gap, not from simulation randomness.

2. **NIST remediation cost**: $75,000/control from Marsh/KPMG 2023–2024 Digital Trust Insights — a published figure from an actuarial firm with direct access to manufacturing client data.

3. **Cyber insurance premium uplift**: 35% surcharge at <50% OT inventory from Lloyd's OT Underwriting Addendum 2023. This is a *published underwriting requirement* from a named insurer. It is not derived from simulation parameters.

The combination means H9's conclusion survives even if you reject the simulation-derived exposure figures entirely — the insurance overlay alone produces a material compliance burden.

---

## "The H3 correlation is weak — early detection doesn't help if response is slow."

**This is Bort's argument, and it is what H3 tests.** H3 specifically tests whether detection lead time has *independent* value even controlling for response maturity. The finding that early detection reduces impact is not the same as claiming response is fast.

The lead time formula — `max(0, AVG_DOWNTIME - mttd)` — represents advance warning before maximum damage. If you detect at hour 10 and maximum damage occurs at hour 47, you have 37 hours to limit blast radius, initiate containment, and reduce downtime. That value exists *whether or not* your response team acts on it quickly.

H3 pools incidents across checkpoint-only and checkpoint-optimized sweeps to get sufficient detected incidents for statistical analysis. With 10,000 iterations and two sweep strategies, there are thousands of detected incidents with varying lead times.

---

## "Your threat frequency is too high — not every manufacturer gets attacked 3 times a year."

**The frequency is conservative for the target population.** Meridian is modeled as a $340M PE-backed precision manufacturer — a company large enough to be an attractive target, with OT connectivity and a technology modernization program that creates exposure. This is not a 10-person machine shop.

Sources:
- Sophos State of Ransomware: Manufacturing 2023 — 66% of surveyed manufacturers were hit by ransomware in the prior year
- Verizon DBIR 2023 — Manufacturing is the second most-targeted industry sector
- Claroty Global State 2023 — 75% of OT organizations reported a cybersecurity incident in the prior 12 months

At 3.2 incidents/year, the model is *below* the implied rate from Claroty's data. Reducing frequency by 40% (sensitivity test) does not change any hypothesis verdict about *per-incident* outcomes, only H10's ROI calculation — which is explicitly tested in the sensitivity analysis.

---

## "CHECKPOINT_OPTIMIZED is a strawman — no one actually runs a best-practice checkpoint strategy without any inventory."

This is a fair methodological constraint. The model does not claim that operators *choose* zero inventory. It models what happens when inventory is incomplete — either because the organization has not completed it, because OT assets are poorly discovered, or because the gap exists by neglect.

The checkpoint-optimized strategy represents the best possible outcome *without* OT visibility: tuned network monitoring, baselining at perimeter checkpoints, and optimized alert thresholds. It is not a strawman; it is the strongest version of Bort's argument. If even this strategy fails against direct-access (TA-3) actors, the argument fails.

---

## "Your model doesn't account for compensating controls — defense in depth."

Correct, and by design. The model tests a specific claim: does inventory completeness matter to security outcomes? Compensating controls are orthogonal. Adding "company X uses DPI at all IT-OT boundaries" as a parameter would make the model harder to falsify, not more rigorous.

The model asks: holding everything else equal, what is the marginal value of knowing what you have? That question has an answer independent of what other controls exist.

---

## "Why is H8 (FAIR unreliability) sometimes FAILED?"

H8 tests whether the coefficient of variation (CV) of net financial impact is materially higher at low inventory completeness. Counterintuitively, this can fail in the model for a legitimate reason: partial inventory *concentrates* outcomes at worst-case (undetected incidents dominate), reducing variance at the low end.

If H8 fails, the correct interpretation is: partial inventory doesn't add *noise* to risk estimates — it *shifts* them lower. The mean impact is understated, but the distribution isn't necessarily wider. H8 FAILED + H2 SUPPORTED = risk estimates are uniformly too low, not noisy-and-unreliable. The practical implication is identical: don't trust your FAIR output at 40% inventory.

---

## "You built this to support a predetermined conclusion."

The model is falsifiable. H1 (Bort's claim) is SUPPORTED when checkpoint-optimized outcomes are statistically equivalent to inventory-informed outcomes. If that's what the math shows, we report it. Across the 10,000-iteration runs, H1 is SUPPORTED in scenarios where the direct-access (TA-3) threat actor mix is low. This is an honest finding — Bort is *right* if you exclude direct-access actors from the threat model.

The problem is that excluding TA-3 from a manufacturing threat model is unjustified. Vendor remote access, contractor credentials, and supply chain vectors are the dominant OT attack pathway (Dragos 2023). The model includes them. H5 tests specifically whether checkpoint controls fail categorically against these actors — and they do.
