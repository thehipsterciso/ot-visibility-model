# OT Visibility Model

A Monte Carlo simulation that mathematically tests whether "OT visibility is overrated."

## The Question

Bryson Bort's S4x26 argument: OT asset visibility is overrated because you cannot act on what you see. Strategic IT/OT boundary checkpoints provide equivalent or superior outcomes.

Counter-position: asset inventory is not a visibility tool. It is the prerequisite to segmentation, risk quantification, incident response, compliance, and financial modeling.

## Synthetic Organization

**Meridian Precision Manufacturing** — $340M PE-backed discrete manufacturer, 1,200 employees, 3 sites, 820 OT assets at ~40% inventory coverage.

## The 10 Hypotheses

| # | Hypothesis |
|---|---|
| H1 | Bort: Checkpoints ≥ inventory for security outcomes |
| H2 | Foundation: Inventory is prerequisite, not tool |
| H3 | Actionability gap is a response problem, not a visibility problem |
| H4 | Unknown assets create super-linear blast radius growth |
| H5 | Checkpoint model fails against direct OT access (insider/vendor) |
| H6 | Segmentation without inventory produces misconfigured boundaries |
| H7 | Incident response degrades non-linearly below ~70% inventory |
| H8 | Risk quantification is unreliable below minimum inventory threshold |
| H9 | Compliance exposure compounds independent of security outcomes |
| H10 | Inventory cost < expected loss from inventory gaps over 3 years |

H1, H2, H4, H5, H7, H10 are fully implemented. H3, H6, H8, H9 are stubbed for Claude Code to complete.

## Quick Start

```bash
poetry install
pytest tests/ -v
ot-model generate-org
ot-model run-all --iterations 10000
```

## License

MIT
