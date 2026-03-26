"""
cli.py — Command-line interface for the OT Visibility Model.
"""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli():
    """OT Visibility Model — Test whether 'OT visibility is overrated' survives mathematical scrutiny."""
    pass


@cli.command()
def generate_org():
    """Build the Meridian synthetic asset graph and display summary."""
    from src.assets.graph import MeridianGraph
    console.print("\n[bold cyan]Building Meridian Precision Manufacturing asset graph...[/bold cyan]")
    graph = MeridianGraph(inventory_completeness=0.40).build()
    summary = graph.summary()
    table = Table(title="Meridian Asset Graph — Baseline (40% OT Inventory)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    for k, v in summary.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


@cli.command()
@click.option("--iterations", "-n", default=10000, help="Monte Carlo iterations per scenario")
@click.option("--output-dir", "-o", default="outputs", help="Output directory")
def run_all(iterations: int, output_dir: str):
    """Run all implemented hypotheses. Full model."""
    from src.simulation.engine import SimulationEngine
    from src.constants import ControlStrategy, ThreatActor
    from src.hypotheses.runner import (
        H1BortHypothesis, H2FoundationHypothesis, H4BlastRadiusHypothesis,
        H5InsiderHypothesis, H7ResponseInflectionHypothesis, H10CostParityHypothesis,
    )

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    console.print(f"\n[bold cyan]Running full OT Visibility Model — {iterations:,} iterations per scenario[/bold cyan]\n")

    with console.status("[cyan]S0: Meridian baseline..."):
        s0 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            iterations=iterations, scenario_id="S0").run()
    with console.status("[cyan]S1: Bort model..."):
        s1 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_OPTIMIZED,
            iterations=iterations, scenario_id="S1").run()
    with console.status("[cyan]S2: Inventory-first..."):
        s2 = SimulationEngine(inventory_completeness=1.00, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            iterations=iterations, scenario_id="S2").run()
    with console.status("[cyan]S5: Insider/vendor only..."):
        s5 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0}, iterations=iterations, scenario_id="S5").run()
    with console.status("[cyan]H5 inventory comparison..."):
        h5_inv = SimulationEngine(inventory_completeness=1.00, control_strategy=ControlStrategy.INVENTORY_INFORMED,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0}, iterations=iterations, scenario_id="H5_inv").run()


    sweep_levels = [0.0, 0.25, 0.40, 0.60, 0.75, 0.90, 1.0]
    sweep_results = []
    for level in sweep_levels:
        with console.status(f"[cyan]Sweep: {level:.0%} inventory..."):
            result = SimulationEngine(inventory_completeness=level,
                control_strategy=ControlStrategy.CHECKPOINT_ONLY,
                iterations=iterations, scenario_id=f"S4_{int(level*100)}").run()
            sweep_results.append(result)

    console.print("\n[bold cyan]Evaluating hypotheses...[/bold cyan]\n")
    results = [
        H1BortHypothesis().evaluate(checkpoint_results=s1, inventory_results=s2),
        H2FoundationHypothesis().evaluate(sweep_results=sweep_results),
        H4BlastRadiusHypothesis().evaluate(sweep_results=sweep_results),
        H5InsiderHypothesis().evaluate(checkpoint_ta3_results=s5, inventory_ta3_results=h5_inv),
        H7ResponseInflectionHypothesis().evaluate(sweep_results=sweep_results),
        H10CostParityHypothesis().evaluate(baseline_results=s0, full_inv_results=s2),
    ]

    table = Table(title="OT Visibility Model — Hypothesis Verdicts", show_lines=True)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Hypothesis", style="white", width=55)
    table.add_column("Verdict", style="bold", width=14)
    verdict_colors = {"SUPPORTED": "green", "FAILED": "red", "INCONCLUSIVE": "yellow"}
    for r in results:
        color = verdict_colors.get(r.verdict.value, "white")
        table.add_row(r.hypothesis_id, r.title[:55], f"[{color}]{r.verdict.value}[/{color}]")
    console.print(table)

    console.print("\n[bold cyan]Key Findings[/bold cyan]\n")
    for r in results:
        console.print(f"[bold]{r.hypothesis_id}:[/bold] {r.key_finding}\n")

    results_data = {r.hypothesis_id: {"verdict": r.verdict.value, "key_finding": r.key_finding,
        "primary_value": r.primary_value} for r in results}
    output_file = output_path / "hypothesis_results.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    console.print(f"\n[green]Results saved to {output_file}[/green]")


if __name__ == "__main__":
    cli()
