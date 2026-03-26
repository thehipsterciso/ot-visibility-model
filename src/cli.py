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
    """Run all 10 hypotheses. Full model."""
    from src.assets.graph import MeridianGraph
    from src.simulation.engine import SimulationEngine
    from src.constants import ControlStrategy, ThreatActor
    from src.hypotheses.runner import (
        H1BortHypothesis, H2FoundationHypothesis, H3ActionabilityGapHypothesis,
        H4BlastRadiusHypothesis, H5InsiderHypothesis, H6SegmentationQualityHypothesis,
        H7ResponseInflectionHypothesis, H8RiskQuantificationHypothesis,
        H9ComplianceExposureHypothesis, H10CostParityHypothesis,
    )

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    console.print(f"\n[bold cyan]Running full OT Visibility Model — {iterations:,} iterations per scenario[/bold cyan]\n")

    with console.status("[cyan]S0: Meridian baseline..."):
        s0 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            iterations=iterations, scenario_id="S0").run()
    with console.status("[cyan]S1: Bort model (checkpoint-optimized)..."):
        s1 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_OPTIMIZED,
            iterations=iterations, scenario_id="S1").run()
    with console.status("[cyan]S2: Inventory-first..."):
        s2 = SimulationEngine(inventory_completeness=1.00, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            iterations=iterations, scenario_id="S2").run()
    # S1_inv: same inventory as S1 (40%) but with INVENTORY_INFORMED strategy
    # Isolates strategy effect from inventory effect for fair H1 comparison
    with console.status("[cyan]S1_inv: Inventory-informed at 40% inventory (H1 same-inventory comparison)..."):
        s1_inv = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.INVENTORY_INFORMED,
            iterations=iterations, scenario_id="S1_inv").run()
    with console.status("[cyan]S5: Insider/vendor only..."):
        s5 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0}, iterations=iterations, scenario_id="S5").run()
    with console.status("[cyan]H5 inventory comparison..."):
        h5_inv = SimulationEngine(inventory_completeness=1.00, control_strategy=ControlStrategy.INVENTORY_INFORMED,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0}, iterations=iterations, scenario_id="H5_inv").run()

    sweep_levels = [0.0, 0.25, 0.40, 0.60, 0.75, 0.90, 1.0]
    sweep_results = []
    for level in sweep_levels:
        with console.status(f"[cyan]Sweep (checkpoint-only): {level:.0%} inventory..."):
            result = SimulationEngine(inventory_completeness=level,
                control_strategy=ControlStrategy.CHECKPOINT_ONLY,
                iterations=iterations, scenario_id=f"S4_{int(level*100)}").run()
            sweep_results.append(result)

    # Optimized sweep: same levels with CHECKPOINT_OPTIMIZED — used by H2/H3/H7/H8
    # to demonstrate that the inventory→outcome correlation holds across strategies
    optimized_sweep_results = []
    for level in sweep_levels:
        with console.status(f"[cyan]Sweep (checkpoint-optimized): {level:.0%} inventory..."):
            result = SimulationEngine(inventory_completeness=level,
                control_strategy=ControlStrategy.CHECKPOINT_OPTIMIZED,
                iterations=iterations, scenario_id=f"S4opt_{int(level*100)}").run()
            optimized_sweep_results.append(result)

    # Build baseline graph for H6
    baseline_graph = MeridianGraph(inventory_completeness=0.40).build()

    console.print("\n[bold cyan]Evaluating hypotheses...[/bold cyan]\n")

    results_list = [
        H1BortHypothesis().evaluate(
            checkpoint_results=s1, inventory_results=s2,
            checkpoint_same_inv_results=s1,
            inventory_same_inv_results=s1_inv,
        ),
        H2FoundationHypothesis().evaluate(
            sweep_results=sweep_results,
            optimized_sweep_results=optimized_sweep_results,
        ),
        H3ActionabilityGapHypothesis().evaluate(
            sweep_results=sweep_results,
            optimized_sweep_results=optimized_sweep_results,
        ),
        H4BlastRadiusHypothesis().evaluate(sweep_results=sweep_results),
        H5InsiderHypothesis().evaluate(checkpoint_ta3_results=s5, inventory_ta3_results=h5_inv),
        H6SegmentationQualityHypothesis().evaluate(sweep_results=sweep_results, graph=baseline_graph),
        H7ResponseInflectionHypothesis().evaluate(
            sweep_results=sweep_results,
            optimized_sweep_results=optimized_sweep_results,
        ),
        H8RiskQuantificationHypothesis().evaluate(
            sweep_results=sweep_results,
            optimized_sweep_results=optimized_sweep_results,
        ),
        H9ComplianceExposureHypothesis().evaluate(sweep_results=sweep_results),
        H10CostParityHypothesis().evaluate(baseline_results=s0, full_inv_results=s2),
    ]

    table = Table(title="OT Visibility Model — Hypothesis Verdicts", show_lines=True)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Hypothesis", style="white", width=55)
    table.add_column("Verdict", style="bold", width=14)
    verdict_colors = {"SUPPORTED": "green", "FAILED": "red", "INCONCLUSIVE": "yellow"}
    for r in results_list:
        color = verdict_colors.get(r.verdict.value, "white")
        table.add_row(r.hypothesis_id, r.title[:55], f"[{color}]{r.verdict.value}[/{color}]")
    console.print(table)

    console.print("\n[bold cyan]Key Findings[/bold cyan]\n")
    for r in results_list:
        console.print(f"[bold]{r.hypothesis_id}:[/bold] {r.key_finding}\n")

    results_data = {r.hypothesis_id: {"verdict": r.verdict.value, "key_finding": r.key_finding,
        "primary_value": r.primary_value} for r in results_list}
    output_file = output_path / "hypothesis_results.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    console.print(f"\n[green]Results saved to {output_file}[/green]")

    return {
        "results": results_list,
        "scenarios": {
            "s0": s0, "s1": s1, "s2": s2, "s1_inv": s1_inv, "s5": s5, "h5_inv": h5_inv,
            "sweep": sweep_results, "optimized_sweep": optimized_sweep_results,
            "baseline_graph": baseline_graph,
        },
    }


@cli.command()
@click.option("--iterations", "-n", default=10000, help="Monte Carlo iterations per scenario")
@click.option("--output", "-o", default="outputs/report.html", help="Output HTML path")
def report(iterations: int, output: str):
    """Run all scenarios, evaluate all hypotheses, generate charts, write HTML report."""
    import matplotlib
    matplotlib.use("Agg")

    from src.assets.graph import MeridianGraph
    from src.simulation.engine import SimulationEngine
    from src.constants import ControlStrategy, ThreatActor
    from src.hypotheses.runner import (
        H1BortHypothesis, H2FoundationHypothesis, H3ActionabilityGapHypothesis,
        H4BlastRadiusHypothesis, H5InsiderHypothesis, H6SegmentationQualityHypothesis,
        H7ResponseInflectionHypothesis, H8RiskQuantificationHypothesis,
        H9ComplianceExposureHypothesis, H10CostParityHypothesis,
    )
    from src.visualization import plots
    from src.visualization.dashboard import generate_html_report

    output_path = Path(output)
    console.print(f"\n[bold cyan]OT Visibility Model — Generating Report ({iterations:,} iterations)[/bold cyan]\n")

    # --- Run scenarios ---
    with console.status("[cyan]S0: Meridian baseline..."):
        s0 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            iterations=iterations, scenario_id="S0").run()
    with console.status("[cyan]S1: Bort model..."):
        s1 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_OPTIMIZED,
            iterations=iterations, scenario_id="S1").run()
    with console.status("[cyan]S2: Inventory-first..."):
        s2 = SimulationEngine(inventory_completeness=1.00, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            iterations=iterations, scenario_id="S2").run()
    with console.status("[cyan]S1_inv: Inventory-informed at 40% (H1 same-inventory comparison)..."):
        s1_inv = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.INVENTORY_INFORMED,
            iterations=iterations, scenario_id="S1_inv").run()
    with console.status("[cyan]S5: Insider/vendor only..."):
        s5 = SimulationEngine(inventory_completeness=0.40, control_strategy=ControlStrategy.CHECKPOINT_ONLY,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0}, iterations=iterations, scenario_id="S5").run()
    with console.status("[cyan]H5 inventory comparison..."):
        h5_inv = SimulationEngine(inventory_completeness=1.00, control_strategy=ControlStrategy.INVENTORY_INFORMED,
            threat_mix={ThreatActor.TA3_INSIDER_VENDOR: 1.0}, iterations=iterations, scenario_id="H5_inv").run()

    sweep_levels = [0.0, 0.25, 0.40, 0.60, 0.75, 0.90, 1.0]
    sweep_results = []
    for level in sweep_levels:
        with console.status(f"[cyan]Sweep (checkpoint-only): {level:.0%} inventory..."):
            r = SimulationEngine(inventory_completeness=level,
                control_strategy=ControlStrategy.CHECKPOINT_ONLY,
                iterations=iterations, scenario_id=f"S4_{int(level*100)}").run()
            sweep_results.append(r)

    optimized_sweep_results = []
    for level in sweep_levels:
        with console.status(f"[cyan]Sweep (checkpoint-optimized): {level:.0%} inventory..."):
            r = SimulationEngine(inventory_completeness=level,
                control_strategy=ControlStrategy.CHECKPOINT_OPTIMIZED,
                iterations=iterations, scenario_id=f"S4opt_{int(level*100)}").run()
            optimized_sweep_results.append(r)

    baseline_graph = MeridianGraph(inventory_completeness=0.40).build()

    # --- Evaluate hypotheses ---
    console.print("[bold cyan]Evaluating hypotheses...[/bold cyan]")
    h1 = H1BortHypothesis().evaluate(
        checkpoint_results=s1, inventory_results=s2,
        checkpoint_same_inv_results=s1,
        inventory_same_inv_results=s1_inv,
    )
    h2 = H2FoundationHypothesis().evaluate(
        sweep_results=sweep_results, optimized_sweep_results=optimized_sweep_results)
    h3 = H3ActionabilityGapHypothesis().evaluate(
        sweep_results=sweep_results, optimized_sweep_results=optimized_sweep_results)
    h4 = H4BlastRadiusHypothesis().evaluate(sweep_results=sweep_results)
    h5 = H5InsiderHypothesis().evaluate(checkpoint_ta3_results=s5, inventory_ta3_results=h5_inv)
    h6 = H6SegmentationQualityHypothesis().evaluate(sweep_results=sweep_results, graph=baseline_graph)
    h7 = H7ResponseInflectionHypothesis().evaluate(
        sweep_results=sweep_results, optimized_sweep_results=optimized_sweep_results)
    h8 = H8RiskQuantificationHypothesis().evaluate(
        sweep_results=sweep_results, optimized_sweep_results=optimized_sweep_results)
    h9 = H9ComplianceExposureHypothesis().evaluate(sweep_results=sweep_results)
    h10 = H10CostParityHypothesis().evaluate(baseline_results=s0, full_inv_results=s2)

    results = {r.hypothesis_id: r for r in [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]}

    # --- Generate charts ---
    console.print("[bold cyan]Generating charts...[/bold cyan]")
    charts = {
        "H1": plots.plot_h1(h1, s1, s2),
        "H2": plots.plot_h2(h2, sweep_results),
        "H3": plots.plot_h3(h3, sweep_results),
        "H4": plots.plot_h4(h4, sweep_results),
        "H5": plots.plot_h5(h5, s5, h5_inv),
        "H6": plots.plot_h6(h6),
        "H7": plots.plot_h7(h7, sweep_results),
        "H8": plots.plot_h8(h8, sweep_results),
        "H9": plots.plot_h9(h9),
        "H10": plots.plot_h10(h10),
    }

    # --- Write report ---
    console.print(f"[bold cyan]Writing report to {output_path}...[/bold cyan]")
    generate_html_report(results, charts, output_path)
    console.print(f"\n[green]Report written to {output_path}[/green]")


@cli.command()
@click.option("--iterations", "-n", default=2000, help="Monte Carlo iterations per perturbation scenario")
@click.option("--output", "-o", default="outputs/sensitivity_report.json", help="Output JSON path")
def sensitivity(iterations: int, output: str):
    """Run parameter sensitivity analysis across ±20%/±40% perturbations on key model parameters."""
    from src.sensitivity import ParameterSensitivityAnalyzer, PERTURBATION_LEVELS

    output_path = Path(output)
    output_path.parent.mkdir(exist_ok=True)

    console.print(f"\n[bold cyan]OT Visibility Model — Parameter Sensitivity Analysis[/bold cyan]")
    console.print(f"[dim]{len(PERTURBATION_LEVELS)} perturbation levels × {iterations:,} iterations each[/dim]\n")

    analyzer = ParameterSensitivityAnalyzer(iterations=iterations)

    def progress(label: str):
        console.print(f"  [cyan]Running:[/cyan] {label}")

    report_data = analyzer.run(progress_callback=progress)

    table = Table(title="Sensitivity Analysis — Hypothesis Stability", show_lines=True)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Baseline", style="bold", width=14)
    table.add_column("Stable?", style="bold", width=8)
    table.add_column("Instability Count", style="white", width=18)
    table.add_column("Primary Value CV", style="white", width=18)
    for r in report_data.results:
        stable = r.is_stable()
        color = "green" if stable else "yellow"
        table.add_row(
            r.hypothesis_id,
            r.baseline_verdict,
            f"[{color}]{'YES' if stable else 'NO'}[/{color}]",
            str(r.instability_count()),
            f"{r.primary_value_cv():.3f}",
        )
    console.print(table)

    stable = report_data.stable_verdicts()
    unstable = report_data.unstable_verdicts()
    console.print(f"\n[green]Stable verdicts: {len(stable)}/10[/green]")
    if unstable:
        console.print(f"[yellow]Parameter-sensitive verdicts: {', '.join(r.hypothesis_id for r in unstable)}[/yellow]")

    output_data = {
        "summary": report_data.summary_table(),
        "details": {
            r.hypothesis_id: {
                "baseline_verdict": r.baseline_verdict,
                "is_stable": r.is_stable(),
                "instability_count": r.instability_count(),
                "primary_value_cv": r.primary_value_cv(),
                "verdicts_by_perturbation": r.verdicts_by_perturbation,
            }
            for r in report_data.results
        },
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    console.print(f"\n[green]Sensitivity report saved to {output_path}[/green]")


if __name__ == "__main__":
    cli()
