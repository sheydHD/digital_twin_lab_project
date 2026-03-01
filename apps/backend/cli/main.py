#!/usr/bin/env python3
"""
Bayesian Model Selection for Beam Theory in Structural Health Monitoring Digital Twins.

Main entry point for the project pipeline. This script orchestrates:
1. Synthetic data generation using high-fidelity FEM
2. Bayesian calibration of Euler-Bernoulli and Timoshenko beam theories
3. Model selection analysis using marginal likelihoods
4. Visualization and reporting of results

Author: Digital Twins Lab
Date: January 2026
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from apps.backend.core.pipeline.orchestrator import PipelineOrchestrator
from apps.backend.core.utils.config import load_config
from apps.backend.core.utils.logging_setup import setup_logging

console = Console()


def print_banner():
    """Print the project banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║     Bayesian Model Selection for Beam Theory in Digital Twins                ║
║                                                                              ║
║     Comparing Euler-Bernoulli vs Timoshenko Beam Theories                    ║
║     for Structural Health Monitoring Applications                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="configs/default_config.yaml",
    help="Path to configuration file.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="outputs",
    help="Directory for output files.",
)
@click.option(
    "--stage",
    "-s",
    type=click.Choice(["all", "data", "calibration", "analysis", "report", "optimize"]),
    default="all",
    help="Pipeline stage to run.",
)
@click.option(
    "--aspect-ratios",
    "-a",
    multiple=True,
    type=float,
    default=None,
    help="Beam aspect ratios (L/h) to analyze. Can be specified multiple times.",
)
@click.option(
    "--optimize",
    is_flag=True,
    default=False,
    help="Run hyperparameter optimization before calibration.",
)
@click.option(
    "--optimize-trials",
    type=int,
    default=20,
    help="Number of optimization trials (default: 20).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode with additional logging.",
)
def main(
    config: str,
    output_dir: str,
    stage: str,
    aspect_ratios: tuple,
    optimize: bool,
    optimize_trials: int,
    verbose: bool,
    debug: bool,
):
    """
    Run the Bayesian Model Selection pipeline for beam theory comparison.

    This tool generates synthetic cantilever beam data, calibrates both
    Euler-Bernoulli and Timoshenko beam models using Bayesian inference,
    and performs model selection to determine which theory is appropriate
    for different beam configurations.
    """
    # Setup logging
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    print_banner()

    try:
        # Load configuration
        console.print("\n[bold cyan]Loading configuration...[/bold cyan]")
        cfg = load_config(config)

        # Override aspect ratios if provided via CLI
        if aspect_ratios:
            cfg["beam_parameters"]["aspect_ratios"] = list(aspect_ratios)
            console.print(f"  Using custom aspect ratios: {list(aspect_ratios)}")

        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cfg["output_dir"] = str(output_path)

        console.print(f"  Config file: {config}")
        console.print(f"  Output directory: {output_path}")
        console.print(f"  Pipeline stage: {stage}")

        # Initialize and run pipeline
        console.print("\n[bold cyan]Initializing pipeline...[/bold cyan]")
        orchestrator = PipelineOrchestrator(cfg)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if stage == "optimize":
                # Run optimization only
                task = progress.add_task("[cyan]Running hyperparameter optimization...", total=None)
                orchestrator.run_data_generation()
                results = orchestrator.run_optimization(n_trials=optimize_trials)
                progress.update(task, completed=True)
                console.print("\n[bold green]Optimization completed![/bold green]")
                console.print(f"Best score: {results['best_score']:.4f}")
                console.print(f"Best parameters: {results['best_params']}")
                return

            elif stage == "all":
                # Run full pipeline
                task = progress.add_task("[cyan]Running full pipeline...", total=None)

                # Optional: run optimization first
                if optimize:
                    progress.update(
                        task, description="[cyan]Running hyperparameter optimization..."
                    )
                    orchestrator.run_data_generation()
                    opt_results = orchestrator.run_optimization(n_trials=optimize_trials)
                    progress.update(
                        task, description="[cyan]Running calibration with optimized params..."
                    )
                    orchestrator.run_calibration_with_optimized_params(opt_results["best_params"])
                    orchestrator.run_analysis()
                    orchestrator.run_frequency_analysis()
                    results = orchestrator.generate_report()
                else:
                    results = orchestrator.run_full_pipeline()

                progress.update(task, completed=True)

            elif stage == "data":
                task = progress.add_task("[cyan]Generating synthetic data...", total=None)
                results = orchestrator.run_data_generation()
                progress.update(task, completed=True)

            elif stage == "calibration":
                task = progress.add_task("[cyan]Running Bayesian calibration...", total=None)
                results = orchestrator.run_calibration()
                progress.update(task, completed=True)

            elif stage == "analysis":
                task = progress.add_task("[cyan]Analyzing model selection...", total=None)
                results = orchestrator.run_analysis()
                progress.update(task, completed=True)

            elif stage == "report":
                task = progress.add_task("[cyan]Generating report...", total=None)
                results = orchestrator.generate_report()
                progress.update(task, completed=True)

        # Print summary
        console.print("\n[bold green]Pipeline completed successfully![/bold green]")
        orchestrator.print_summary(results)

    except FileNotFoundError as e:
        console.print(f"\n[bold red]Configuration error:[/bold red] {e}")
        logger.exception("Configuration file not found")
        raise SystemExit(1) from e

    except Exception as e:
        console.print(f"\n[bold red]Pipeline error:[/bold red] {e}")
        logger.exception("Pipeline failed with error")
        if debug:
            raise
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
