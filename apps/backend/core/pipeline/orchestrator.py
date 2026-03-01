"""
Pipeline Orchestrator.

Coordinates the full Bayesian model selection workflow:
1. Synthetic data generation
2. Bayesian calibration of beam models
3. Model comparison and selection
4. Frequency analysis
5. Hyperparameter optimization (optional)
6. Analysis and reporting
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from apps.backend.core.analysis.reporter import ResultsReporter
from apps.backend.core.analysis.visualization import BeamVisualization
from apps.backend.core.bayesian.calibration import (
    CalibrationResult,
    EulerBernoulliCalibrator,
    TimoshenkoCalibrator,
    create_default_priors,
    create_timoshenko_priors,
)
from apps.backend.core.bayesian.model_selection import (
    BayesianModelSelector,
    ModelComparisonResult,
)
from apps.backend.core.data.synthetic_generator import (
    NoiseModel,
    SensorConfiguration,
    SyntheticDataGenerator,
    SyntheticDataset,
    save_dataset,
)
from apps.backend.core.models.base_beam import LoadCase, MaterialProperties

logger = logging.getLogger(__name__)
console = Console()


# ── Module-level worker (must be picklable for ProcessPoolExecutor) ──────────


def _run_single_calibration(
    dataset: SyntheticDataset,
    calibrator_cls,  # EulerBernoulliCalibrator | TimoshenkoCalibrator
    priors: list,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    target_accept: float,
) -> CalibrationResult:
    """Run one Bayesian calibration inside a subprocess worker.

    Lives at module level so it is picklable by ``ProcessPoolExecutor``.
    Each worker gets an isolated Python interpreter, which means separate
    PyMC / PyTensor state — no shared global graph, no GIL contention.
    """
    calibrator = calibrator_cls(
        priors=priors,
        n_samples=n_samples,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=target_accept,
    )
    result = calibrator.calibrate(dataset)
    result.marginal_likelihood_estimate = calibrator.compute_marginal_likelihood()
    return result


class PipelineOrchestrator:
    """
    Orchestrate the complete Bayesian model selection pipeline.

    This class coordinates all stages of the analysis:
    - Data generation using FEM
    - Bayesian calibration of competing models
    - Model selection using Bayes factors
    - Result visualization and reporting

    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Configuration dictionary with all parameters
        """
        self.config = config
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_components()

        # Storage for results
        self.datasets: list[SyntheticDataset] = []
        self.eb_results: list[CalibrationResult] = []
        self.timo_results: list[CalibrationResult] = []
        self.comparisons: list[ModelComparisonResult] = []
        self.study_results: dict | None = None
        self.frequency_results: dict | None = None
        self.optimization_results: dict | None = None

    def _setup_components(self) -> None:
        """Set up pipeline components from configuration."""
        # Extract configuration sections
        beam_cfg = self.config.get("beam_parameters", {})
        bayesian_cfg = self.config.get("bayesian", {})
        data_cfg = self.config.get("data_generation", {})

        # Sensor configuration
        n_sensors = data_cfg.get("n_displacement_sensors", 5)
        n_strain = data_cfg.get("n_strain_gauges", 4)

        # Default sensor locations (will be scaled to beam length)
        self.sensors = SensorConfiguration(
            displacement_locations=np.linspace(0.2, 1.0, n_sensors),
            strain_locations=np.linspace(0.1, 0.9, n_strain),
        )

        # Noise model
        self.noise = NoiseModel(
            displacement_std=data_cfg.get("displacement_noise", 1e-6),
            strain_std=data_cfg.get("strain_noise", 1e-6),
            relative_noise=data_cfg.get("relative_noise", True),
            noise_fraction=data_cfg.get("noise_fraction", 0.005),
            seed=data_cfg.get("seed", 42),
        )

        # Material properties
        mat_cfg = self.config.get("material", {})
        self.material = MaterialProperties(
            elastic_modulus=mat_cfg.get("elastic_modulus", 210e9),
            poisson_ratio=mat_cfg.get("poisson_ratio", 0.3),
            density=mat_cfg.get("density", 7850),
        )

        # Load case
        load_cfg = self.config.get("load", {})
        self.load = LoadCase(
            point_load=load_cfg.get("point_load", 1000),
            distributed_load=load_cfg.get("distributed_load", 0),
        )

        # Aspect ratios to study
        self.aspect_ratios = beam_cfg.get("aspect_ratios", [5, 8, 10, 12, 15, 20, 30, 50])
        self.base_length = beam_cfg.get("length", 1.0)
        self.base_width = beam_cfg.get("width", 0.1)

        # Bayesian sampling parameters
        self.n_samples = bayesian_cfg.get("n_samples", 2000)
        self.n_tune = bayesian_cfg.get("n_tune", 1000)
        self.n_chains = bayesian_cfg.get("n_chains", 4)

        # Visualization and reporting
        self.visualizer = BeamVisualization(self.output_dir / "figures")
        self.reporter = ResultsReporter(self.output_dir / "reports")

    def run_full_pipeline(self) -> dict:
        """
        Run the complete pipeline.

        Returns:
            Dictionary with all results

        """
        console.print("\n[bold green]Starting Full Pipeline[/bold green]")

        # Stage 1: Generate synthetic data
        console.print("\n[cyan]Stage 1: Generating synthetic data...[/cyan]")
        self.run_data_generation()

        # Stage 2: Bayesian calibration
        console.print("\n[cyan]Stage 2: Running Bayesian calibration...[/cyan]")
        self.run_calibration()

        # Stage 3: Model selection analysis
        console.print("\n[cyan]Stage 3: Analyzing model selection...[/cyan]")
        self.run_analysis()

        # Stage 4: Frequency analysis
        console.print("\n[cyan]Stage 4: Running frequency analysis...[/cyan]")
        self.run_frequency_analysis()

        # Stage 5: Generate reports
        console.print("\n[cyan]Stage 5: Generating reports...[/cyan]")
        self.generate_report()

        return {
            "datasets": self.datasets,
            "eb_results": self.eb_results,
            "timo_results": self.timo_results,
            "study_results": self.study_results,
            "frequency_results": self.frequency_results,
        }

    def run_data_generation(self) -> list[SyntheticDataset]:
        """
        Generate synthetic measurement data for all aspect ratios.

        Returns:
            List of synthetic datasets

        """
        logger.info("Generating synthetic data...")

        generator = SyntheticDataGenerator(
            sensors=self.sensors,
            noise=self.noise,
            fem_refinement=(40, 8),
        )

        self.datasets = generator.generate_parametric_study(
            aspect_ratios=self.aspect_ratios,
            base_length=self.base_length,
            base_material=self.material,
            base_load=self.load,
            width=self.base_width,
        )

        # Save datasets
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        for _i, dataset in enumerate(self.datasets):
            L_h = dataset.geometry.aspect_ratio
            save_dataset(dataset, data_dir / f"dataset_Lh_{L_h:.0f}.h5")

        console.print(f"  Generated {len(self.datasets)} datasets")

        return self.datasets

    def run_calibration(self, progress_callback=None) -> dict:
        """
        Run Bayesian calibration for both beam theories.

        Euler-Bernoulli and Timoshenko calibrations are independent for
        every dataset, so they are submitted to a ``ProcessPoolExecutor``
        and executed concurrently.  Using separate *processes* (not threads)
        gives each calibration its own PyMC / PyTensor graph — no shared
        state, no GIL bottleneck.

        Wall-clock time is roughly halved compared with the serial loop on
        a dual-core machine.  ``max_workers`` is capped to 2 (one per model
        per dataset at a time) so PyMC's own internal chain-parallelism
        (``n_chains``) doesn't cascade into too many nested subprocesses.

        Args:
            progress_callback: Optional callable ``(step, total, lh)`` called
                after each dataset pair is calibrated.

        Returns:
            Dictionary with calibration results
        """
        logger.info("Running Bayesian calibration (parallel EB + Timo)...")

        if not self.datasets:
            console.print("[yellow]No datasets found. Running data generation...[/yellow]")
            self.run_data_generation()

        eb_priors = create_default_priors(self.config)
        timo_priors = create_timoshenko_priors(self.config)
        target_accept = self.config.get("bayesian", {}).get("target_accept", 0.95)

        # Cap outer workers to 2 (EB vs Timo per dataset).
        # Each PyMC run already spawns n_chains sub-processes internally;
        # running more than 2 outer workers would create n_chains×workers
        # concurrent processes and thrash the CPU.
        max_outer = min(2, os.cpu_count() or 1)

        self.eb_results = []
        self.timo_results = []

        with ProcessPoolExecutor(max_workers=max_outer) as pool:
            for i, dataset in enumerate(self.datasets):
                L_h = dataset.geometry.aspect_ratio
                console.print(
                    f"\n  Calibrating for L/h = {L_h:.1f} "
                    f"({i + 1}/{len(self.datasets)})  [parallel EB + Timo]"
                )

                # Submit EB and Timo at the same time — they run concurrently.
                eb_future = pool.submit(
                    _run_single_calibration,
                    dataset,
                    EulerBernoulliCalibrator,
                    eb_priors,
                    self.n_samples,
                    self.n_tune,
                    self.n_chains,
                    target_accept,
                )
                timo_future = pool.submit(
                    _run_single_calibration,
                    dataset,
                    TimoshenkoCalibrator,
                    timo_priors,
                    self.n_samples,
                    self.n_tune,
                    self.n_chains,
                    target_accept,
                )

                self.eb_results.append(eb_future.result())
                self.timo_results.append(timo_future.result())

                if progress_callback:
                    progress_callback(i + 1, len(self.datasets), L_h)

        console.print(f"\n  Calibrated {len(self.eb_results)} model pairs")
        return {
            "euler_bernoulli": self.eb_results,
            "timoshenko": self.timo_results,
        }

    def run_analysis(self) -> dict:
        """
        Analyze model selection results.

        Returns:
            Study results dictionary

        """
        logger.info("Analyzing model selection...")

        if not self.eb_results or not self.timo_results:
            raise ValueError("Must run calibration before analysis")

        # Model selection with inconclusive threshold handling
        inconclusive_threshold = self.config.get("model_selection", {}).get(
            "inconclusive_threshold", 0.5
        )
        selector = BayesianModelSelector(inconclusive_threshold=inconclusive_threshold)

        self.study_results = selector.analyze_aspect_ratio_study(
            eb_results=self.eb_results,
            timo_results=self.timo_results,
            aspect_ratios=self.aspect_ratios,
        )

        # Generate visualizations
        self._generate_visualizations()

        return self.study_results

    def run_frequency_analysis(self) -> dict:
        """
        Analyze model selection across frequencies.

        This addresses the task requirement to analyze "loading frequencies".

        Returns:
            Frequency analysis results
        """
        logger.info("Running frequency analysis...")

        from apps.backend.core.bayesian.hyperparameter_optimization import (
            FrequencyBasedModelSelector,
        )

        freq_selector = FrequencyBasedModelSelector()
        self.frequency_results = freq_selector.analyze_frequency_study(self.datasets)

        # Print frequency summary
        console.print("  Frequency analysis complete")

        if self.frequency_results and "summary" in self.frequency_results:
            summary = self.frequency_results["summary"]
            if summary.get("typical_transition_mode"):
                console.print(f"  Typical transition mode: {summary['typical_transition_mode']}")

        return self.frequency_results

    def run_optimization(
        self,
        n_trials: int = 20,
        fast_mode: bool = True,
    ) -> dict:
        """
        Run hyperparameter optimization to improve model selection.

        This uses Optuna to find optimal prior parameters and MCMC settings
        that maximize model selection accuracy.

        Args:
            n_trials: Number of optimization trials
            fast_mode: Use subset of datasets for faster optimization

        Returns:
            Optimization results with best parameters
        """
        logger.info("Running hyperparameter optimization...")
        console.print("\n[cyan]Running hyperparameter optimization...[/cyan]")

        if not self.datasets:
            console.print("[yellow]No datasets found. Running data generation...[/yellow]")
            self.run_data_generation()

        from apps.backend.core.bayesian.hyperparameter_optimization import (
            BayesianHyperparameterOptimizer,
        )

        optimizer = BayesianHyperparameterOptimizer(
            datasets=self.datasets,
            output_dir=self.output_dir / "optimization",
        )

        result = optimizer.optimize(
            n_trials=n_trials,
            fast_mode=fast_mode,
        )

        self.optimization_results = {
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
        }

        console.print(f"  Best score: {result.best_score:.4f}")
        console.print(f"  Best parameters: {result.best_params}")

        return self.optimization_results

    def run_calibration_with_optimized_params(
        self,
        optimized_params: dict | None = None,
    ) -> dict:
        """
        Run calibration using optimized hyperparameters.

        Args:
            optimized_params: Parameters from optimization (uses defaults if None)

        Returns:
            Calibration results dictionary
        """
        logger.info("Running calibration with optimized parameters...")

        if not self.datasets:
            console.print("[yellow]No datasets found. Running data generation...[/yellow]")
            self.run_data_generation()

        from apps.backend.core.bayesian.hyperparameter_optimization import create_priors_from_params

        if optimized_params is None:
            # Use default optimized values
            optimized_params = {
                "E_prior_sigma": 0.05,
                "sigma_prior_scale": 1e-6,
                "nu_prior_sigma": 0.03,
                "n_samples": 800,
                "n_tune": 400,
                "target_accept": 0.95,
            }

        # Create priors from optimized parameters
        eb_priors = create_priors_from_params(optimized_params, include_poisson=False)
        timo_priors = create_priors_from_params(optimized_params, include_poisson=True)

        # Get sampling parameters
        n_samples = optimized_params.get("n_samples", self.n_samples)
        n_tune = optimized_params.get("n_tune", self.n_tune)
        target_accept = optimized_params.get("target_accept", 0.95)

        self.eb_results = []
        self.timo_results = []

        max_outer = min(2, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_outer) as pool:
            for i, dataset in enumerate(self.datasets):
                L_h = dataset.geometry.aspect_ratio
                console.print(
                    f"\n  Calibrating for L/h = {L_h:.1f} "
                    f"({i + 1}/{len(self.datasets)})  [parallel EB + Timo]"
                )
                eb_future = pool.submit(
                    _run_single_calibration,
                    dataset,
                    EulerBernoulliCalibrator,
                    eb_priors,
                    n_samples,
                    n_tune,
                    self.n_chains,
                    target_accept,
                )
                timo_future = pool.submit(
                    _run_single_calibration,
                    dataset,
                    TimoshenkoCalibrator,
                    timo_priors,
                    n_samples,
                    n_tune,
                    self.n_chains,
                    target_accept,
                )
                self.eb_results.append(eb_future.result())
                self.timo_results.append(timo_future.result())

        console.print(f"\n  Calibrated {len(self.eb_results)} model pairs with optimized params")
        return {
            "euler_bernoulli": self.eb_results,
            "timoshenko": self.timo_results,
        }

    def _generate_visualizations(self) -> None:
        """Generate all visualization outputs."""
        if not self.study_results:
            return

        # Ensure output directory exists (may have been cleaned after init)
        self.visualizer.output_dir.mkdir(parents=True, exist_ok=True)

        console.print("  Generating visualizations...")

        # Aspect ratio study plot
        self.visualizer.plot_aspect_ratio_study(self.study_results)

        # WAIC comparison plot
        self.visualizer.plot_waic_comparison(
            study_results=self.study_results,
            eb_results=self.eb_results,
            tim_results=self.timo_results,
        )

        # Deflection error plot (shear contribution)
        self.visualizer.plot_deflection_error(
            aspect_ratios=self.aspect_ratios,
            base_length=self.base_length,
            material=self.material,
            load=self.load,
        )

        # Single beam comparison (first dataset as representative)
        if self.datasets:
            dataset = self.datasets[0]
            L_h = dataset.geometry.aspect_ratio
            self.visualizer.plot_beam_comparison(
                geometry=dataset.geometry,
                material=dataset.material,
                load=dataset.load_case,
                show_data=True,
                data_x=dataset.x_disp,
                data_y=dataset.displacements,
                filename=f"beam_comparison_Lh_{L_h:.0f}.png",
            )

        # Frequency comparison for sample geometry
        mid_idx = len(self.datasets) // 2
        self.visualizer.plot_frequency_comparison(
            geometry=self.datasets[mid_idx].geometry,
            material=self.material,
        )

        # Model comparison for first pair
        if self.comparisons:
            self.visualizer.plot_model_comparison(self.comparisons[0])

        # Prior vs posterior overlay (one per model)
        if self.eb_results and self.timo_results:
            eb_priors = create_default_priors()
            timo_priors = create_timoshenko_priors()
            L_h = self.aspect_ratios[0]

            self.visualizer.plot_prior_posterior_comparison(
                self.eb_results[0],
                eb_priors,
                filename=f"prior_posterior_eb_Lh_{L_h:.0f}.png",
            )
            self.visualizer.plot_prior_posterior_comparison(
                self.timo_results[0],
                timo_priors,
                filename=f"prior_posterior_timo_Lh_{L_h:.0f}.png",
            )

    def generate_report(self) -> dict:
        """
        Generate all reports.

        Returns:
            Dictionary with report paths

        """
        logger.info("Generating reports...")

        reports = {}

        if self.study_results:
            # Text summary
            self.reporter.generate_study_summary(
                self.study_results,
                filename="study_summary.txt",
            )
            reports["summary"] = self.output_dir / "reports" / "study_summary.txt"

            # JSON export
            self.reporter.export_results_json(
                self.study_results,
                filename="results.json",
            )
            reports["json"] = self.output_dir / "reports" / "results.json"

            # CSV export
            self.reporter.export_results_csv(
                self.study_results,
                filename="results.csv",
            )
            reports["csv"] = self.output_dir / "reports" / "results.csv"

        # Individual calibration reports
        if self.eb_results:
            for i, result in enumerate(self.eb_results[:3]):  # First 3
                self.reporter.generate_calibration_report(
                    result,
                    filename=f"calibration_eb_{i}.txt",
                )

        # Frequency analysis report
        if self.frequency_results:
            self.reporter.generate_frequency_report(
                self.frequency_results,
                filename="frequency_analysis.txt",
            )

        console.print(f"  Reports saved to {self.output_dir / 'reports'}")

        return reports

    def print_summary(self, results: dict) -> None:
        """
        Print a summary table of results.

        Args:
            results: Pipeline results dictionary
        """
        if not self.study_results:
            console.print("[yellow]No results to summarize[/yellow]")
            return

        # Create summary table
        table = Table(title="Model Selection Summary")
        table.add_column("Aspect Ratio (L/h)", style="cyan")
        table.add_column("Log Bayes Factor", style="magenta")
        table.add_column("Recommended Model", style="green")

        for L_h, bf, rec in zip(
            self.study_results["aspect_ratios"],
            self.study_results["log_bayes_factors"],
            self.study_results["recommendations"],
            strict=True,
        ):
            bf_str = f"{bf:.3f}"
            table.add_row(f"{L_h:.1f}", bf_str, rec)

        console.print(table)
        console.print(
            "\n[dim]Note: Log BF > 0 favors Euler-Bernoulli, Log BF < 0 favors Timoshenko[/dim]"
        )

        # Print transition point
        transition = self.study_results.get("transition_aspect_ratio")
        if transition:
            console.print(f"\n[bold]Transition aspect ratio: L/h ≈ {transition:.1f}[/bold]")

        # Print key guideline
        guidelines = self.study_results.get("guidelines", {})
        if "digital_twin_recommendation" in guidelines:
            console.print("\n[bold cyan]Digital Twin Recommendation:[/bold cyan]")
            console.print(guidelines["digital_twin_recommendation"])
