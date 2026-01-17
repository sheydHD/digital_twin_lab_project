"""
Pipeline Orchestrator.

Coordinates the full Bayesian model selection workflow:
1. Synthetic data generation
2. Bayesian calibration of beam models
3. Model comparison and selection
4. Analysis and reporting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from rich.console import Console
from rich.table import Table

from apps.models.base_beam import BeamGeometry, MaterialProperties, LoadCase
from apps.models.euler_bernoulli import EulerBernoulliBeam
from apps.models.timoshenko import TimoshenkoBeam
from apps.fem.cantilever_fem import CantileverFEM
from apps.data.synthetic_generator import (
    SyntheticDataGenerator,
    SensorConfiguration,
    NoiseModel,
    SyntheticDataset,
    save_dataset,
    load_dataset,
)
from apps.bayesian.calibration import (
    EulerBernoulliCalibrator,
    TimoshenkoCalibrator,
    CalibrationResult,
    create_default_priors,
    create_timoshenko_priors,
)
from apps.bayesian.model_selection import (
    BayesianModelSelector,
    ModelComparisonResult,
)
from apps.analysis.visualization import BeamVisualization
from apps.analysis.reporter import ResultsReporter


logger = logging.getLogger(__name__)
console = Console()


class PipelineOrchestrator:
    """
    Orchestrate the complete Bayesian model selection pipeline.

    This class coordinates all stages of the analysis:
    - Data generation using FEM
    - Bayesian calibration of competing models
    - Model selection using Bayes factors
    - Result visualization and reporting

    """

    def __init__(self, config: Dict[str, Any]):
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
        self.datasets: List[SyntheticDataset] = []
        self.eb_results: List[CalibrationResult] = []
        self.timo_results: List[CalibrationResult] = []
        self.comparisons: List[ModelComparisonResult] = []
        self.study_results: Optional[Dict] = None

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
        self.aspect_ratios = beam_cfg.get(
            "aspect_ratios",
            [5, 8, 10, 12, 15, 20, 30, 50]
        )
        self.base_length = beam_cfg.get("length", 1.0)
        self.base_width = beam_cfg.get("width", 0.1)

        # Bayesian sampling parameters
        self.n_samples = bayesian_cfg.get("n_samples", 2000)
        self.n_tune = bayesian_cfg.get("n_tune", 1000)
        self.n_chains = bayesian_cfg.get("n_chains", 4)

        # Visualization and reporting
        self.visualizer = BeamVisualization(self.output_dir / "figures")
        self.reporter = ResultsReporter(self.output_dir / "reports")

    def run_full_pipeline(self) -> Dict:
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

        # Stage 4: Generate reports
        console.print("\n[cyan]Stage 4: Generating reports...[/cyan]")
        self.generate_report()

        return {
            "datasets": self.datasets,
            "eb_results": self.eb_results,
            "timo_results": self.timo_results,
            "study_results": self.study_results,
        }

    def run_data_generation(self) -> List[SyntheticDataset]:
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

        for i, dataset in enumerate(self.datasets):
            L_h = dataset.geometry.aspect_ratio
            save_dataset(dataset, data_dir / f"dataset_Lh_{L_h:.0f}.h5")

        console.print(f"  Generated {len(self.datasets)} datasets")

        return self.datasets

    def run_calibration(self) -> Dict:
        """
        Run Bayesian calibration for both beam theories.

        Returns:
            Dictionary with calibration results

        """
        logger.info("Running Bayesian calibration...")

        if not self.datasets:
            console.print("[yellow]No datasets found. Running data generation...[/yellow]")
            self.run_data_generation()

        # Create calibrators
        eb_priors = create_default_priors()
        timo_priors = create_timoshenko_priors()

        self.eb_results = []
        self.timo_results = []

        for i, dataset in enumerate(self.datasets):
            L_h = dataset.geometry.aspect_ratio
            console.print(f"\n  Calibrating for L/h = {L_h:.1f} ({i+1}/{len(self.datasets)})")

            # Get target_accept from config (default 0.95 for stability)
            target_accept = self.config.get("bayesian", {}).get("target_accept", 0.95)
            
            # Euler-Bernoulli calibration
            eb_calibrator = EulerBernoulliCalibrator(
                priors=eb_priors,
                n_samples=self.n_samples,
                n_tune=self.n_tune,
                n_chains=self.n_chains,
                target_accept=target_accept,
            )
            eb_result = eb_calibrator.calibrate(dataset)
            eb_result.marginal_likelihood_estimate = eb_calibrator.compute_marginal_likelihood()
            self.eb_results.append(eb_result)

            # Timoshenko calibration
            timo_calibrator = TimoshenkoCalibrator(
                priors=timo_priors,
                n_samples=self.n_samples,
                n_tune=self.n_tune,
                n_chains=self.n_chains,
                target_accept=target_accept,
            )
            timo_result = timo_calibrator.calibrate(dataset)
            timo_result.marginal_likelihood_estimate = timo_calibrator.compute_marginal_likelihood()
            self.timo_results.append(timo_result)

        console.print(f"\n  Calibrated {len(self.eb_results)} model pairs")

        return {
            "euler_bernoulli": self.eb_results,
            "timoshenko": self.timo_results,
        }

    def run_analysis(self) -> Dict:
        """
        Analyze model selection results.

        Returns:
            Study results dictionary

        """
        logger.info("Analyzing model selection...")

        if not self.eb_results or not self.timo_results:
            raise ValueError("Must run calibration before analysis")

        # Model selection
        selector = BayesianModelSelector()

        self.study_results = selector.analyze_aspect_ratio_study(
            eb_results=self.eb_results,
            timo_results=self.timo_results,
            aspect_ratios=self.aspect_ratios,
        )

        # Generate visualizations
        self._generate_visualizations()

        return self.study_results

    def _generate_visualizations(self) -> None:
        """Generate all visualization outputs."""
        if not self.study_results:
            return

        console.print("  Generating visualizations...")

        # Aspect ratio study plot
        self.visualizer.plot_aspect_ratio_study(self.study_results)

        # Deflection error plot
        self.visualizer.plot_deflection_error(
            aspect_ratios=self.aspect_ratios,
            base_length=self.base_length,
            material=self.material,
            load=self.load,
        )

        # Individual beam comparisons
        for dataset in self.datasets[:3]:  # First 3 only to save time
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

        # Summary report figure
        self.visualizer.create_summary_report(self.study_results)

    def generate_report(self) -> Dict:
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

        console.print(f"  Reports saved to {self.output_dir / 'reports'}")

        return reports

    def print_summary(self, results: Dict) -> None:
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
        ):
            bf_str = f"{bf:.3f}"
            table.add_row(f"{L_h:.1f}", bf_str, rec)

        console.print(table)
        console.print("\n[dim]Note: Log BF > 0 favors Euler-Bernoulli, Log BF < 0 favors Timoshenko[/dim]")

        # Print transition point
        transition = self.study_results.get("transition_aspect_ratio")
        if transition:
            console.print(
                f"\n[bold]Transition aspect ratio: L/h ≈ {transition:.1f}[/bold]"
            )

        # Print key guideline
        guidelines = self.study_results.get("guidelines", {})
        if "digital_twin_recommendation" in guidelines:
            console.print(f"\n[bold cyan]Digital Twin Recommendation:[/bold cyan]")
            console.print(guidelines["digital_twin_recommendation"])

    def load_previous_results(self, results_dir: Path) -> None:
        """
        Load results from a previous run.

        Args:
            results_dir: Directory containing previous results

        """
        raise NotImplementedError("Results loading not yet implemented")
