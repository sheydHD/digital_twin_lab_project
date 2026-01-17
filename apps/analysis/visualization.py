"""
Visualization Module for Model Selection Analysis.

Provides plotting functions for:
- Beam deflection comparisons
- Bayesian posterior distributions
- Model selection results (Bayes factors, WAIC)
- Aspect ratio study visualizations
- Practical guideline summaries
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import arviz as az

from apps.models.base_beam import BeamGeometry, MaterialProperties, LoadCase
from apps.models.euler_bernoulli import EulerBernoulliBeam
from apps.models.timoshenko import TimoshenkoBeam
from apps.bayesian.calibration import CalibrationResult
from apps.bayesian.model_selection import ModelComparisonResult


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class BeamVisualization:
    """
    Visualization tools for beam analysis results.

    TODO: Task 25.1 - Complete all visualization functions
    TODO: Task 25.2 - Add interactive plots with plotly
    TODO: Task 25.3 - Generate publication-quality figures
    """

    def __init__(self, output_dir: Path = Path("outputs/figures")):
        """
        Initialize visualization module.

        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Figure defaults
        self.figsize = (10, 6)
        self.dpi = 150

    def plot_beam_comparison(
        self,
        geometry: BeamGeometry,
        material: MaterialProperties,
        load: LoadCase,
        n_points: int = 100,
        show_data: bool = False,
        data_x: Optional[np.ndarray] = None,
        data_y: Optional[np.ndarray] = None,
        save: bool = True,
        filename: str = "beam_comparison.png",
    ) -> plt.Figure:
        """
        Plot deflection comparison between Euler-Bernoulli and Timoshenko.

        Args:
            geometry: Beam geometry
            material: Material properties
            load: Load case
            n_points: Number of points along beam
            show_data: Whether to show synthetic data points
            data_x: X-coordinates of data points
            data_y: Y-coordinates of data points
            save: Whether to save figure
            filename: Output filename

        Returns:
            Matplotlib figure

        TODO: Task 25.4 - Add error bands from Bayesian posterior
        """
        # Create beam models
        eb_beam = EulerBernoulliBeam(geometry, material)
        timo_beam = TimoshenkoBeam(geometry, material)

        # Compute deflections
        x = np.linspace(0, geometry.length, n_points)
        w_eb = eb_beam.compute_deflection(x, load)
        w_timo = timo_beam.compute_deflection(x, load)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot deflections
        ax.plot(x * 1000, w_eb * 1000, 'b-', linewidth=2, label='Euler-Bernoulli')
        ax.plot(x * 1000, w_timo * 1000, 'r--', linewidth=2, label='Timoshenko')

        # Plot data points if provided
        if show_data and data_x is not None and data_y is not None:
            ax.scatter(data_x * 1000, data_y * 1000, c='k', s=50, marker='o',
                      label='Measurements', zorder=5)

        # Labels and formatting
        ax.set_xlabel('Position along beam [mm]', fontsize=12)
        ax.set_ylabel('Deflection [mm]', fontsize=12)
        ax.set_title(
            f'Beam Deflection Comparison (L/h = {geometry.aspect_ratio:.1f})',
            fontsize=14
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add info text
        info_text = (
            f'L = {geometry.length*1000:.1f} mm\n'
            f'h = {geometry.height*1000:.1f} mm\n'
            f'E = {material.elastic_modulus/1e9:.0f} GPa\n'
            f'P = {load.point_load:.0f} N'
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_deflection_error(
        self,
        aspect_ratios: List[float],
        base_length: float,
        material: MaterialProperties,
        load: LoadCase,
        save: bool = True,
        filename: str = "deflection_error.png",
    ) -> plt.Figure:
        """
        Plot relative deflection error between theories vs aspect ratio.

        Args:
            aspect_ratios: List of L/h ratios to analyze
            base_length: Reference length
            material: Material properties
            load: Load case
            save: Whether to save figure
            filename: Output filename

        Returns:
            Matplotlib figure

        TODO: Task 25.5 - Add theoretical prediction for error
        """
        tip_errors = []
        shear_ratios = []

        for L_h in aspect_ratios:
            h = base_length / L_h
            geometry = BeamGeometry(length=base_length, height=h, width=0.1)

            eb = EulerBernoulliBeam(geometry, material)
            timo = TimoshenkoBeam(geometry, material)

            w_eb = eb.tip_deflection(load)
            w_timo = timo.tip_deflection(load)

            error = (w_timo - w_eb) / w_timo * 100
            tip_errors.append(error)

            shear_ratio = timo.shear_deformation_ratio(load) * 100
            shear_ratios.append(shear_ratio)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Tip deflection error
        ax1.semilogy(aspect_ratios, tip_errors, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=5, color='r', linestyle='--', label='5% threshold')
        ax1.axhline(y=1, color='g', linestyle='--', label='1% threshold')
        ax1.set_xlabel('Aspect Ratio (L/h)', fontsize=12)
        ax1.set_ylabel('Relative Error [%]', fontsize=12)
        ax1.set_title('Euler-Bernoulli Error vs Timoshenko', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Shear deformation contribution
        ax2.plot(aspect_ratios, shear_ratios, 'rs-', linewidth=2, markersize=8)
        ax2.set_xlabel('Aspect Ratio (L/h)', fontsize=12)
        ax2.set_ylabel('Shear Deflection / Total Deflection [%]', fontsize=12)
        ax2.set_title('Shear Deformation Contribution', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_posterior_distributions(
        self,
        result: CalibrationResult,
        params_to_plot: Optional[List[str]] = None,
        save: bool = True,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot posterior distributions from Bayesian calibration.

        Args:
            result: Calibration result with trace
            params_to_plot: Parameter names to plot (None = all)
            save: Whether to save figure
            filename: Output filename

        Returns:
            Matplotlib figure

        TODO: Task 26.1 - Add prior overlays
        TODO: Task 26.2 - Add true value markers for synthetic data
        """
        if filename is None:
            filename = f"posterior_{result.model_name.lower().replace('-', '_')}.png"

        # Use ArviZ for posterior plotting
        var_names = params_to_plot if params_to_plot else None

        axes = az.plot_posterior(
            result.trace,
            var_names=var_names,
            figsize=(12, 4),
        )

        fig = plt.gcf()
        fig.suptitle(f'Posterior Distributions: {result.model_name}', fontsize=14, y=1.02)

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_trace(
        self,
        result: CalibrationResult,
        params_to_plot: Optional[List[str]] = None,
        save: bool = True,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot MCMC trace plots for convergence diagnostics.

        Args:
            result: Calibration result
            params_to_plot: Parameters to plot
            save: Whether to save
            filename: Output filename

        Returns:
            Matplotlib figure

        TODO: Task 26.3 - Add divergence markers
        """
        if filename is None:
            filename = f"trace_{result.model_name.lower().replace('-', '_')}.png"

        var_names = params_to_plot if params_to_plot else None

        axes = az.plot_trace(
            result.trace,
            var_names=var_names,
            figsize=(12, 8),
        )

        fig = plt.gcf()
        fig.suptitle(f'MCMC Traces: {result.model_name}', fontsize=14, y=1.02)

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_model_comparison(
        self,
        comparison: ModelComparisonResult,
        save: bool = True,
        filename: str = "model_comparison.png",
    ) -> plt.Figure:
        """
        Plot model comparison results.

        Args:
            comparison: Model comparison result
            save: Whether to save
            filename: Output filename

        Returns:
            Matplotlib figure

        TODO: Task 27.1 - Add WAIC/LOO comparison bars
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Posterior model probabilities
        models = [comparison.model1_name, comparison.model2_name]
        probs = [comparison.model1_probability, comparison.model2_probability]
        colors = ['steelblue', 'coral']

        bars = ax1.bar(models, probs, color=colors, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Posterior Probability', fontsize=12)
        ax1.set_title('Model Posterior Probabilities', fontsize=14)
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Add probability labels on bars
        for bar, prob in zip(bars, probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob:.2%}', ha='center', va='bottom', fontsize=11)

        # Evidence interpretation
        ax2.axis('off')
        info_text = (
            f"Model Comparison Summary\n"
            f"{'='*40}\n\n"
            f"Log Bayes Factor: {comparison.log_bayes_factor:.2f}\n"
            f"Bayes Factor: {comparison.bayes_factor:.2f}\n\n"
            f"Evidence: {comparison.evidence_interpretation.value}\n\n"
            f"Recommended Model: {comparison.recommended_model}\n"
        )

        if comparison.waic_difference is not None:
            info_text += f"\nΔWAIC: {comparison.waic_difference:.2f}"
        if comparison.loo_difference is not None:
            info_text += f"\nΔLOO: {comparison.loo_difference:.2f}"

        ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_aspect_ratio_study(
        self,
        study_results: Dict,
        save: bool = True,
        filename: str = "aspect_ratio_study.png",
    ) -> plt.Figure:
        """
        Plot model selection results across aspect ratios.

        Args:
            study_results: Results from BayesianModelSelector.analyze_aspect_ratio_study
            save: Whether to save
            filename: Output filename

        Returns:
            Matplotlib figure

        TODO: Task 27.2 - Add confidence intervals
        TODO: Task 27.3 - Mark transition region
        """
        aspect_ratios = study_results["aspect_ratios"]
        log_bfs = study_results["log_bayes_factors"]
        transition = study_results.get("transition_aspect_ratio")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Log Bayes Factor vs Aspect Ratio
        ax1.plot(aspect_ratios, log_bfs, 'bo-', linewidth=2, markersize=10)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)

        # Shade regions
        ax1.axhspan(np.log(3), np.log(10), alpha=0.2, color='blue', label='Weak EB')
        ax1.axhspan(np.log(10), max(log_bfs) + 1, alpha=0.2, color='blue')
        ax1.axhspan(-np.log(3), -np.log(10), alpha=0.2, color='red', label='Weak Timo')
        ax1.axhspan(-np.log(10), min(log_bfs) - 1, alpha=0.2, color='red')

        if transition:
            ax1.axvline(x=transition, color='green', linestyle='--',
                       linewidth=2, label=f'Transition: L/h={transition:.1f}')

        ax1.set_xlabel('Aspect Ratio (L/h)', fontsize=12)
        ax1.set_ylabel('Log Bayes Factor', fontsize=12)
        ax1.set_title('Model Evidence vs Beam Slenderness', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Interpretation text
        ax2.axis('off')

        guidelines = study_results.get("guidelines", {})
        guide_text = "Practical Guidelines\n" + "="*40 + "\n\n"
        for key, value in guidelines.items():
            guide_text += f"{value}\n\n"

        ax2.text(0.05, 0.95, guide_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                family='serif',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_frequency_comparison(
        self,
        geometry: BeamGeometry,
        material: MaterialProperties,
        n_modes: int = 5,
        save: bool = True,
        filename: str = "frequency_comparison.png",
    ) -> plt.Figure:
        """
        Plot natural frequency comparison between theories.

        Args:
            geometry: Beam geometry
            material: Material properties
            n_modes: Number of modes to plot
            save: Whether to save
            filename: Output filename

        Returns:
            Matplotlib figure

        TODO: Task 28.1 - Add percentage difference plot
        """
        eb_beam = EulerBernoulliBeam(geometry, material)
        timo_beam = TimoshenkoBeam(geometry, material)

        freq_eb = eb_beam.compute_natural_frequencies(n_modes)
        freq_timo = timo_beam.compute_natural_frequencies(n_modes)

        modes = np.arange(1, n_modes + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Frequency comparison
        width = 0.35
        ax1.bar(modes - width/2, freq_eb, width, label='Euler-Bernoulli', color='steelblue')
        ax1.bar(modes + width/2, freq_timo, width, label='Timoshenko', color='coral')
        ax1.set_xlabel('Mode Number', fontsize=12)
        ax1.set_ylabel('Natural Frequency [Hz]', fontsize=12)
        ax1.set_title(f'Natural Frequencies (L/h = {geometry.aspect_ratio:.1f})', fontsize=14)
        ax1.legend()
        ax1.set_xticks(modes)

        # Frequency ratio
        freq_ratio = freq_timo / freq_eb
        ax2.plot(modes, freq_ratio * 100, 'go-', linewidth=2, markersize=10)
        ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Mode Number', fontsize=12)
        ax2.set_ylabel('Timoshenko / Euler-Bernoulli [%]', fontsize=12)
        ax2.set_title('Frequency Ratio', fontsize=14)
        ax2.set_xticks(modes)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')

        return fig

    def create_summary_report(
        self,
        study_results: Dict,
        save: bool = True,
        filename: str = "summary_report.png",
    ) -> plt.Figure:
        """
        Create comprehensive summary report figure.

        Args:
            study_results: Complete study results
            save: Whether to save
            filename: Output filename

        Returns:
            Matplotlib figure

        TODO: Task 28.2 - Design comprehensive report layout
        TODO: Task 28.3 - Add key findings highlights
        """
        fig = plt.figure(figsize=(16, 12))

        # Use gridspec for complex layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Log BF vs aspect ratio
        ax1 = fig.add_subplot(gs[0, :2])
        aspect_ratios = study_results["aspect_ratios"]
        log_bfs = study_results["log_bayes_factors"]
        ax1.plot(aspect_ratios, log_bfs, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=0, color='k', linestyle='-')
        ax1.set_xlabel('Aspect Ratio (L/h)')
        ax1.set_ylabel('Log Bayes Factor')
        ax1.set_title('Model Selection Across Aspect Ratios')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Model probabilities
        ax2 = fig.add_subplot(gs[0, 2])
        # Aggregate model probabilities
        eb_probs = [c.model1_probability for c in study_results["comparisons"]]
        timo_probs = [c.model2_probability for c in study_results["comparisons"]]
        ax2.fill_between(aspect_ratios, 0, eb_probs, alpha=0.5, label='Euler-Bernoulli')
        ax2.fill_between(aspect_ratios, eb_probs, 1, alpha=0.5, label='Timoshenko')
        ax2.set_xlabel('Aspect Ratio (L/h)')
        ax2.set_ylabel('Probability')
        ax2.set_title('Model Probabilities')
        ax2.legend()

        # Text: Guidelines
        ax3 = fig.add_subplot(gs[1:, :])
        ax3.axis('off')

        guidelines = study_results.get("guidelines", {})
        transition = study_results.get("transition_aspect_ratio", "N/A")

        summary_text = f"""
        BAYESIAN MODEL SELECTION SUMMARY REPORT
        {'='*60}

        Study Parameters:
        - Aspect ratios analyzed: {min(aspect_ratios):.1f} to {max(aspect_ratios):.1f}
        - Number of configurations: {len(aspect_ratios)}
        - Transition aspect ratio: {transition if transition else 'Not found'}

        Key Findings:
        {guidelines.get('transition_rule', 'N/A')}

        Recommendations:
        {guidelines.get('slender_beams', 'N/A')}

        {guidelines.get('thick_beams', 'N/A')}

        Digital Twin Guidelines:
        {guidelines.get('digital_twin_recommendation', 'N/A')}
        """

        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

        plt.suptitle('Bayesian Model Selection for Beam Theory', fontsize=16, fontweight='bold')

        if save:
            fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')

        return fig
