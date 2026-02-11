"""
Visualization Module for Model Selection Analysis.

Provides plotting functions for:
- Beam deflection comparisons
- Bayesian posterior distributions
- Model selection results (Bayes factors, WAIC)
- Aspect ratio study visualizations
- Practical guideline summaries
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from apps.bayesian.calibration import CalibrationResult, PriorConfig
from apps.bayesian.model_selection import ModelComparisonResult
from apps.models.base_beam import BeamGeometry, LoadCase, MaterialProperties
from apps.models.euler_bernoulli import EulerBernoulliBeam
from apps.models.timoshenko import TimoshenkoBeam

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class BeamVisualization:
    """
    Visualization tools for beam analysis results.

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
               bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

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

    def plot_prior_posterior_comparison(
        self,
        result: CalibrationResult,
        priors: List[PriorConfig],
        save: bool = True,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot prior and posterior distributions overlaid for comparison.

        Shows how the data updated our beliefs: narrow posteriors relative
        to priors indicate informative data.

        Args:
            result: Calibration result with posterior trace
            priors: Prior configurations used
            save: Whether to save figure
            filename: Output filename

        Returns:
            Matplotlib figure
        """
        from scipy import stats

        if filename is None:
            filename = f"prior_posterior_{result.model_name.lower().replace('-', '_')}.png"

        # Determine which parameters to plot
        param_names = [p.param_name for p in priors]
        n_params = len(param_names)

        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4.5))
        if n_params == 1:
            axes = [axes]

        prior_color = '#E57373'   # red-ish for prior
        post_color = '#42A5F5'    # blue for posterior

        for ax, prior in zip(axes, priors, strict=False):
            name = prior.param_name

            # Get posterior samples
            if name in result.trace.posterior:
                post_samples = result.trace.posterior[name].values.flatten()
            else:
                continue

            # Plot posterior as KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(post_samples)
            x_min, x_max = post_samples.min(), post_samples.max()
            margin = (x_max - x_min) * 0.3
            x_post = np.linspace(x_min - margin, x_max + margin, 300)
            post_pdf = kde(x_post)

            # Plot prior
            if name == "elastic_modulus":
                sigma = prior.params.get("sigma", 0.05)
                x_prior = np.linspace(1.0 - 4 * sigma, 1.0 + 4 * sigma, 300)
                prior_pdf = stats.norm.pdf(x_prior, loc=1.0, scale=sigma)
                xlabel = "E (normalized)"
                true_val = 1.0
            elif name == "sigma":
                x_prior = np.linspace(0, max(4.0, x_max + margin), 300)
                prior_pdf = stats.halfnorm.pdf(x_prior, scale=1.0)
                xlabel = "σ (normalized)"
                true_val = None
            elif name == "poisson_ratio":
                mu = prior.params.get("mu", 0.3)
                sigma = prior.params.get("sigma", 0.03)
                x_prior = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
                prior_pdf = stats.norm.pdf(x_prior, loc=mu, scale=sigma)
                xlabel = "ν (Poisson's ratio)"
                true_val = 0.3
            else:
                continue

            # Normalize both to same visual scale for comparison
            prior_max = prior_pdf.max()
            post_max = post_pdf.max()
            scale = prior_max / post_max if post_max > 0 else 1.0

            ax.fill_between(x_prior, prior_pdf, alpha=0.2, color=prior_color)
            ax.plot(x_prior, prior_pdf, color=prior_color, linewidth=2,
                    linestyle='--', label='Prior')

            ax.fill_between(x_post, post_pdf * scale, alpha=0.3, color=post_color)
            ax.plot(x_post, post_pdf * scale, color=post_color, linewidth=2,
                    label='Posterior')

            if true_val is not None:
                ax.axvline(true_val, color='#2E7D32', linestyle=':',
                          linewidth=1.5, label=f'True = {true_val}')

            # Add posterior summary text
            post_mean = float(np.mean(post_samples))
            post_std = float(np.std(post_samples))
            ax.text(0.97, 0.97,
                    f"μ = {post_mean:.4f}\nσ = {post_std:.4f}",
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})

            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel("Density (scaled)", fontsize=11)
            ax.set_title(f"{name}", fontsize=12)
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Prior vs Posterior: {result.model_name}",
            fontsize=14, y=1.02
        )
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
        for bar, prob in zip(bars, probs, strict=False):
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

        ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='center',
                family='monospace',
                bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8})

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

        """
        aspect_ratios = study_results["aspect_ratios"]
        log_bfs = study_results["log_bayes_factors"]
        transition = study_results.get("transition_aspect_ratio")

        fig, ax1 = plt.subplots(figsize=(10, 6))

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
