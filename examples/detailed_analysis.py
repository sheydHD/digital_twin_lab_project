#!/usr/bin/env python3
"""
Analysis and Visualization for Bayesian Model Selection.

This script generates visualizations for the Bayesian model selection study
comparing Euler-Bernoulli and Timoshenko beam theories.

Generates:
1. Shear contribution analysis
2. Evidence strength bars
3. Transition analysis
4. Frequency mode analysis
5. Data fit comparison
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open("outputs/reports/results.json", "r") as f:
    results = json.load(f)


def plot_shear_contribution_analysis():
    """
    Detailed analysis of shear deformation contribution vs aspect ratio.
    """
    # Beam parameters
    E = 210e9  # Pa
    nu = 0.3

    aspect_ratios = np.linspace(3, 120, 200)

    # Shear deflection ratio: w_shear / w_total = 1 / (1 + 3(L/h)^2 * k*G/(5E))
    # For rectangular section: w_shear/w_bending ≈ 3*(1+ν)*(h/L)^2
    shear_ratio = 3 * (1 + nu) / aspect_ratios**2 * 100  # percentage

    # Euler-Bernoulli error relative to Timoshenko
    eb_error = shear_ratio / (1 + shear_ratio/100) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Shear contribution
    ax1 = axes[0]
    ax1.semilogy(aspect_ratios, shear_ratio, 'b-', linewidth=2.5, label='Shear/Bending Ratio')
    ax1.axhline(5, color='orange', linestyle='--', linewidth=2, label='5% Engineering Threshold')
    ax1.axhline(1, color='green', linestyle='--', linewidth=2, label='1% High Precision')
    ax1.axvline(results["transition_aspect_ratio"], color='red', linestyle=':',
               linewidth=2, label=f'Bayesian Transition (L/h={results["transition_aspect_ratio"]:.1f})')

    # Shade regions
    ax1.axvspan(3, 10, alpha=0.2, color='red', label='Timoshenko Essential')
    ax1.axvspan(10, 20, alpha=0.2, color='yellow', label='Transition Zone')
    ax1.axvspan(20, 120, alpha=0.2, color='green', label='E-B Adequate')

    ax1.set_xlabel('Aspect Ratio (L/h)', fontsize=12)
    ax1.set_ylabel('Shear Deformation Contribution [%]', fontsize=12)
    ax1.set_title('Shear Deformation vs Beam Slenderness', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(3, 120)
    ax1.set_ylim(0.01, 100)

    # Right: E-B error
    ax2 = axes[1]
    ax2.plot(aspect_ratios, eb_error, 'r-', linewidth=2.5)
    ax2.axhline(5, color='orange', linestyle='--', linewidth=2)
    ax2.axhline(1, color='green', linestyle='--', linewidth=2)

    # Mark our study points
    study_ratios = results["aspect_ratios"]
    study_errors = 3 * (1 + nu) / np.array(study_ratios)**2 * 100
    ax2.scatter(study_ratios, study_errors, s=100, c='blue', edgecolors='black',
               zorder=5, label='Study Points')

    ax2.set_xlabel('Aspect Ratio (L/h)', fontsize=12)
    ax2.set_ylabel('Euler-Bernoulli Error [%]', fontsize=12)
    ax2.set_title('Theoretical Error in Euler-Bernoulli Approximation', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(3, 120)
    ax2.set_yscale('log')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "shear_contribution_analysis.png")
    plt.close()
    print("✓ Created shear_contribution_analysis.png")


def plot_evidence_strength_bars():
    """
    Create bar chart showing evidence strength for each aspect ratio.
    """
    aspect_ratios = results["aspect_ratios"]
    log_bfs = results["log_bayes_factors"]
    recommendations = results["recommendations"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(aspect_ratios))
    colors = ['coral' if r == 'Timoshenko' else 'steelblue' for r in recommendations]

    bars = ax.bar(x, log_bfs, color=colors, edgecolor='black', linewidth=1.5)

    # Add evidence threshold lines
    ax.axhline(0, color='black', linestyle='-', linewidth=2)
    ax.axhline(np.log(3), color='gray', linestyle='--', alpha=0.7, label='Substantial (BF=3)')
    ax.axhline(-np.log(3), color='gray', linestyle='--', alpha=0.7)
    ax.axhline(np.log(10), color='gray', linestyle=':', alpha=0.7, label='Strong (BF=10)')
    ax.axhline(-np.log(10), color='gray', linestyle=':', alpha=0.7)

    # Add value labels
    for bar, lbf in zip(bars, log_bfs, strict=False):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.1 if height >= 0 else -0.1
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
               f'{lbf:.2f}', ha='center', va=va, fontsize=9, fontweight='bold')

    ax.set_xlabel('Aspect Ratio (L/h)', fontsize=12)
    ax.set_ylabel('Log Bayes Factor', fontsize=12)
    ax.set_title('Evidence Strength for Model Selection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.0f}' for r in aspect_ratios])
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)

    # Add annotation
    ax.annotate('← Favors Timoshenko | Favors E-B →',
               xy=(0.5, 0.02), xycoords='axes fraction',
               ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "evidence_strength_bars.png")
    plt.close()
    print("✓ Created evidence_strength_bars.png")


def plot_transition_analysis():
    """
    Detailed analysis around the transition aspect ratio.
    """
    aspect_ratios = np.array(results["aspect_ratios"])
    log_bfs = np.array(results["log_bayes_factors"])
    transition = results["transition_aspect_ratio"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Log BF with transition
    ax1 = axes[0]
    ax1.plot(aspect_ratios, log_bfs, 'ko-', markersize=10, linewidth=2)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1.5)
    ax1.axvline(transition, color='red', linestyle='-', linewidth=2,
               label=f'Transition: L/h = {transition:.1f}')

    # Fit interpolation
    from scipy.interpolate import interp1d
    f = interp1d(aspect_ratios, log_bfs, kind='cubic')
    x_fine = np.linspace(min(aspect_ratios), max(aspect_ratios), 200)
    y_fine = f(x_fine)
    ax1.plot(x_fine, y_fine, 'b-', alpha=0.5, linewidth=1)

    ax1.fill_between(aspect_ratios, log_bfs, 0, where=np.array(log_bfs) < 0,
                    alpha=0.3, color='coral', label='Timoshenko Region')
    ax1.fill_between(aspect_ratios, log_bfs, 0, where=np.array(log_bfs) >= 0,
                    alpha=0.3, color='steelblue', label='E-B Region')

    ax1.set_xlabel('Aspect Ratio (L/h)')
    ax1.set_ylabel('Log Bayes Factor')
    ax1.set_title('Transition Identification', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Middle: Zoomed view around transition
    ax2 = axes[1]
    mask = (aspect_ratios >= 10) & (aspect_ratios <= 30)
    ax2.plot(aspect_ratios[mask], log_bfs[mask], 'ko-', markersize=12, linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1.5)
    ax2.axvline(transition, color='red', linestyle='-', linewidth=2)
    ax2.set_xlabel('Aspect Ratio (L/h)')
    ax2.set_ylabel('Log Bayes Factor')
    ax2.set_title('Zoomed: Transition Region', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(10, 30)

    # Right: Evidence interpretation
    ax3 = axes[2]
    ax3.axis('off')

    text = f"""
    TRANSITION ANALYSIS
    {'='*40}

    Bayesian Transition Point: L/h ≈ {transition:.1f}

    Evidence Interpretation:

    L/h = 5:   Log BF = {log_bfs[0]:.2f}
               → Decisive for Timoshenko

    L/h = 15:  Log BF = {log_bfs[4]:.2f}
               → Substantial for Timoshenko

    L/h = 20:  Log BF = {log_bfs[5]:.2f}
               → Inconclusive (near zero)

    L/h = 50:  Log BF = {log_bfs[7]:.2f}
               → Slight preference for E-B

    Classical Threshold: L/h ≈ 10-20
    (matches Bayesian result!)
    """

    ax3.text(0.05, 0.95, text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox={'boxstyle': 'round', 'facecolor': 'lightyellow', 'edgecolor': 'gray'})

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "transition_analysis.png")
    plt.close()
    print("✓ Created transition_analysis.png")


def plot_frequency_mode_analysis():
    """
    Analyze natural frequency differences across modes and aspect ratios.
    """
    from apps.models.base_beam import BeamGeometry, MaterialProperties
    from apps.models.euler_bernoulli import EulerBernoulliBeam
    from apps.models.timoshenko import TimoshenkoBeam

    L = 1.0
    E = 210e9
    nu = 0.3
    rho = 7850

    aspect_ratios = [5, 10, 20, 50]
    n_modes = 5

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, L_h in enumerate(aspect_ratios):
        ax = axes.flatten()[idx]
        h = L / L_h

        geometry = BeamGeometry(length=L, height=h, width=0.05)
        material = MaterialProperties(elastic_modulus=E, poisson_ratio=nu, density=rho)

        eb_beam = EulerBernoulliBeam(geometry, material)
        timo_beam = TimoshenkoBeam(geometry, material)

        freq_eb = eb_beam.compute_natural_frequencies(n_modes)
        freq_timo = timo_beam.compute_natural_frequencies(n_modes)

        modes = np.arange(1, n_modes + 1)
        ratio = freq_timo / freq_eb * 100

        ax.bar(modes - 0.2, freq_eb, 0.4, label='E-B', color='steelblue', edgecolor='black')
        ax.bar(modes + 0.2, freq_timo, 0.4, label='Timoshenko', color='coral', edgecolor='black')

        ax2 = ax.twinx()
        ax2.plot(modes, ratio, 'go-', markersize=8, linewidth=2, label='Ratio (%)')
        ax2.axhline(100, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Timo/E-B Ratio [%]', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(50, 105)

        ax.set_xlabel('Mode Number')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(f'L/h = {L_h}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.set_xticks(modes)

    plt.suptitle('Natural Frequency Comparison: Higher Modes Show Larger Differences',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "frequency_mode_analysis.png")
    plt.close()
    print("✓ Created frequency_mode_analysis.png")


def plot_data_fit_comparison():
    """
    Show how well each model fits the synthetic data at different L/h.
    """
    from apps.models.base_beam import BeamGeometry, LoadCase, MaterialProperties
    from apps.models.euler_bernoulli import EulerBernoulliBeam
    from apps.models.timoshenko import TimoshenkoBeam

    selected_ratios = [5, 10, 20, 50]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    L = 1.0
    E = 210e9
    nu = 0.3
    P = 1000
    noise_level = 0.02  # 2% noise

    for idx, L_h in enumerate(selected_ratios):
        ax = axes.flatten()[idx]
        h = L / L_h

        geometry = BeamGeometry(length=L, height=h, width=0.05)
        material = MaterialProperties(elastic_modulus=E, poisson_ratio=nu)
        load = LoadCase(point_load=P)

        timo_beam = TimoshenkoBeam(geometry, material)
        eb_beam = EulerBernoulliBeam(geometry, material)

        # Generate "true" data from Timoshenko + noise
        x_data = np.linspace(0.1, L, 10)
        w_true = timo_beam.compute_deflection(x_data, load)
        np.random.seed(42 + idx)
        noise = np.random.normal(0, noise_level * np.abs(w_true.min()), len(x_data))
        w_data = w_true + noise

        # Model predictions
        x_pred = np.linspace(0, L, 100)
        w_eb = eb_beam.compute_deflection(x_pred, load)
        w_timo = timo_beam.compute_deflection(x_pred, load)

        # Plot
        ax.scatter(x_data * 1000, w_data * 1000, s=80, c='black', marker='o',
                  label='Data', zorder=5, edgecolors='white', linewidths=1.5)
        ax.plot(x_pred * 1000, w_eb * 1000, 'b-', linewidth=2, label='E-B fit')
        ax.plot(x_pred * 1000, w_timo * 1000, 'r--', linewidth=2, label='Timoshenko fit')

        # Calculate residuals
        w_eb_at_data = eb_beam.compute_deflection(x_data, load)
        w_timo_at_data = timo_beam.compute_deflection(x_data, load)
        rmse_eb = np.sqrt(np.mean((w_data - w_eb_at_data)**2)) * 1000
        rmse_timo = np.sqrt(np.mean((w_data - w_timo_at_data)**2)) * 1000

        rec = 'Timoshenko' if L_h < 20 else 'E-B'
        ax.set_title(f'L/h = {L_h}\nRMSE: E-B={rmse_eb:.3f}mm, Timo={rmse_timo:.3f}mm\n(Rec: {rec})',
                    fontsize=11)
        ax.set_xlabel('Position [mm]')
        ax.set_ylabel('Deflection [mm]')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Model Fit to Synthetic Data (Generated from Timoshenko + 2% Noise)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "data_fit_comparison.png")
    plt.close()
    print("✓ Created data_fit_comparison.png")


def create_all_figures():
    """Generate all visualization figures."""
    print("\n" + "="*60)
    print("Generating Analysis Figures")
    print("="*60 + "\n")

    plot_shear_contribution_analysis()
    plot_evidence_strength_bars()
    plot_transition_analysis()
    plot_frequency_mode_analysis()
    plot_data_fit_comparison()

    print("\n" + "="*60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("="*60)

    # List all files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  • {f.name}")


if __name__ == "__main__":
    create_all_figures()
