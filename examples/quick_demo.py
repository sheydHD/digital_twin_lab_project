#!/usr/bin/env python3
"""
Quick Demo: Compare Euler-Bernoulli vs Timoshenko Beam Theories

This script demonstrates the core functionality of the beam models
without running the full Bayesian pipeline. Useful for initial testing.
"""

# Add project root to path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from apps.models.base_beam import BeamGeometry, LoadCase, MaterialProperties
from apps.models.euler_bernoulli import EulerBernoulliBeam
from apps.models.timoshenko import TimoshenkoBeam


def main():
    """Run a quick comparison of beam theories."""
    print("=" * 60)
    print("Euler-Bernoulli vs Timoshenko Beam Theory Comparison")
    print("=" * 60)

    # Define material properties (structural steel)
    material = MaterialProperties(
        elastic_modulus=210e9,  # 210 GPa
        poisson_ratio=0.3,
        density=7850,
    )

    # Define load case
    load = LoadCase(point_load=1000)  # 1 kN at tip

    # Compare for different aspect ratios
    aspect_ratios = [5, 10, 15, 20, 30, 50]
    base_length = 1.0  # 1 meter

    print(f"\nBase length: {base_length} m")
    print(f"Point load at tip: {load.point_load} N")
    print(f"Elastic modulus: {material.elastic_modulus/1e9} GPa")
    print()

    results = []

    print(f"{'L/h':<8}{'h [mm]':<12}{'EB [mm]':<12}{'Timo [mm]':<12}{'Diff [%]':<10}{'Shear %':<10}")
    print("-" * 60)

    for L_h in aspect_ratios:
        h = base_length / L_h
        geometry = BeamGeometry(length=base_length, height=h, width=0.1)

        # Create beam models
        eb = EulerBernoulliBeam(geometry, material)
        timo = TimoshenkoBeam(geometry, material)

        # Compute tip deflections
        tip_eb = eb.tip_deflection(load)
        tip_timo = timo.tip_deflection(load)

        # Percentage difference
        diff = (tip_timo - tip_eb) / tip_timo * 100

        # Shear contribution
        shear_ratio = timo.shear_deformation_ratio(load) * 100

        results.append({
            'L_h': L_h,
            'h': h * 1000,
            'tip_eb': tip_eb * 1000,
            'tip_timo': tip_timo * 1000,
            'diff': diff,
            'shear': shear_ratio,
        })

        print(f"{L_h:<8}{h*1000:<12.2f}{tip_eb*1000:<12.4f}{tip_timo*1000:<12.4f}{diff:<10.2f}{shear_ratio:<10.2f}")

    print()
    print("Key Observations:")
    print("-" * 40)

    # Find where difference drops below threshold
    for threshold in [5, 1]:
        for r in results:
            if r['diff'] < threshold:
                print(f"  • Difference < {threshold}% for L/h >= {r['L_h']}")
                break

    # Natural frequencies comparison
    print("\nNatural Frequency Comparison (L/h = 10):")
    print("-" * 40)

    geometry = BeamGeometry(length=base_length, height=0.1, width=0.1)
    eb = EulerBernoulliBeam(geometry, material)
    timo = TimoshenkoBeam(geometry, material)

    freq_eb = eb.compute_natural_frequencies(5)
    freq_timo = timo.compute_natural_frequencies(5)

    print(f"{'Mode':<8}{'EB [Hz]':<15}{'Timo [Hz]':<15}{'Ratio':<10}")
    for i in range(5):
        ratio = freq_timo[i] / freq_eb[i]
        print(f"{i+1:<8}{freq_eb[i]:<15.2f}{freq_timo[i]:<15.2f}{ratio:<10.4f}")

    # Create visualization
    create_comparison_plot(results, base_length, material, load)


def create_comparison_plot(results, length, material, load):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    L_h = [r['L_h'] for r in results]

    # Plot 1: Tip deflection comparison
    ax1 = axes[0]
    ax1.plot(L_h, [r['tip_eb'] for r in results], 'bo-', label='Euler-Bernoulli', markersize=8)
    ax1.plot(L_h, [r['tip_timo'] for r in results], 'rs--', label='Timoshenko', markersize=8)
    ax1.set_xlabel('Aspect Ratio (L/h)')
    ax1.set_ylabel('Tip Deflection [mm]')
    ax1.set_title('Tip Deflection Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Percentage difference
    ax2 = axes[1]
    ax2.semilogy(L_h, [r['diff'] for r in results], 'go-', markersize=8)
    ax2.axhline(y=5, color='r', linestyle='--', label='5% threshold')
    ax2.axhline(y=1, color='orange', linestyle='--', label='1% threshold')
    ax2.set_xlabel('Aspect Ratio (L/h)')
    ax2.set_ylabel('Relative Difference [%]')
    ax2.set_title('EB Error vs Timoshenko')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Deflection along beam for thick vs slender
    ax3 = axes[2]
    x = np.linspace(0, length, 100)

    for L_h_val, style in [(5, 'solid'), (20, 'dashed')]:
        h = length / L_h_val
        geometry = BeamGeometry(length=length, height=h, width=0.1)
        eb = EulerBernoulliBeam(geometry, material)
        timo = TimoshenkoBeam(geometry, material)

        w_eb = eb.compute_deflection(x, load) * 1000
        w_timo = timo.compute_deflection(x, load) * 1000

        ax3.plot(x * 1000, w_eb, 'b', linestyle=style, label=f'EB (L/h={L_h_val})')
        ax3.plot(x * 1000, w_timo, 'r', linestyle=style, label=f'Timo (L/h={L_h_val})')

    ax3.set_xlabel('Position [mm]')
    ax3.set_ylabel('Deflection [mm]')
    ax3.set_title('Deflection Along Beam')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'quick_demo_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_dir / 'quick_demo_comparison.png'}")

    plt.show()


if __name__ == "__main__":
    main()
