"""
Results Reporter Module.

Generates comprehensive reports summarizing:
- Model calibration results
- Model selection outcomes
- Practical guidelines for digital twin implementation
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from apps.backend.core.bayesian.calibration import CalibrationResult
from apps.backend.core.bayesian.model_selection import ModelComparisonResult


class ResultsReporter:
    """
    Generate reports from Bayesian model selection analysis.

    """

    def __init__(self, output_dir: Path = Path("outputs/reports")):
        """
        Initialize reporter.

        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_calibration_report(
        self,
        result: CalibrationResult,
        filename: str | None = None,
    ) -> str:
        """
        Generate calibration report for a single model.

        Args:
            result: Calibration result
            filename: Output filename (optional)

        Returns:
            Report text

        """
        report = []
        report.append("=" * 70)
        report.append(f"BAYESIAN CALIBRATION REPORT: {result.model_name}")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Posterior summary
        report.append("POSTERIOR PARAMETER ESTIMATES")
        report.append("-" * 40)

        for param, stats in result.posterior_summary.items():
            if param not in ["y_pred"]:
                report.append(f"\n{param}:")
                mean_val = stats.get("mean", "N/A")
                std_val = stats.get("sd", "N/A")
                hdi_low = stats.get("hdi_3%", "N/A")
                hdi_high = stats.get("hdi_97%", "N/A")
                # Format numbers or show N/A
                report.append(
                    f"  Mean:   {mean_val:.6e}"
                    if isinstance(mean_val, (int, float))
                    else f"  Mean:   {mean_val}"
                )
                report.append(
                    f"  Std:    {std_val:.6e}"
                    if isinstance(std_val, (int, float))
                    else f"  Std:    {std_val}"
                )
                report.append(
                    f"  HDI 3%: {hdi_low:.6e}"
                    if isinstance(hdi_low, (int, float))
                    else f"  HDI 3%: {hdi_low}"
                )
                report.append(
                    f"  HDI 97%: {hdi_high:.6e}"
                    if isinstance(hdi_high, (int, float))
                    else f"  HDI 97%: {hdi_high}"
                )

        # Model comparison metrics
        report.append("")
        report.append("MODEL COMPARISON METRICS")
        report.append("-" * 40)
        if result.waic is not None:
            report.append(f"WAIC (elpd_waic): {result.waic:.2f}")

        # Convergence diagnostics
        if result.convergence_diagnostics:
            report.append("")
            report.append("CONVERGENCE DIAGNOSTICS")
            report.append("-" * 40)
            for param, rhat in result.convergence_diagnostics.get("rhat", {}).items():
                ess = result.convergence_diagnostics.get("ess_bulk", {}).get(param, "N/A")
                status = "✓" if rhat < 1.01 else "✗"
                ess_str = f"{ess:.0f}" if isinstance(ess, (int, float)) else str(ess)
                report.append(f"{param}: R-hat={rhat:.3f} {status}, ESS={ess_str}")

        report_text = "\n".join(report)

        if filename:
            filepath = self.output_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)

        return report_text

    def generate_comparison_report(
        self,
        comparison: ModelComparisonResult,
        filename: str | None = None,
    ) -> str:
        """
        Generate model comparison report.

        Args:
            comparison: Model comparison result
            filename: Output filename

        Returns:
            Report text
        """
        report = []
        report.append("=" * 70)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 70)
        report.append(f"Model 1: {comparison.model1_name}")
        report.append(f"Model 2: {comparison.model2_name}")
        report.append("")

        report.append("BAYES FACTOR ANALYSIS")
        report.append("-" * 40)
        report.append(f"Log Bayes Factor (M1 vs M2): {comparison.log_bayes_factor:.4f}")
        report.append(f"Bayes Factor: {comparison.bayes_factor:.4f}")
        report.append(f"Evidence: {comparison.evidence_interpretation.value}")
        report.append("")

        report.append("POSTERIOR MODEL PROBABILITIES (equal priors)")
        report.append("-" * 40)
        report.append(f"P({comparison.model1_name}|data): {comparison.model1_probability:.4f}")
        report.append(f"P({comparison.model2_name}|data): {comparison.model2_probability:.4f}")
        report.append("")

        report.append("INFORMATION CRITERIA")
        report.append("-" * 40)
        if comparison.waic_difference is not None:
            report.append(f"ΔWAIC (M1 - M2): {comparison.waic_difference:.2f}")
        report.append("")

        report.append("RECOMMENDATION")
        report.append("-" * 40)
        report.append(f"Selected model: {comparison.recommended_model}")

        report_text = "\n".join(report)

        if filename:
            filepath = self.output_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)

        return report_text

    def generate_study_summary(
        self,
        study_results: dict,
        filename: str = "study_summary.txt",
    ) -> str:
        """
        Generate comprehensive study summary.

        Args:
            study_results: Results from aspect ratio study
            filename: Output filename

        Returns:
            Report text

        """
        report = []
        report.append("=" * 70)
        report.append("BAYESIAN MODEL SELECTION STUDY SUMMARY")
        report.append("Beam Theory Selection for Digital Twin Applications")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Study parameters
        aspect_ratios = study_results["aspect_ratios"]
        report.append("STUDY PARAMETERS")
        report.append("-" * 40)
        report.append(f"Aspect ratios analyzed: {aspect_ratios}")
        report.append(f"Number of configurations: {len(aspect_ratios)}")
        report.append("")

        # Results summary
        report.append("RESULTS SUMMARY")
        report.append("-" * 40)

        log_bfs = study_results["log_bayes_factors"]
        recommendations = study_results["recommendations"]

        # Count recommendations
        eb_count = sum(1 for r in recommendations if r == "Euler-Bernoulli")
        timo_count = len(recommendations) - eb_count

        report.append(f"Euler-Bernoulli preferred: {eb_count}/{len(recommendations)} cases")
        report.append(f"Timoshenko preferred: {timo_count}/{len(recommendations)} cases")
        report.append("")

        # Transition point
        transition = study_results.get("transition_aspect_ratio")
        if transition:
            report.append(f"Transition aspect ratio (L/h): {transition:.2f}")
        else:
            report.append("No clear transition point found in studied range")
        report.append("")

        # Detailed results table
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        report.append(f"{'L/h':<10}{'Log BF':<15}{'Recommendation':<20}")
        report.append("-" * 40)
        for L_h, bf, rec in zip(aspect_ratios, log_bfs, recommendations, strict=True):
            report.append(f"{L_h:<10.1f}{bf:<15.4f}{rec:<20}")
        report.append("")

        # Guidelines
        report.append("PRACTICAL GUIDELINES")
        report.append("-" * 40)
        guidelines = study_results.get("guidelines", {})
        for key, value in guidelines.items():
            report.append(f"\n{key.upper()}:")
            report.append(value)

        report_text = "\n".join(report)

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)

        return report_text

    def export_results_json(
        self,
        study_results: dict,
        filename: str = "results.json",
    ) -> None:
        """
        Export results to JSON format.

        Args:
            study_results: Study results dictionary
            filename: Output filename

        """
        # Create serializable version
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "aspect_ratios": list(study_results["aspect_ratios"]),
            "log_bayes_factors": list(study_results["log_bayes_factors"]),
            "recommendations": study_results["recommendations"],
            "transition_aspect_ratio": study_results.get("transition_aspect_ratio"),
            "guidelines": study_results.get("guidelines", {}),
        }

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

    def export_results_csv(
        self,
        study_results: dict,
        filename: str = "results.csv",
    ) -> None:
        """
        Export results to CSV format.

        Args:
            study_results: Study results dictionary
            filename: Output filename
        """
        df = pd.DataFrame(
            {
                "aspect_ratio": study_results["aspect_ratios"],
                "log_bayes_factor": study_results["log_bayes_factors"],
                "recommendation": study_results["recommendations"],
            }
        )

        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)

    def generate_frequency_report(
        self,
        frequency_results: dict,
        filename: str = "frequency_analysis.txt",
    ) -> str:
        """
        Generate frequency analysis report.

        Args:
            frequency_results: Results from frequency analysis
            filename: Output filename

        Returns:
            Report text
        """
        report = []
        report.append("=" * 70)
        report.append("FREQUENCY-BASED MODEL SELECTION ANALYSIS")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        summary = frequency_results.get("summary", {})
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Datasets analyzed: {summary.get('n_datasets_analyzed', 'N/A')}")

        transition_mode = summary.get("typical_transition_mode")
        if transition_mode:
            report.append(f"Typical transition mode: {transition_mode}")
        report.append("")

        # Detailed results per aspect ratio
        report.append("FREQUENCY ANALYSIS BY ASPECT RATIO")
        report.append("-" * 40)

        freq_analysis = frequency_results.get("frequency_analysis", [])
        for result in freq_analysis:
            L_h = result.get("aspect_ratio", "N/A")
            report.append(f"\nL/h = {L_h:.1f}:")

            # Natural frequencies
            eb_freqs = result.get("eb_frequencies_hz", [])
            timo_freqs = result.get("timoshenko_frequencies_hz", [])

            if eb_freqs and timo_freqs:
                report.append("  Mode |  EB (Hz)   | Timo (Hz)  | Ratio")
                report.append("  -----|------------|------------|-------")
                for i, (f_eb, f_t) in enumerate(zip(eb_freqs[:5], timo_freqs[:5], strict=True)):
                    ratio = f_t / f_eb if f_eb > 0 else 1.0
                    report.append(f"    {i + 1}  | {f_eb:10.2f} | {f_t:10.2f} | {ratio:.4f}")

            transition = result.get("transition_mode")
            if transition:
                report.append(f"  Transition mode: {transition}")

            report.append(f"  Recommendation: {result.get('recommendation', 'N/A')}")

        report.append("")
        report.append("GUIDELINE")
        report.append("-" * 40)
        report.append(summary.get("guideline", "No guideline available"))

        report_text = "\n".join(report)

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)

        return report_text
