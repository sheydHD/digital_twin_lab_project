"""
Unit tests for the results reporter module.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apps.backend.core.analysis.reporter import ResultsReporter


@pytest.fixture
def reporter(tmp_path: Path) -> ResultsReporter:
    return ResultsReporter(output_dir=tmp_path)


@pytest.fixture
def study_results() -> dict:
    return {
        "aspect_ratios": [5.0, 10.0, 20.0],
        "log_bayes_factors": [-3.0, -0.2, 2.5],
        "comparisons": [],
        "recommendations": ["Timoshenko", "Euler-Bernoulli", "Euler-Bernoulli"],
        "transition_aspect_ratio": 12.0,
        "guidelines": {
            "transition_rule": "Transition at L/h ≈ 12",
            "slender_beams": "Use EB for L/h > 15",
            "thick_beams": "Use Timo for L/h < 8",
            "intermediate_beams": "Run Bayesian model selection",
            "digital_twin_recommendation": "Check beam aspect ratio",
        },
    }


class TestResultsReporter:
    def test_generate_study_summary(self, reporter: ResultsReporter, study_results: dict) -> None:
        text = reporter.generate_study_summary(study_results)
        assert "MODEL SELECTION STUDY SUMMARY" in text
        assert "Transition" in text
        assert (reporter.output_dir / "study_summary.txt").exists()

    def test_export_json(self, reporter: ResultsReporter, study_results: dict) -> None:
        reporter.export_results_json(study_results, "test.json")
        path = reporter.output_dir / "test.json"
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["aspect_ratios"] == [5.0, 10.0, 20.0]
        assert data["transition_aspect_ratio"] == 12.0

    def test_export_csv(self, reporter: ResultsReporter, study_results: dict) -> None:
        reporter.export_results_csv(study_results, "test.csv")
        path = reporter.output_dir / "test.csv"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_calibration_report(self, reporter: ResultsReporter) -> None:
        result = MagicMock()
        result.model_name = "Euler-Bernoulli"
        result.posterior_summary = {
            "elastic_modulus": {"mean": 2.1e11, "sd": 1e9, "hdi_3%": 2.08e11, "hdi_97%": 2.12e11},
        }
        result.waic = -150.0
        result.convergence_diagnostics = {
            "rhat": {"elastic_modulus": 1.001},
            "ess_bulk": {"elastic_modulus": 3000.0},
        }

        text = reporter.generate_calibration_report(result, "cal.txt")
        assert "CALIBRATION REPORT" in text
        assert (reporter.output_dir / "cal.txt").exists()
