"""
Unit tests for the model selection module.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from apps.backend.core.bayesian.model_selection import (
    BayesianModelSelector,
    ModelEvidence,
)


@pytest.fixture
def selector() -> BayesianModelSelector:
    return BayesianModelSelector(inconclusive_threshold=0.5)


def _make_result(name: str, log_ml: float, waic: float | None = None) -> MagicMock:
    """Create a mock CalibrationResult."""
    r = MagicMock()
    r.model_name = name
    r.marginal_likelihood_estimate = log_ml
    r.waic = waic
    return r


class TestBayesianModelSelector:
    def test_strong_evidence_m1(self, selector: BayesianModelSelector) -> None:
        r1 = _make_result("EB", log_ml=0.0)
        r2 = _make_result("Timo", log_ml=-10.0)
        result = selector.compare_models(r1, r2)

        assert result.log_bayes_factor == pytest.approx(10.0)
        assert result.evidence_interpretation == ModelEvidence.STRONG_M1
        assert result.recommended_model == "EB"

    def test_strong_evidence_m2(self, selector: BayesianModelSelector) -> None:
        r1 = _make_result("EB", log_ml=-10.0)
        r2 = _make_result("Timo", log_ml=0.0)
        result = selector.compare_models(r1, r2)

        assert result.log_bayes_factor == pytest.approx(-10.0)
        assert result.evidence_interpretation == ModelEvidence.STRONG_M2
        assert result.recommended_model == "Timo"

    def test_inconclusive(self, selector: BayesianModelSelector) -> None:
        r1 = _make_result("EB", log_ml=-50.0)
        r2 = _make_result("Timo", log_ml=-50.1)
        result = selector.compare_models(r1, r2)

        assert abs(result.log_bayes_factor) < 0.5
        assert result.evidence_interpretation == ModelEvidence.INCONCLUSIVE

    def test_missing_marginal_raises(self, selector: BayesianModelSelector) -> None:
        r1 = _make_result("EB", log_ml=None)
        r2 = _make_result("Timo", log_ml=-5.0)
        with pytest.raises(ValueError, match="Marginal likelihood"):
            selector.compare_models(r1, r2)

    def test_posterior_probabilities_sum_to_one(self, selector: BayesianModelSelector) -> None:
        r1 = _make_result("EB", log_ml=-100.0)
        r2 = _make_result("Timo", log_ml=-105.0)
        result = selector.compare_models(r1, r2)

        assert result.model1_probability + result.model2_probability == pytest.approx(1.0)

    def test_analyze_aspect_ratio_study(self, selector: BayesianModelSelector) -> None:
        eb = [_make_result("EB", log_ml=-50 + i * 5) for i in range(4)]
        timo = [_make_result("Timo", log_ml=-40 - i * 3) for i in range(4)]
        ratios = [5.0, 10.0, 20.0, 50.0]

        study = selector.analyze_aspect_ratio_study(eb, timo, ratios)

        assert len(study["log_bayes_factors"]) == 4
        assert len(study["recommendations"]) == 4
        assert "guidelines" in study
