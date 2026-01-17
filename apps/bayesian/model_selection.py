"""
Bayesian Model Selection Module.

Implements model comparison methods using marginal likelihoods,
Bayes factors, and information criteria for selecting between
Euler-Bernoulli and Timoshenko beam theories.

Key concepts:
- Bayes Factor: BF = p(y|M1) / p(y|M2) - ratio of marginal likelihoods
- Model Evidence: p(y|M) = ∫ p(y|θ,M) p(θ|M) dθ
- Information Criteria: WAIC, LOO-CV as approximations

Reference:
- Kass & Raftery (1995) "Bayes Factors"
- Vehtari et al. (2017) "Practical Bayesian model evaluation"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from apps.bayesian.calibration import CalibrationResult


class ModelEvidence(Enum):
    """
    Interpretation of Bayes factor evidence strength.

    Following Kass & Raftery (1995) guidelines.
    """

    STRONG_M2 = "Strong evidence for Model 2"
    MODERATE_M2 = "Moderate evidence for Model 2"
    WEAK_M2 = "Weak evidence for Model 2"
    INCONCLUSIVE = "Inconclusive - models equally supported"
    WEAK_M1 = "Weak evidence for Model 1"
    MODERATE_M1 = "Moderate evidence for Model 1"
    STRONG_M1 = "Strong evidence for Model 1"


@dataclass
class ModelComparisonResult:
    """
    Result of Bayesian model comparison.

    Attributes:
        model1_name: Name of first model
        model2_name: Name of second model
        log_bayes_factor: Log of Bayes factor (log BF = log p(y|M1) - log p(y|M2))
        bayes_factor: Bayes factor value
        evidence_interpretation: Qualitative evidence strength
        model1_probability: Posterior probability of Model 1 (assuming equal priors)
        model2_probability: Posterior probability of Model 2
        waic_difference: WAIC(M1) - WAIC(M2)
        loo_difference: LOO(M1) - LOO(M2)
        recommended_model: Name of recommended model
    """

    model1_name: str
    model2_name: str
    log_bayes_factor: float
    bayes_factor: float
    evidence_interpretation: ModelEvidence
    model1_probability: float
    model2_probability: float
    waic_difference: Optional[float]
    loo_difference: Optional[float]
    recommended_model: str


class BayesianModelSelector:
    """
    Bayesian model selection for beam theories.

    Compares Euler-Bernoulli and Timoshenko models using various
    criteria to determine which theory is most appropriate for
    a given beam configuration.

    """

    # Bayes factor thresholds (Kass & Raftery, 1995)
    BF_THRESHOLDS = {
        "strong": 100,  # log BF > 4.6
        "moderate": 10,  # log BF > 2.3
        "weak": 3,  # log BF > 1.1
    }

    def __init__(self):
        """Initialize the model selector."""
        self.results_cache = {}

    def compare_models(
        self,
        result1: CalibrationResult,
        result2: CalibrationResult,
        use_marginal_likelihood: bool = True,
    ) -> ModelComparisonResult:
        """
        Compare two calibrated models using Bayesian criteria.

        IMPORTANT - Bayes Factor Convention:
        ---------------------------------
        M1 (numerator)   = result1 (typically Euler-Bernoulli)
        M2 (denominator) = result2 (typically Timoshenko)
        
        BF = P(Data | M1) / P(Data | M2)
        log_BF = log P(Data | M1) - log P(Data | M2)
        
        Interpretation:
        - log_BF > 0  =>  Evidence FAVORS M1 (result1, typically EB)
        - log_BF < 0  =>  Evidence FAVORS M2 (result2, typically Timoshenko)
        - log_BF ≈ 0  =>  Inconclusive, models equally supported

        Args:
            result1: Calibration result for first model (M1, numerator)
            result2: Calibration result for second model (M2, denominator)
            use_marginal_likelihood: If True, use marginal likelihood;
                                    otherwise use information criteria (WAIC)

        Returns:
            ModelComparisonResult with comparison metrics

        """
        if use_marginal_likelihood:
            if result1.marginal_likelihood_estimate is None or \
               result2.marginal_likelihood_estimate is None:
                raise ValueError(
                    "Marginal likelihood not computed. "
                    "Use information criteria or compute marginal likelihood first."
                )

            # log_BF = log P(D|M1) - log P(D|M2)
            log_ml1 = result1.marginal_likelihood_estimate
            log_ml2 = result2.marginal_likelihood_estimate
            log_bf = log_ml1 - log_ml2
        else:
            # Use WAIC (elpd_waic) as approximation to log marginal likelihood
            # Note: elpd = expected log pointwise predictive density (HIGHER is better)
            # elpd approximates log P(D|M), so elpd_diff approximates log_BF
            if result1.waic is None or result2.waic is None:
                raise ValueError("WAIC not computed for one or both models")

            # log_BF ≈ elpd_M1 - elpd_M2 (positive favors M1)
            elpd_diff = result1.waic - result2.waic
            log_bf = elpd_diff

        # Compute Bayes factor: BF = exp(log_BF)
        # BF > 1 favors M1, BF < 1 favors M2
        bf = np.exp(np.clip(log_bf, -500, 500))  # Clip to prevent overflow

        # Interpret evidence
        evidence = self._interpret_bayes_factor(log_bf)

        # Compute posterior probabilities (assuming equal priors)
        prob1 = bf / (1 + bf)
        prob2 = 1 - prob1

        # Information criteria differences
        waic_diff = None
        loo_diff = None
        if result1.waic is not None and result2.waic is not None:
            waic_diff = result1.waic - result2.waic
        if result1.loo is not None and result2.loo is not None:
            loo_diff = result1.loo - result2.loo

        # Recommend model
        recommended = result1.model_name if log_bf > 0 else result2.model_name

        return ModelComparisonResult(
            model1_name=result1.model_name,
            model2_name=result2.model_name,
            log_bayes_factor=log_bf,
            bayes_factor=bf,
            evidence_interpretation=evidence,
            model1_probability=prob1,
            model2_probability=prob2,
            waic_difference=waic_diff,
            loo_difference=loo_diff,
            recommended_model=recommended,
        )

    def _interpret_bayes_factor(self, log_bf: float) -> ModelEvidence:
        """
        Interpret Bayes factor according to Kass & Raftery guidelines.

        Args:
            log_bf: Natural log of Bayes factor

        Returns:
            ModelEvidence enum value
        """
        abs_log_bf = abs(log_bf)

        if abs_log_bf < np.log(self.BF_THRESHOLDS["weak"]):
            return ModelEvidence.INCONCLUSIVE
        elif abs_log_bf < np.log(self.BF_THRESHOLDS["moderate"]):
            return ModelEvidence.WEAK_M1 if log_bf > 0 else ModelEvidence.WEAK_M2
        elif abs_log_bf < np.log(self.BF_THRESHOLDS["strong"]):
            return ModelEvidence.MODERATE_M1 if log_bf > 0 else ModelEvidence.MODERATE_M2
        else:
            return ModelEvidence.STRONG_M1 if log_bf > 0 else ModelEvidence.STRONG_M2

    def analyze_aspect_ratio_study(
        self,
        eb_results: List[CalibrationResult],
        timo_results: List[CalibrationResult],
        aspect_ratios: List[float],
    ) -> Dict:
        """
        Analyze model selection across different aspect ratios.

        This is the core analysis for determining when each beam theory
        should be selected in digital twin applications.

        Args:
            eb_results: Euler-Bernoulli calibration results for each L/h
            timo_results: Timoshenko calibration results for each L/h
            aspect_ratios: Corresponding aspect ratios

        Returns:
            Dictionary with analysis results and recommendations

        """
        if len(eb_results) != len(timo_results) != len(aspect_ratios):
            raise ValueError("Mismatched result lengths")

        comparisons = []
        log_bfs = []
        recommendations = []

        for i, L_h in enumerate(aspect_ratios):
            comp = self.compare_models(
                eb_results[i],
                timo_results[i],
                use_marginal_likelihood=False,  # Use WAIC for robustness
            )
            comparisons.append(comp)
            log_bfs.append(comp.log_bayes_factor)
            recommendations.append(comp.recommended_model)

        # Find transition point
        transition_point = self._find_transition_point(
            aspect_ratios, log_bfs
        )

        # Generate guidelines
        guidelines = self._generate_guidelines(
            aspect_ratios, log_bfs, transition_point
        )

        return {
            "aspect_ratios": aspect_ratios,
            "log_bayes_factors": log_bfs,
            "comparisons": comparisons,
            "recommendations": recommendations,
            "transition_aspect_ratio": transition_point,
            "guidelines": guidelines,
        }

    def _find_transition_point(
        self,
        aspect_ratios: List[float],
        log_bfs: List[float],
    ) -> Optional[float]:
        """
        Find the aspect ratio where model preference transitions.

        The transition occurs when log BF crosses zero, indicating
        equal evidence for both models.

        Args:
            aspect_ratios: L/h values
            log_bfs: Corresponding log Bayes factors

        Returns:
            Estimated transition aspect ratio, or None if not found

        """
        L_h = np.array(aspect_ratios)
        bf = np.array(log_bfs)

        # Look for sign change
        sign_changes = np.where(np.diff(np.sign(bf)))[0]

        if len(sign_changes) == 0:
            return None

        # Linear interpolation to find crossing point
        idx = sign_changes[0]
        # Linear interpolation: L_h_transition where bf = 0
        L_h_trans = L_h[idx] - bf[idx] * (L_h[idx + 1] - L_h[idx]) / (bf[idx + 1] - bf[idx])

        return float(L_h_trans)

    def _generate_guidelines(
        self,
        aspect_ratios: List[float],
        log_bfs: List[float],
        transition_point: Optional[float],
    ) -> Dict[str, str]:
        """
        Generate practical guidelines for model selection.

        These guidelines are intended for digital twin initialization
        and recalibration decisions.

        Args:
            aspect_ratios: L/h values studied
            log_bfs: Log Bayes factors
            transition_point: Transition aspect ratio

        Returns:
            Dictionary with guideline text

        """
        guidelines = {}

        if transition_point is not None:
            guidelines["transition_rule"] = (
                f"Transition from Euler-Bernoulli to Timoshenko theory "
                f"is recommended around L/h ≈ {transition_point:.1f}"
            )
        else:
            # Determine which model is preferred across all aspect ratios
            avg_bf = np.mean(log_bfs)
            preferred = "Euler-Bernoulli" if avg_bf > 0 else "Timoshenko"
            guidelines["transition_rule"] = (
                f"No clear transition found in studied range. "
                f"{preferred} is generally preferred."
            )

        # Slender beam guideline
        min_L_h = min(aspect_ratios)
        max_L_h = max(aspect_ratios)

        guidelines["slender_beams"] = (
            f"For L/h > {max(transition_point or 20, 15):.0f}: "
            f"Euler-Bernoulli theory is recommended. "
            f"Shear deformation is negligible."
        )

        guidelines["thick_beams"] = (
            f"For L/h < {min(transition_point or 10, 8):.0f}: "
            f"Timoshenko theory is strongly recommended. "
            f"Shear deformation significantly affects response."
        )

        guidelines["intermediate_beams"] = (
            "For intermediate aspect ratios: "
            "Both theories may be acceptable. Use Bayesian model selection "
            "or consider model averaging for critical applications."
        )

        guidelines["digital_twin_recommendation"] = (
            "Digital Twin Implementation:\n"
            "1. During initialization: Check beam aspect ratio against thresholds\n"
            "2. Default to Timoshenko for safety in uncertain cases\n"
            "3. Recalibrate model selection when geometry or loading changes\n"
            "4. For high-frequency analysis (f > f_1), prefer Timoshenko"
        )

        return guidelines


def compute_bayes_factor_direct(
    eb_result: CalibrationResult,
    timo_result: CalibrationResult,
) -> Tuple[float, float]:
    """
    Compute Bayes factor using log-likelihood samples.

    This uses the harmonic mean estimator as a simple approach.

    Args:
        eb_result: Euler-Bernoulli calibration result
        timo_result: Timoshenko calibration result

    Returns:
        (log_bayes_factor, bayes_factor)

    """
    # Get log-likelihood values
    ll_eb = eb_result.log_likelihood.sum(axis=-1).flatten()
    ll_timo = timo_result.log_likelihood.sum(axis=-1).flatten()

    # Harmonic mean estimator for marginal likelihood
    # log p(y|M) ≈ -log(mean(exp(-log_lik)))

    # Use log-sum-exp trick for numerical stability
    def log_marginal_harmonic(log_lik):
        n = len(log_lik)
        return -np.log(n) + np.log(np.sum(np.exp(-log_lik - np.max(-log_lik)))) + np.max(-log_lik)

    log_ml_eb = -log_marginal_harmonic(ll_eb)
    log_ml_timo = -log_marginal_harmonic(ll_timo)

    log_bf = log_ml_eb - log_ml_timo
    bf = np.exp(log_bf)

    return log_bf, bf
