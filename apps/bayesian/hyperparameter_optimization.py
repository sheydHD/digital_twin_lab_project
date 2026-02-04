"""
Hyperparameter Optimization for Bayesian Calibration using Optuna.

This module implements automatic tuning of:
- Prior distribution parameters
- MCMC sampling parameters
- Noise model parameters

The optimization maximizes model selection accuracy by finding hyperparameters
that produce the most physically correct and confident Bayes factors.

Reference:
- Akiba et al. (2019) "Optuna: A Next-generation Hyperparameter Optimization Framework"
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from apps.bayesian.calibration import (
    EulerBernoulliCalibrator,
    PriorConfig,
    TimoshenkoCalibrator,
)
from apps.bayesian.model_selection import BayesianModelSelector
from apps.data.synthetic_generator import SyntheticDataset

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """
    Results from hyperparameter optimization.

    Attributes:
        best_params: Best hyperparameters found
        best_score: Best objective value achieved
        n_trials: Number of trials completed
        study: Optuna study object
        optimization_history: History of objective values
    """
    best_params: Dict
    best_score: float
    n_trials: int
    study: optuna.Study
    optimization_history: List[float]


def get_physical_expectations(aspect_ratios: List[float]) -> List[str]:
    """
    Get physically expected model recommendations based on aspect ratio.

    Based on beam theory literature and engineering practice:
    - L/h < 10: Timoshenko strongly preferred (shear deformation significant)
    - 10 <= L/h <= 20: Transition zone (Timoshenko usually better)
    - L/h > 20: Euler-Bernoulli acceptable (shear negligible)
    - L/h > 40: Both equivalent (either acceptable)

    Args:
        aspect_ratios: List of L/h values

    Returns:
        List of expected model names
    """
    expectations = []
    for L_h in aspect_ratios:
        if L_h <= 12:
            expectations.append("Timoshenko")
        elif L_h >= 25:
            expectations.append("Euler-Bernoulli")
        else:
            # Transition zone - Timoshenko is conservative default
            expectations.append("Timoshenko")
    return expectations


def create_priors_from_params(
    trial_params: Dict,
    include_poisson: bool = False,
) -> List[PriorConfig]:
    """
    Create PriorConfig list from trial parameters.

    Args:
        trial_params: Dictionary of hyperparameters from Optuna trial
        include_poisson: Whether to include Poisson ratio prior (for Timoshenko)

    Returns:
        List of PriorConfig objects
    """
    priors = [
        PriorConfig(
            param_name="elastic_modulus",
            distribution="lognormal",
            params={
                "mu": np.log(210e9),
                "sigma": trial_params["E_prior_sigma"],
            },
        ),
        PriorConfig(
            param_name="sigma",
            distribution="halfnormal",
            params={"sigma": trial_params["sigma_prior_scale"]},
        ),
    ]

    if include_poisson:
        priors.append(
            PriorConfig(
                param_name="poisson_ratio",
                distribution="normal",
                params={
                    "mu": 0.3,
                    "sigma": trial_params["nu_prior_sigma"],
                },
            )
        )

    return priors


class BayesianHyperparameterOptimizer:
    """
    Optimize hyperparameters for Bayesian model selection.

    Uses Optuna to find the best combination of:
    - Prior distribution parameters (mean, variance)
    - MCMC sampling settings (samples, tuning, acceptance rate)
    - Model selection thresholds

    The objective maximizes model selection accuracy relative to
    physically expected outcomes.
    """

    def __init__(
        self,
        datasets: List[SyntheticDataset],
        expected_recommendations: Optional[List[str]] = None,
        output_dir: Path = Path("outputs/optimization"),
    ):
        """
        Initialize the optimizer.

        Args:
            datasets: List of synthetic datasets at different L/h ratios
            expected_recommendations: Expected model for each L/h
            output_dir: Directory for optimization outputs
        """
        self.datasets = datasets
        self.aspect_ratios = [d.geometry.aspect_ratio for d in datasets]

        if expected_recommendations is None:
            self.expected = get_physical_expectations(self.aspect_ratios)
        else:
            self.expected = expected_recommendations

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.selector = BayesianModelSelector()

    def _create_objective(self, fast_mode: bool = True) -> Callable:
        """
        Create the Optuna objective function.

        Args:
            fast_mode: If True, use fewer samples for faster optimization

        Returns:
            Objective function for Optuna
        """

        def objective(trial: optuna.Trial) -> float:
            """
            Objective function that maximizes model selection accuracy.

            Returns:
                Combined score of accuracy and confidence
            """
            # Sample hyperparameters
            n_samples = trial.suggest_int("n_samples", 400, 1000, step=200)
            n_tune = trial.suggest_int("n_tune", 200, 500, step=100)
            target_accept = trial.suggest_float("target_accept", 0.88, 0.98)

            # Prior hyperparameters
            trial.suggest_float("E_prior_sigma", 0.02, 0.12)
            trial.suggest_float(
                "sigma_prior_scale", 1e-8, 5e-6, log=True
            )
            trial.suggest_float("nu_prior_sigma", 0.01, 0.08)

            # Inconclusive threshold
            inconclusive_threshold = trial.suggest_float(
                "inconclusive_threshold", 0.3, 1.0
            )

            # Create priors
            eb_priors = create_priors_from_params(trial.params, include_poisson=False)
            timo_priors = create_priors_from_params(trial.params, include_poisson=True)

            # Track results
            correct_selections = 0
            total_confidence = 0.0
            n_evaluated = 0

            # Use subset of datasets for faster optimization
            if fast_mode:
                # Use key aspect ratios: thick, transition, slender
                indices = self._get_key_indices()
            else:
                indices = range(len(self.datasets))

            for idx in indices:
                dataset = self.datasets[idx]
                expected = self.expected[idx]
                L_h = self.aspect_ratios[idx]

                try:
                    # Calibrate Euler-Bernoulli
                    eb_cal = EulerBernoulliCalibrator(
                        priors=eb_priors,
                        n_samples=n_samples,
                        n_tune=n_tune,
                        n_chains=2,  # Use 2 chains for speed
                        target_accept=target_accept,
                    )
                    eb_result = eb_cal.calibrate(dataset)

                    # Calibrate Timoshenko
                    timo_cal = TimoshenkoCalibrator(
                        priors=timo_priors,
                        n_samples=n_samples,
                        n_tune=n_tune,
                        n_chains=2,
                        target_accept=target_accept,
                    )
                    timo_result = timo_cal.calibrate(dataset)

                    # Compare models
                    comparison = self.selector.compare_models(
                        eb_result, timo_result, use_marginal_likelihood=False
                    )

                    log_bf = comparison.log_bayes_factor

                    # Handle inconclusive cases
                    if abs(log_bf) < inconclusive_threshold:
                        # For inconclusive cases, prefer simpler model (EB) for slender,
                        # or conservative (Timo) for thick
                        if L_h > 20:
                            recommendation = "Euler-Bernoulli"
                        else:
                            recommendation = "Timoshenko"
                    else:
                        recommendation = comparison.recommended_model

                    # Check if recommendation matches expected
                    if recommendation == expected:
                        correct_selections += 1
                        # Reward confident correct decisions
                        if expected == "Timoshenko" and log_bf < 0:
                            total_confidence += min(abs(log_bf), 10)  # Cap at 10
                        elif expected == "Euler-Bernoulli" and log_bf > 0:
                            total_confidence += min(abs(log_bf), 10)
                    else:
                        # Penalize incorrect decisions based on how wrong they are
                        total_confidence -= min(abs(log_bf), 5)

                    n_evaluated += 1

                    # Report intermediate value for pruning
                    trial.report(correct_selections / n_evaluated, step=n_evaluated)

                    # Prune trial if clearly not promising
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                except Exception as e:
                    logger.warning(f"Trial failed for L/h={L_h}: {e}")
                    continue

            if n_evaluated == 0:
                return float('-inf')

            # Combined objective: accuracy + scaled confidence
            accuracy = correct_selections / n_evaluated
            confidence_bonus = total_confidence / n_evaluated / 20  # Scale to ~0.5 max

            return accuracy + confidence_bonus

        return objective

    def _get_key_indices(self) -> List[int]:
        """
        Get indices of key datasets for fast optimization.

        Selects datasets representing:
        - Thick beams (lowest L/h)
        - Transition zone
        - Slender beams (highest L/h)
        """
        n = len(self.datasets)
        if n <= 3:
            return list(range(n))

        # Select ~4-5 key points
        indices = [
            0,                  # Thickest
            n // 4,             # Lower-mid
            n // 2,             # Middle (transition)
            3 * n // 4,         # Upper-mid
            n - 1,              # Most slender
        ]

        return sorted(set(indices))

    def optimize(
        self,
        n_trials: int = 30,
        timeout: Optional[int] = 3600,
        fast_mode: bool = True,
        study_name: str = "bayesian_calibration_optimization",
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Maximum number of optimization trials
            timeout: Maximum optimization time in seconds
            fast_mode: Use subset of datasets for faster optimization
            study_name: Name for the Optuna study

        Returns:
            OptimizationResult with best parameters
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        # Create study with TPE sampler
        sampler = TPESampler(seed=42, n_startup_trials=10)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
        )

        # Create objective
        objective = self._create_objective(fast_mode=fast_mode)

        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            catch=(Exception,),
        )

        # Extract history
        history = [
            t.value for t in study.trials
            if t.value is not None
        ]

        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study=study,
            optimization_history=history,
        )

        # Save results
        self._save_results(result)

        return result

    def _save_results(self, result: OptimizationResult) -> None:
        """Save optimization results to file."""
        output = {
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "optimization_history": result.optimization_history,
        }

        filepath = self.output_dir / "optimization_results.json"
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Optimization results saved to {filepath}")

    def get_optimized_priors(
        self,
        params: Optional[Dict] = None,
    ) -> Tuple[List[PriorConfig], List[PriorConfig]]:
        """
        Get optimized prior configurations.

        Args:
            params: Optimized parameters (if None, uses defaults)

        Returns:
            Tuple of (eb_priors, timo_priors)
        """
        if params is None:
            # Use default optimized values
            params = {
                "E_prior_sigma": 0.05,
                "sigma_prior_scale": 1e-6,
                "nu_prior_sigma": 0.03,
            }

        eb_priors = create_priors_from_params(params, include_poisson=False)
        timo_priors = create_priors_from_params(params, include_poisson=True)

        return eb_priors, timo_priors


class FrequencyBasedModelSelector:
    """
    Model selection based on loading frequency analysis.

    At higher frequencies, shear deformation and rotary inertia become
    more important, making Timoshenko theory necessary even for
    slender beams.
    """

    def __init__(self):
        """Initialize the frequency-based selector."""
        pass

    def compute_frequency_threshold(
        self,
        geometry,
        material,
        mode_number: int = 1,
    ) -> Dict:
        """
        Compute the frequency at which model selection changes.

        For higher vibration modes, Timoshenko becomes necessary
        regardless of aspect ratio.

        Args:
            geometry: Beam geometry
            material: Material properties
            mode_number: Vibration mode number

        Returns:
            Dictionary with frequency thresholds and recommendations
        """
        from apps.models.euler_bernoulli import EulerBernoulliBeam
        from apps.models.timoshenko import TimoshenkoBeam

        eb_beam = EulerBernoulliBeam(geometry, material)
        timo_beam = TimoshenkoBeam(geometry, material)

        # Compute natural frequencies
        n_modes = max(mode_number + 2, 5)
        freq_eb = eb_beam.compute_natural_frequencies(n_modes)
        freq_timo = timo_beam.compute_natural_frequencies(n_modes)

        # Compute frequency ratios
        freq_ratios = freq_timo / freq_eb

        # Find the mode where difference exceeds 5%
        threshold_mode = None
        for i, ratio in enumerate(freq_ratios):
            if ratio < 0.95:
                threshold_mode = i + 1
                break

        # Compute transition frequency
        if threshold_mode is not None:
            transition_freq = freq_timo[threshold_mode - 1]
        else:
            transition_freq = freq_timo[-1]

        return {
            "eb_frequencies_hz": freq_eb.tolist(),
            "timoshenko_frequencies_hz": freq_timo.tolist(),
            "frequency_ratios": freq_ratios.tolist(),
            "transition_mode": threshold_mode,
            "transition_frequency_hz": float(transition_freq),
            "aspect_ratio": geometry.aspect_ratio,
            "recommendation": self._get_frequency_recommendation(
                geometry.aspect_ratio, threshold_mode
            ),
        }

    def _get_frequency_recommendation(
        self,
        aspect_ratio: float,
        threshold_mode: Optional[int],
    ) -> str:
        """Generate frequency-based recommendation."""
        if aspect_ratio < 10:
            return (
                "Timoshenko theory recommended for all frequencies. "
                "Shear deformation significant even at fundamental mode."
            )
        elif aspect_ratio > 30:
            if threshold_mode and threshold_mode <= 3:
                return (
                    f"Euler-Bernoulli acceptable for modes 1-{threshold_mode-1}. "
                    f"Use Timoshenko for mode {threshold_mode} and higher."
                )
            else:
                return (
                    "Euler-Bernoulli acceptable for most practical frequencies. "
                    "Consider Timoshenko for very high-frequency analysis."
                )
        else:
            return (
                "Transition region: Use Bayesian model selection. "
                "For dynamic analysis, prefer Timoshenko for safety."
            )

    def analyze_frequency_study(
        self,
        datasets: List[SyntheticDataset],
    ) -> Dict:
        """
        Analyze model selection across frequencies for all datasets.

        Args:
            datasets: List of synthetic datasets

        Returns:
            Dictionary with frequency analysis results
        """
        results = []

        for dataset in datasets:
            freq_result = self.compute_frequency_threshold(
                dataset.geometry,
                dataset.material,
            )
            results.append(freq_result)

        return {
            "frequency_analysis": results,
            "summary": self._summarize_frequency_study(results),
        }

    def _summarize_frequency_study(self, results: List[Dict]) -> Dict:
        """Generate summary of frequency study."""
        transition_modes = [
            r["transition_mode"] for r in results
            if r["transition_mode"] is not None
        ]

        return {
            "n_datasets_analyzed": len(results),
            "typical_transition_mode": int(np.median(transition_modes)) if transition_modes else None,
            "guideline": (
                "For dynamic digital twins:\n"
                "1. Check if excitation frequency exceeds first natural frequency\n"
                "2. For higher modes (n > 2), prefer Timoshenko theory\n"
                "3. For impact or shock loading, always use Timoshenko"
            ),
        }


def run_quick_optimization(
    datasets: List[SyntheticDataset],
    n_trials: int = 20,
) -> Dict:
    """
    Run a quick optimization with minimal settings.

    Args:
        datasets: Synthetic datasets
        n_trials: Number of optimization trials

    Returns:
        Dictionary with optimized parameters
    """
    optimizer = BayesianHyperparameterOptimizer(datasets)
    result = optimizer.optimize(
        n_trials=n_trials,
        fast_mode=True,
    )

    return {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "eb_priors": create_priors_from_params(result.best_params, False),
        "timo_priors": create_priors_from_params(result.best_params, True),
    }
