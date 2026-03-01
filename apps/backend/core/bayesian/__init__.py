"""Bayesian calibration subpackage with lazy imports.

Heavy dependencies (Optuna, PyMC, ArviZ) are loaded on first access
to keep ``import apps.bayesian`` fast.
"""

from __future__ import annotations

# Mapping of public names → (module, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Calibration
    "BayesianCalibrator": ("apps.bayesian.calibration", "BayesianCalibrator"),
    "EulerBernoulliCalibrator": ("apps.bayesian.calibration", "EulerBernoulliCalibrator"),
    "TimoshenkoCalibrator": ("apps.bayesian.calibration", "TimoshenkoCalibrator"),
    "CalibrationResult": ("apps.bayesian.calibration", "CalibrationResult"),
    "PriorConfig": ("apps.bayesian.calibration", "PriorConfig"),
    "ConvergenceError": ("apps.bayesian.calibration", "ConvergenceError"),
    "ConvergenceWarning": ("apps.bayesian.calibration", "ConvergenceWarning"),
    "create_default_priors": ("apps.bayesian.calibration", "create_default_priors"),
    "create_timoshenko_priors": ("apps.bayesian.calibration", "create_timoshenko_priors"),
    # Model selection
    "BayesianModelSelector": ("apps.bayesian.model_selection", "BayesianModelSelector"),
    "ModelComparisonResult": ("apps.bayesian.model_selection", "ModelComparisonResult"),
    "ModelEvidence": ("apps.bayesian.model_selection", "ModelEvidence"),
    # Optimization (heavy – Optuna)
    "BayesianHyperparameterOptimizer": (
        "apps.bayesian.hyperparameter_optimization",
        "BayesianHyperparameterOptimizer",
    ),
    "FrequencyBasedModelSelector": (
        "apps.bayesian.hyperparameter_optimization",
        "FrequencyBasedModelSelector",
    ),
    "OptimizationResult": (
        "apps.bayesian.hyperparameter_optimization",
        "OptimizationResult",
    ),
    "get_physical_expectations": (
        "apps.bayesian.hyperparameter_optimization",
        "get_physical_expectations",
    ),
    "create_priors_from_params": (
        "apps.bayesian.hyperparameter_optimization",
        "create_priors_from_params",
    ),
    "run_quick_optimization": (
        "apps.bayesian.hyperparameter_optimization",
        "run_quick_optimization",
    ),
    # Normalization
    "NormalizationParams": ("apps.bayesian.normalization", "NormalizationParams"),
    "compute_normalization_params": ("apps.bayesian.normalization", "compute_normalization_params"),
    "create_normalizer_from_dataset": (
        "apps.bayesian.normalization",
        "create_normalizer_from_dataset",
    ),
    "normalize_displacements": ("apps.bayesian.normalization", "normalize_displacements"),
    "denormalize_displacements": ("apps.bayesian.normalization", "denormalize_displacements"),
    "normalize_elastic_modulus": ("apps.bayesian.normalization", "normalize_elastic_modulus"),
    "denormalize_elastic_modulus": ("apps.bayesian.normalization", "denormalize_elastic_modulus"),
    "normalize_E": ("apps.bayesian.normalization", "normalize_E"),
    "denormalize_E": ("apps.bayesian.normalization", "denormalize_E"),
    # Bridge Sampling
    "BridgeSampler": ("apps.bayesian.bridge_sampling", "BridgeSampler"),
    "BridgeSamplingResult": ("apps.bayesian.bridge_sampling", "BridgeSamplingResult"),
    "compute_bayes_factor_bridge": ("apps.bayesian.bridge_sampling", "compute_bayes_factor_bridge"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> object:
    """Lazily import public symbols on first access."""
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        # Cache on the module so __getattr__ is not called again
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
