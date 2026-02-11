# Bayesian calibration subpackage

from apps.bayesian.bridge_sampling import (
    BridgeSampler,
    BridgeSamplingResult,
    compute_bayes_factor_bridge,
)
from apps.bayesian.calibration import (
    BayesianCalibrator,
    CalibrationResult,
    ConvergenceError,
    ConvergenceWarning,
    EulerBernoulliCalibrator,
    PriorConfig,
    TimoshenkoCalibrator,
    create_default_priors,
    create_timoshenko_priors,
)
from apps.bayesian.hyperparameter_optimization import (
    BayesianHyperparameterOptimizer,
    FrequencyBasedModelSelector,
    OptimizationResult,
    create_priors_from_params,
    get_physical_expectations,
    run_quick_optimization,
)
from apps.bayesian.model_selection import (
    BayesianModelSelector,
    ModelComparisonResult,
    ModelEvidence,
)
from apps.bayesian.normalization import (
    NormalizationParams,
    compute_normalization_params,
    create_normalizer_from_dataset,
    denormalize_displacements,
    denormalize_E,
    denormalize_elastic_modulus,
    normalize_displacements,
    normalize_E,
    normalize_elastic_modulus,
)

__all__ = [
    # Calibration
    "BayesianCalibrator",
    "EulerBernoulliCalibrator",
    "TimoshenkoCalibrator",
    "CalibrationResult",
    "PriorConfig",
    "ConvergenceError",
    "ConvergenceWarning",
    "create_default_priors",
    "create_timoshenko_priors",
    # Model selection
    "BayesianModelSelector",
    "ModelComparisonResult",
    "ModelEvidence",
    # Optimization
    "BayesianHyperparameterOptimizer",
    "FrequencyBasedModelSelector",
    "OptimizationResult",
    "get_physical_expectations",
    "create_priors_from_params",
    "run_quick_optimization",
    # Normalization
    "NormalizationParams",
    "compute_normalization_params",
    "create_normalizer_from_dataset",
    "normalize_displacements",
    "denormalize_displacements",
    "normalize_elastic_modulus",
    "denormalize_elastic_modulus",
    "normalize_E",
    "denormalize_E",
    # Bridge Sampling
    "BridgeSampler",
    "BridgeSamplingResult",
    "compute_bayes_factor_bridge",
]
