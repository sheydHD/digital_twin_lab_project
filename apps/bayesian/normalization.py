"""
Data Normalization for Stable MCMC Sampling.

This module provides utilities to normalize physical quantities to O(1) scale
for numerically stable Bayesian inference with PyMC.

The Problem:
-----------
Beam mechanics involves quantities spanning 14+ orders of magnitude:
- Displacements: 1e-6 to 1e-3 m
- Elastic modulus: 2.1e11 Pa
- Strains: 1e-9 to 1e-6

This range causes numerical issues in MCMC:
1. Gradient instability in NUTS sampler
2. Poor mass matrix estimation
3. Step size adaptation problems

The Solution:
------------
Normalize all quantities to O(1) scale before sampling, then denormalize
the results back to physical units.

Reference:
- Stan Best Practices: https://mc-stan.org/docs/stan-users-guide/
- PyMC Documentation: https://www.pymc.io/
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormalizationParams:
    """
    Container for reversible normalization parameters.

    All scale factors are chosen so that normalized quantities are O(1),
    which is optimal for MCMC samplers.

    Attributes:
        displacement_scale: Scale factor for displacements [m]
        strain_scale: Scale factor for strains [-]
        E_scale: Scale factor for elastic modulus [Pa]
        sigma_scale: Scale factor for observation noise
        is_active: Whether normalization is enabled
    """

    displacement_scale: float = 1.0
    strain_scale: float = 1.0
    E_scale: float = 210e9  # Nominal steel E
    sigma_scale: float = 1.0
    is_active: bool = False

    # Internal storage for diagnostics
    _stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate normalization parameters."""
        if self.displacement_scale <= 0:
            raise ValueError("displacement_scale must be positive")
        if self.E_scale <= 0:
            raise ValueError("E_scale must be positive")

    def summary(self) -> str:
        """Return human-readable summary of normalization params."""
        return (
            f"NormalizationParams(\n"
            f"  displacement_scale={self.displacement_scale:.2e} m,\n"
            f"  E_scale={self.E_scale:.2e} Pa,\n"
            f"  sigma_scale={self.sigma_scale:.2e},\n"
            f"  is_active={self.is_active}\n"
            f")"
        )


def compute_normalization_params(
    displacements: np.ndarray,
    strains: Optional[np.ndarray] = None,
    E_nominal: float = 210e9,
    method: str = "max_abs",
) -> NormalizationParams:
    """
    Compute normalization parameters from measurement data.

    The goal is to choose scale factors such that all normalized
    quantities are O(1), which is optimal for MCMC sampling.

    Args:
        displacements: Array of displacement measurements [m]
        strains: Optional array of strain measurements [-]
        E_nominal: Nominal elastic modulus for E scaling [Pa]
        method: Normalization method
            - "max_abs": Scale by maximum absolute value (default)
            - "std": Scale by standard deviation
            - "range": Scale by (max - min)

    Returns:
        NormalizationParams with computed scale factors

    Example:
        >>> displacements = np.array([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
        >>> params = compute_normalization_params(displacements)
        >>> params.displacement_scale
        5e-05
        >>> normalized = displacements / params.displacement_scale
        >>> np.max(np.abs(normalized))  # Should be ~1.0
        1.0
    """
    # Compute displacement scale
    if method == "max_abs":
        disp_scale = np.max(np.abs(displacements))
    elif method == "std":
        disp_scale = np.std(displacements)
        if disp_scale < 1e-15:
            disp_scale = np.max(np.abs(displacements))
    elif method == "range":
        disp_scale = np.max(displacements) - np.min(displacements)
        if disp_scale < 1e-15:
            disp_scale = np.max(np.abs(displacements))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Fallback for zero/tiny displacements
    if disp_scale < 1e-15:
        logger.warning(
            f"Very small displacement scale ({disp_scale:.2e}), "
            "using fallback of 1e-6 m"
        )
        disp_scale = 1e-6

    # Compute strain scale if provided
    if strains is not None and len(strains) > 0:
        if method == "max_abs":
            strain_scale = np.max(np.abs(strains))
        elif method == "std":
            strain_scale = np.std(strains)
        else:
            strain_scale = np.max(strains) - np.min(strains)

        if strain_scale < 1e-15:
            strain_scale = 1e-6
    else:
        strain_scale = 1.0

    # Create params object
    params = NormalizationParams(
        displacement_scale=float(disp_scale),
        strain_scale=float(strain_scale),
        E_scale=float(E_nominal),
        sigma_scale=float(disp_scale),  # Noise scales with displacement
        is_active=True,
    )

    # Store statistics for diagnostics
    params._stats = {
        "displacement_mean": float(np.mean(displacements)),
        "displacement_std": float(np.std(displacements)),
        "displacement_min": float(np.min(displacements)),
        "displacement_max": float(np.max(displacements)),
        "n_observations": len(displacements),
        "method": method,
    }

    logger.debug(
        f"Computed normalization: disp_scale={disp_scale:.2e}, "
        f"E_scale={E_nominal:.2e}"
    )

    return params


def normalize_displacements(
    displacements: np.ndarray,
    params: NormalizationParams,
) -> np.ndarray:
    """
    Normalize displacement data to O(1) scale.

    Args:
        displacements: Raw displacement data [m]
        params: Normalization parameters

    Returns:
        Normalized displacements (dimensionless, typically in [-1, 1])
    """
    if not params.is_active:
        return displacements
    return displacements / params.displacement_scale


def denormalize_displacements(
    displacements_norm: np.ndarray,
    params: NormalizationParams,
) -> np.ndarray:
    """
    Convert normalized displacements back to physical units.

    Args:
        displacements_norm: Normalized displacements
        params: Normalization parameters

    Returns:
        Displacements in physical units [m]
    """
    if not params.is_active:
        return displacements_norm
    return displacements_norm * params.displacement_scale


def normalize_elastic_modulus(
    E: float,
    params: NormalizationParams,
) -> float:
    """
    Normalize elastic modulus to O(1).

    After normalization, E_norm ≈ 1.0 for typical steel.

    Args:
        E: Elastic modulus [Pa]
        params: Normalization parameters

    Returns:
        Normalized elastic modulus (dimensionless, ~1.0)
    """
    if not params.is_active:
        return E
    return E / params.E_scale


def denormalize_elastic_modulus(
    E_norm: float,
    params: NormalizationParams,
) -> float:
    """
    Convert normalized E back to physical units [Pa].

    Args:
        E_norm: Normalized elastic modulus
        params: Normalization parameters

    Returns:
        Elastic modulus in physical units [Pa]
    """
    if not params.is_active:
        return E_norm
    return E_norm * params.E_scale


def normalize_sigma(
    sigma: float,
    params: NormalizationParams,
) -> float:
    """
    Normalize observation noise.

    Sigma should scale with displacements since the likelihood is:
    y_obs ~ Normal(y_pred, sigma)

    Args:
        sigma: Observation noise [m]
        params: Normalization parameters

    Returns:
        Normalized sigma (dimensionless)
    """
    if not params.is_active:
        return sigma
    return sigma / params.sigma_scale


def denormalize_sigma(
    sigma_norm: float,
    params: NormalizationParams,
) -> float:
    """
    Convert normalized sigma back to physical units.

    Args:
        sigma_norm: Normalized sigma
        params: Normalization parameters

    Returns:
        Sigma in physical units [m]
    """
    if not params.is_active:
        return sigma_norm
    return sigma_norm * params.sigma_scale


def get_normalized_prior_params(
    params: NormalizationParams,
    E_relative_uncertainty: float = 0.05,
    sigma_prior_scale: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """
    Get prior parameters appropriate for normalized space.

    When working in normalized space where E_norm ~ 1 and y_norm ~ 1,
    we need priors that make sense at this scale.

    Args:
        params: Normalization parameters
        E_relative_uncertainty: Relative uncertainty on E (e.g., 0.05 = 5%)
        sigma_prior_scale: Scale for sigma prior in normalized space

    Returns:
        Dictionary of prior parameters for normalized space

    Example:
        >>> params = compute_normalization_params(displacements)
        >>> priors = get_normalized_prior_params(params)
        >>> # E_norm ~ LogNormal(0, 0.05) -> E_norm ~ 1.0 with 5% uncertainty
        >>> priors["E_normalized"]["mu"]
        0.0
    """
    return {
        "E_normalized": {
            # LogNormal centered at 1.0 in normalized space
            # ln(1) = 0, so mu=0 gives mode at 1.0
            "mu": 0.0,
            "sigma": E_relative_uncertainty,
        },
        "sigma_normalized": {
            # HalfNormal for noise, scaled appropriately
            # In normalized space, expect sigma_norm ~ 0.001 to 0.01
            "sigma": sigma_prior_scale,
        },
    }


class DataNormalizer:
    """
    Convenience class for managing data normalization in Bayesian calibration.

    This class wraps the normalization functions and provides a clean
    interface for normalizing/denormalizing data and parameters.

    Example:
        >>> # Initialize with data
        >>> normalizer = DataNormalizer(displacements, E_nominal=210e9)
        >>>
        >>> # Normalize observations for MCMC
        >>> y_norm = normalizer.normalize_y(displacements)
        >>> E_norm_prior_params = normalizer.get_normalized_priors()
        >>>
        >>> # ... run MCMC in normalized space ...
        >>>
        >>> # Denormalize posterior samples
        >>> E_physical = normalizer.denormalize_E(E_norm_posterior)
        >>> sigma_physical = normalizer.denormalize_sigma(sigma_norm_posterior)
    """

    def __init__(
        self,
        displacements: np.ndarray,
        strains: Optional[np.ndarray] = None,
        E_nominal: float = 210e9,
        method: str = "max_abs",
        enabled: bool = True,
    ):
        """
        Initialize normalizer with measurement data.

        Args:
            displacements: Displacement measurements [m]
            strains: Optional strain measurements [-]
            E_nominal: Nominal elastic modulus [Pa]
            method: Normalization method ("max_abs", "std", "range")
            enabled: Whether to enable normalization
        """
        if enabled:
            self.params = compute_normalization_params(
                displacements, strains, E_nominal, method
            )
        else:
            self.params = NormalizationParams(is_active=False)

        self._original_displacements = displacements.copy()
        self._E_nominal = E_nominal

    @property
    def is_active(self) -> bool:
        """Check if normalization is active."""
        return self.params.is_active

    @property
    def displacement_scale(self) -> float:
        """Get displacement scale factor."""
        return self.params.displacement_scale

    @property
    def E_scale(self) -> float:
        """Get elastic modulus scale factor."""
        return self.params.E_scale

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize displacements to O(1)."""
        return normalize_displacements(y, self.params)

    def denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize displacements to physical units [m]."""
        return denormalize_displacements(y_norm, self.params)

    def normalize_E(self, E: float) -> float:
        """Normalize elastic modulus to O(1)."""
        return normalize_elastic_modulus(E, self.params)

    def denormalize_E(self, E_norm: float) -> float:
        """Denormalize E to physical units [Pa]."""
        return denormalize_elastic_modulus(E_norm, self.params)

    def normalize_sigma(self, sigma: float) -> float:
        """Normalize observation noise."""
        return normalize_sigma(sigma, self.params)

    def denormalize_sigma(self, sigma_norm: float) -> float:
        """Denormalize sigma to physical units [m]."""
        return denormalize_sigma(sigma_norm, self.params)

    def get_normalized_priors(
        self,
        E_relative_uncertainty: float = 0.05,
        sigma_prior_scale: float = 0.01,
    ) -> Dict[str, Dict[str, float]]:
        """Get prior parameters for normalized space."""
        return get_normalized_prior_params(
            self.params, E_relative_uncertainty, sigma_prior_scale
        )

    def normalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Normalize model predictions (same as displacements)."""
        return self.normalize_y(predictions)

    def compute_normalized_physics_scale(self) -> float:
        """
        Compute the scale factor for physics predictions.

        When computing deflection in normalized space:
        w_norm = (P * x^2 / (6 * E_norm * I)) * (3*L - x) / disp_scale

        This method returns disp_scale for use in normalized physics models.
        """
        return self.params.displacement_scale

    def summary(self) -> str:
        """Get human-readable summary of normalizer state."""
        if not self.is_active:
            return "DataNormalizer: INACTIVE (no normalization applied)"

        stats = self.params._stats
        return (
            f"DataNormalizer Summary:\n"
            f"  Status: ACTIVE\n"
            f"  Displacement scale: {self.params.displacement_scale:.2e} m\n"
            f"  E scale: {self.params.E_scale:.2e} Pa\n"
            f"  Sigma scale: {self.params.sigma_scale:.2e}\n"
            f"  Method: {stats.get('method', 'unknown')}\n"
            f"  Original data stats:\n"
            f"    Mean: {stats.get('displacement_mean', 'N/A'):.2e} m\n"
            f"    Std:  {stats.get('displacement_std', 'N/A'):.2e} m\n"
            f"    Range: [{stats.get('displacement_min', 'N/A'):.2e}, "
            f"{stats.get('displacement_max', 'N/A'):.2e}] m\n"
            f"    N obs: {stats.get('n_observations', 'N/A')}"
        )

    def __repr__(self) -> str:
        """String representation."""
        if self.is_active:
            return (
                f"DataNormalizer(disp_scale={self.displacement_scale:.2e}, "
                f"E_scale={self.E_scale:.2e}, active=True)"
            )
        return "DataNormalizer(active=False)"


def validate_normalized_data(
    y_norm: np.ndarray,
    E_norm: float,
    sigma_norm: float,
    warn_threshold: float = 10.0,
) -> Tuple[bool, str]:
    """
    Validate that normalized data is in appropriate range.

    Good normalization should produce quantities in range [-10, 10]
    approximately. Larger values may indicate normalization issues.

    Args:
        y_norm: Normalized observations
        E_norm: Normalized elastic modulus
        sigma_norm: Normalized sigma
        warn_threshold: Threshold for warnings (default 10)

    Returns:
        Tuple of (is_valid, message)
    """
    issues = []

    if np.max(np.abs(y_norm)) > warn_threshold:
        issues.append(
            f"y_norm max abs = {np.max(np.abs(y_norm)):.1f} > {warn_threshold}"
        )

    if E_norm > warn_threshold or E_norm < 1 / warn_threshold:
        issues.append(f"E_norm = {E_norm:.2f} outside [0.1, 10] range")

    if sigma_norm > warn_threshold:
        issues.append(f"sigma_norm = {sigma_norm:.2f} > {warn_threshold}")

    if issues:
        return False, "Normalization issues: " + "; ".join(issues)

    return True, "Normalization looks good"


# Aliases for convenience
normalize_E = normalize_elastic_modulus
denormalize_E = denormalize_elastic_modulus


def create_normalizer_from_dataset(dataset) -> NormalizationParams:
    """
    Create normalization parameters from a SyntheticDataset.

    Args:
        dataset: SyntheticDataset object with displacements and/or strains

    Returns:
        NormalizationParams computed from the dataset
    """
    displacements = getattr(dataset, 'displacements', None)
    strains = getattr(dataset, 'strains', None)

    if displacements is None or len(displacements) == 0:
        logger.warning("No displacements in dataset, using default normalization")
        return NormalizationParams(is_active=False)

    return compute_normalization_params(
        displacements=np.asarray(displacements),
        strains=np.asarray(strains) if strains is not None else None,
    )
