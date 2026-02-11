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
from typing import Any, Dict, Optional

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
