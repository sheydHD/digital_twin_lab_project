# Implementation Plan: Bayesian Model Selection Improvements

## Overview

This document outlines the step-by-step implementation plan to address all issues identified in the [Bayesian Audit Analysis](BAYESIAN_AUDIT_ANALYSIS.md).

**Estimated Total Time:** 4-6 hours  
**Priority Order:** Critical → Medium → Low

---

## Quick Reference: Changes by File

| File | Changes | Priority |
|------|---------|----------|
| `configs/default_config.yaml` | Update MCMC params, priors | HIGH |
| `apps/bayesian/normalization.py` | **NEW FILE** - Data normalization | HIGH |
| `apps/bayesian/bridge_sampling.py` | **NEW FILE** - True marginal likelihood | HIGH |
| `apps/bayesian/calibration.py` | Add normalization, R-hat checks | HIGH |
| `apps/bayesian/model_selection.py` | Use bridge sampling, diagnostics | MEDIUM |
| `apps/bayesian/__init__.py` | Export new modules | LOW |

---

## Phase 1: Quick Wins (Config Only) ⏱️ 5 minutes

### Task 1.1: Update MCMC Parameters

**File:** `configs/default_config.yaml`

**Changes:**
```yaml
# BEFORE
bayesian:
  n_samples: 800
  n_tune: 400
  n_chains: 2
  target_accept: 0.95

# AFTER
bayesian:
  n_samples: 1500     # Increased for WAIC stability
  n_tune: 800         # Longer warmup for adaptation
  n_chains: 4         # Better R-hat estimation
  target_accept: 0.95 # Keep as is
  
  # NEW: Convergence thresholds
  convergence:
    r_hat_threshold: 1.01      # Reject if exceeded
    r_hat_warning: 1.005       # Warn if exceeded
    ess_min_threshold: 400     # Minimum effective sample size
    retry_on_failure: true     # Auto-retry with more samples
    max_retries: 2             # Maximum retry attempts
```

**Rationale:**
- 1500 samples × 4 chains = 6000 total samples (vs. 1600 before)
- 4 chains enables reliable R-hat computation
- 800 tune samples allows better mass matrix adaptation

---

### Task 1.2: Widen Poisson Ratio Prior

**File:** `configs/default_config.yaml`

**Changes:**
```yaml
# BEFORE
priors:
  poisson_ratio:
    distribution: normal
    mu: 0.3
    sigma: 0.03

# AFTER
priors:
  poisson_ratio:
    distribution: normal
    mu: 0.3
    sigma: 0.05  # Widened to reduce Occam asymmetry
```

**Rationale:**
- Reduces unfair Occam penalty on Timoshenko model
- Still physically reasonable (ν ∈ [0.15, 0.45] at 3σ)
- More "agnostic" prior lets data drive selection

---

### Task 1.3: Add Convergence Configuration Section

**File:** `configs/default_config.yaml`

**Add new section:**
```yaml
# Convergence diagnostics settings
convergence:
  # R-hat thresholds (Gelman-Rubin statistic)
  r_hat_reject: 1.05       # Hard reject threshold
  r_hat_warn: 1.01         # Warning threshold
  
  # Effective sample size
  ess_bulk_min: 400        # Minimum bulk ESS
  ess_tail_min: 200        # Minimum tail ESS
  
  # Retry behavior
  retry_on_divergence: true
  max_retries: 2
  samples_multiplier: 1.5   # Multiply samples on retry
```

---

## Phase 2: Data Normalization ⏱️ 30 minutes

### Task 2.1: Create Normalization Module

**File:** `apps/bayesian/normalization.py` (NEW FILE)

**Contents:**
```python
"""
Data normalization utilities for stable MCMC sampling.

This module provides functions to normalize physical quantities
to O(1) scale for numerically stable Bayesian inference.

The key insight is that MCMC samplers (especially NUTS) work best
when all parameters are on similar scales. With beam mechanics:
- Displacements: 1e-6 to 1e-3 m
- Elastic modulus: 2.1e11 Pa
- Strains: 1e-9 to 1e-6

This 14+ order of magnitude range causes numerical issues.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizationParams:
    """
    Container for normalization parameters.
    
    Stores all scaling factors needed to convert between
    physical units and normalized (O(1)) units.
    
    Attributes:
        displacement_scale: Scale factor for displacements
        strain_scale: Scale factor for strains
        E_scale: Scale factor for elastic modulus (typically E_nominal)
        sigma_scale: Scale factor for observation noise
        is_active: Whether normalization is being used
    """
    displacement_scale: float = 1.0
    strain_scale: float = 1.0
    E_scale: float = 210e9  # Nominal steel E
    sigma_scale: float = 1.0
    is_active: bool = False
    
    # Store original data statistics for diagnostics
    _stats: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate normalization parameters."""
        if self.displacement_scale <= 0:
            raise ValueError("displacement_scale must be positive")
        if self.E_scale <= 0:
            raise ValueError("E_scale must be positive")


def compute_normalization_params(
    displacements: np.ndarray,
    strains: Optional[np.ndarray] = None,
    E_nominal: float = 210e9,
    method: str = "max_abs"
) -> NormalizationParams:
    """
    Compute normalization parameters from data.
    
    Args:
        displacements: Array of displacement measurements [m]
        strains: Optional array of strain measurements [-]
        E_nominal: Nominal elastic modulus for E scaling [Pa]
        method: Normalization method
            - "max_abs": Scale by maximum absolute value
            - "std": Scale by standard deviation
            - "range": Scale by (max - min)
    
    Returns:
        NormalizationParams with computed scale factors
    
    Example:
        >>> displacements = np.array([1e-5, 2e-5, 3e-5])
        >>> params = compute_normalization_params(displacements)
        >>> params.displacement_scale
        3e-05
    """
    # Compute displacement scale
    if method == "max_abs":
        disp_scale = np.max(np.abs(displacements))
    elif method == "std":
        disp_scale = np.std(displacements)
    elif method == "range":
        disp_scale = np.max(displacements) - np.min(displacements)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fallback for zero/tiny displacements
    if disp_scale < 1e-15:
        logger.warning(
            f"Very small displacement scale ({disp_scale:.2e}), "
            "using fallback of 1e-6"
        )
        disp_scale = 1e-6
    
    # Compute strain scale if provided
    if strains is not None:
        if method == "max_abs":
            strain_scale = np.max(np.abs(strains))
        else:
            strain_scale = np.std(strains) if method == "std" else \
                          (np.max(strains) - np.min(strains))
        
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
        is_active=True
    )
    
    # Store statistics for diagnostics
    params._stats = {
        "displacement_mean": float(np.mean(displacements)),
        "displacement_std": float(np.std(displacements)),
        "displacement_min": float(np.min(displacements)),
        "displacement_max": float(np.max(displacements)),
        "n_observations": len(displacements),
    }
    
    logger.debug(
        f"Normalization params: disp_scale={disp_scale:.2e}, "
        f"E_scale={E_nominal:.2e}"
    )
    
    return params


def normalize_displacements(
    displacements: np.ndarray,
    params: NormalizationParams
) -> np.ndarray:
    """
    Normalize displacement data to O(1) scale.
    
    Args:
        displacements: Raw displacement data [m]
        params: Normalization parameters
    
    Returns:
        Normalized displacements (dimensionless, O(1))
    """
    if not params.is_active:
        return displacements
    return displacements / params.displacement_scale


def denormalize_displacements(
    displacements_norm: np.ndarray,
    params: NormalizationParams
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
    params: NormalizationParams
) -> float:
    """Normalize elastic modulus to O(1)."""
    if not params.is_active:
        return E
    return E / params.E_scale


def denormalize_elastic_modulus(
    E_norm: float,
    params: NormalizationParams
) -> float:
    """Convert normalized E back to physical units [Pa]."""
    if not params.is_active:
        return E_norm
    return E_norm * params.E_scale


def normalize_sigma(
    sigma: float,
    params: NormalizationParams
) -> float:
    """Normalize observation noise."""
    if not params.is_active:
        return sigma
    return sigma / params.sigma_scale


def denormalize_sigma(
    sigma_norm: float,
    params: NormalizationParams
) -> float:
    """Convert normalized sigma back to physical units."""
    if not params.is_active:
        return sigma_norm
    return sigma_norm * params.sigma_scale


def get_normalized_prior_params(
    params: NormalizationParams,
    E_relative_uncertainty: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Get prior parameters appropriate for normalized space.
    
    When working in normalized space where E_norm ~ 1,
    we need to adjust the prior parameters accordingly.
    
    Args:
        params: Normalization parameters
        E_relative_uncertainty: Relative uncertainty on E (e.g., 0.05 = 5%)
    
    Returns:
        Dictionary of prior parameters for normalized space
    """
    return {
        "E_normalized": {
            # LogNormal centered at 1.0 in normalized space
            "mu": 0.0,  # ln(1) = 0
            "sigma": E_relative_uncertainty,
        },
        "sigma_normalized": {
            # HalfNormal for noise, scaled appropriately
            # In normalized space, expect sigma_norm ~ 0.001 to 0.01
            "sigma": 0.01,
        }
    }


class DataNormalizer:
    """
    Convenience class for managing data normalization.
    
    Example:
        >>> normalizer = DataNormalizer(displacements, E_nominal=210e9)
        >>> y_norm = normalizer.normalize_y(displacements)
        >>> E_norm = normalizer.normalize_E(210e9)
        >>> # ... run MCMC in normalized space ...
        >>> E_physical = normalizer.denormalize_E(E_norm_posterior)
    """
    
    def __init__(
        self,
        displacements: np.ndarray,
        strains: Optional[np.ndarray] = None,
        E_nominal: float = 210e9,
        method: str = "max_abs"
    ):
        """Initialize normalizer with data."""
        self.params = compute_normalization_params(
            displacements, strains, E_nominal, method
        )
        self._original_displacements = displacements.copy()
    
    @property
    def is_active(self) -> bool:
        """Check if normalization is active."""
        return self.params.is_active
    
    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize displacements."""
        return normalize_displacements(y, self.params)
    
    def denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize displacements."""
        return denormalize_displacements(y_norm, self.params)
    
    def normalize_E(self, E: float) -> float:
        """Normalize elastic modulus."""
        return normalize_elastic_modulus(E, self.params)
    
    def denormalize_E(self, E_norm: float) -> float:
        """Denormalize elastic modulus."""
        return denormalize_elastic_modulus(E_norm, self.params)
    
    def normalize_sigma(self, sigma: float) -> float:
        """Normalize observation noise."""
        return normalize_sigma(sigma, self.params)
    
    def denormalize_sigma(self, sigma_norm: float) -> float:
        """Denormalize observation noise."""
        return denormalize_sigma(sigma_norm, self.params)
    
    def get_normalized_priors(self) -> Dict[str, Dict[str, float]]:
        """Get prior parameters for normalized space."""
        return get_normalized_prior_params(self.params)
    
    def summary(self) -> str:
        """Get summary of normalization parameters."""
        return (
            f"DataNormalizer Summary:\n"
            f"  Displacement scale: {self.params.displacement_scale:.2e} m\n"
            f"  E scale: {self.params.E_scale:.2e} Pa\n"
            f"  Sigma scale: {self.params.sigma_scale:.2e}\n"
            f"  Active: {self.params.is_active}\n"
            f"  Original data stats:\n"
            f"    Mean: {self.params._stats.get('displacement_mean', 'N/A'):.2e}\n"
            f"    Std:  {self.params._stats.get('displacement_std', 'N/A'):.2e}\n"
            f"    Range: [{self.params._stats.get('displacement_min', 'N/A'):.2e}, "
            f"{self.params._stats.get('displacement_max', 'N/A'):.2e}]"
        )
```

---

### Task 2.2: Integrate Normalization into Calibrator

**File:** `apps/bayesian/calibration.py`

**Modifications Required:**

1. Add import at top:
```python
from .normalization import DataNormalizer, NormalizationParams
```

2. Add to `__init__`:
```python
def __init__(self, ...):
    ...
    self.use_normalization = True  # New parameter
    self._normalizer: Optional[DataNormalizer] = None
```

3. Modify `_build_euler_bernoulli_model`:
```python
def _build_euler_bernoulli_model(
    self, 
    data: SyntheticDataset,
    normalize: bool = True
) -> pm.Model:
    """Build E-B model with optional normalization."""
    
    if normalize:
        # Initialize normalizer
        self._normalizer = DataNormalizer(
            data.displacements,
            E_nominal=data.material.elastic_modulus
        )
        obs_normalized = self._normalizer.normalize_y(data.displacements)
        norm_priors = self._normalizer.get_normalized_priors()
    else:
        self._normalizer = None
        obs_normalized = data.displacements
    
    with pm.Model() as model:
        if normalize:
            # Normalized parameters (all ~O(1))
            E_norm = pm.LogNormal(
                "E_normalized",
                mu=norm_priors["E_normalized"]["mu"],
                sigma=norm_priors["E_normalized"]["sigma"]
            )
            sigma_norm = pm.HalfNormal(
                "sigma_normalized",
                sigma=norm_priors["sigma_normalized"]["sigma"]
            )
            
            # Convert back for physics calculation
            E = E_norm * self._normalizer.params.E_scale
            sigma = sigma_norm * self._normalizer.params.sigma_scale
        else:
            E = pm.LogNormal("elastic_modulus", mu=26.07, sigma=0.05)
            sigma = pm.HalfNormal("sigma", sigma=1e-6)
        
        # Physics prediction (always in physical units)
        predictions = self._compute_eb_deflection(E, data)
        
        if normalize:
            # Normalize predictions for comparison
            predictions_norm = predictions / self._normalizer.params.displacement_scale
            pm.Normal("obs", mu=predictions_norm, sigma=sigma_norm, observed=obs_normalized)
        else:
            pm.Normal("obs", mu=predictions, sigma=sigma, observed=data.displacements)
    
    return model
```

4. Add denormalization in results extraction:
```python
def _extract_results(self, trace) -> CalibrationResult:
    """Extract and denormalize results."""
    if self._normalizer is not None:
        # Denormalize posterior samples
        E_samples = trace.posterior["E_normalized"].values * self._normalizer.params.E_scale
        sigma_samples = trace.posterior["sigma_normalized"].values * self._normalizer.params.sigma_scale
    else:
        E_samples = trace.posterior["elastic_modulus"].values
        sigma_samples = trace.posterior["sigma"].values
    
    # ... rest of extraction ...
```

---

## Phase 3: Convergence Validation ⏱️ 20 minutes

### Task 3.1: Add Convergence Validator

**File:** `apps/bayesian/calibration.py`

**Add new method:**
```python
class ConvergenceError(Exception):
    """Raised when MCMC fails to converge."""
    pass


class ConvergenceWarning(UserWarning):
    """Warning for marginal MCMC convergence."""
    pass


def _validate_convergence(
    self, 
    trace: az.InferenceData,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Validate MCMC convergence before accepting results.
    
    Args:
        trace: ArviZ InferenceData with posterior samples
        strict: If True, raise errors for poor convergence
    
    Returns:
        Dictionary with convergence diagnostics
    
    Raises:
        ConvergenceError: If R-hat > threshold (when strict=True)
    """
    import warnings
    
    # Compute summary statistics
    summary = az.summary(trace)
    
    # Extract key diagnostics
    r_hat_max = float(summary["r_hat"].max())
    r_hat_mean = float(summary["r_hat"].mean())
    ess_bulk_min = float(summary["ess_bulk"].min())
    ess_tail_min = float(summary["ess_tail"].min())
    
    diagnostics = {
        "r_hat_max": r_hat_max,
        "r_hat_mean": r_hat_mean,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "converged": True,
        "warnings": [],
    }
    
    # Check R-hat
    r_hat_reject = self.config.get("convergence", {}).get("r_hat_reject", 1.05)
    r_hat_warn = self.config.get("convergence", {}).get("r_hat_warn", 1.01)
    
    if r_hat_max > r_hat_reject:
        diagnostics["converged"] = False
        msg = (
            f"MCMC failed to converge: R-hat = {r_hat_max:.4f} > {r_hat_reject}. "
            "Evidence values are unreliable."
        )
        if strict:
            raise ConvergenceError(msg)
        else:
            diagnostics["warnings"].append(msg)
    
    elif r_hat_max > r_hat_warn:
        msg = (
            f"Marginal convergence: R-hat = {r_hat_max:.4f} > {r_hat_warn}. "
            "Evidence may have ~10-20% error."
        )
        warnings.warn(msg, ConvergenceWarning)
        diagnostics["warnings"].append(msg)
    
    # Check ESS
    ess_min = self.config.get("convergence", {}).get("ess_bulk_min", 400)
    
    if ess_bulk_min < ess_min:
        msg = (
            f"Low effective sample size: ESS = {ess_bulk_min:.0f} < {ess_min}. "
            "Consider increasing n_samples."
        )
        warnings.warn(msg, ConvergenceWarning)
        diagnostics["warnings"].append(msg)
    
    return diagnostics
```

### Task 3.2: Integrate Validation into Calibration Flow

**File:** `apps/bayesian/calibration.py`

**Modify `calibrate` method:**
```python
def calibrate(self, data: SyntheticDataset, ...) -> CalibrationResult:
    """Run calibration with convergence validation."""
    
    # ... existing model building and sampling code ...
    
    # NEW: Validate convergence before proceeding
    convergence_diagnostics = self._validate_convergence(trace, strict=True)
    
    if not convergence_diagnostics["converged"]:
        # Optionally retry with more samples
        if self.config.get("convergence", {}).get("retry_on_failure", False):
            retries = self.config.get("convergence", {}).get("max_retries", 2)
            multiplier = self.config.get("convergence", {}).get("samples_multiplier", 1.5)
            
            for attempt in range(retries):
                logger.warning(f"Retry {attempt + 1}/{retries} with more samples")
                self.n_samples = int(self.n_samples * multiplier)
                self.n_tune = int(self.n_tune * multiplier)
                
                # Re-run sampling
                trace = self._sample(model)
                convergence_diagnostics = self._validate_convergence(trace, strict=False)
                
                if convergence_diagnostics["converged"]:
                    break
    
    # Include diagnostics in result
    result = self._extract_results(trace)
    result.convergence_diagnostics = convergence_diagnostics
    
    return result
```

---

## Phase 4: Bridge Sampling (True Marginal Likelihood) ⏱️ 1-2 hours

### Task 4.1: Create Bridge Sampling Module

**File:** `apps/bayesian/bridge_sampling.py` (NEW FILE)

**Contents:** (See full implementation in Analysis document, Section 2.3)

Key components:
- `BridgeSampler` class
- `estimate()` method using iterative bridge sampling algorithm
- `compute_bridge_sampling_bf()` convenience function

### Task 4.2: Integrate Bridge Sampling Option

**File:** `apps/bayesian/model_selection.py`

**Modifications:**

1. Add import:
```python
from .bridge_sampling import BridgeSampler, compute_bridge_sampling_bf
```

2. Add method:
```python
def _compute_bayes_factor_bridge(
    self,
    eb_result: CalibrationResult,
    timo_result: CalibrationResult,
    data: SyntheticDataset
) -> Tuple[float, float]:
    """
    Compute Bayes factor using bridge sampling.
    
    Returns:
        Tuple of (log_bayes_factor, standard_error)
    """
    # Define log-likelihood functions
    def eb_log_likelihood(params):
        E = params[0]
        sigma = params[1]
        predictions = eb_result.model_predictions
        return np.sum(norm.logpdf(data.displacements, predictions, sigma))
    
    def timo_log_likelihood(params):
        E = params[0]
        nu = params[1]
        sigma = params[2]
        predictions = timo_result.model_predictions
        return np.sum(norm.logpdf(data.displacements, predictions, sigma))
    
    # Define log-prior functions
    def eb_log_prior(params):
        E, sigma = params
        log_p = lognorm.logpdf(E, s=0.05, scale=np.exp(26.07))
        log_p += halfnorm.logpdf(sigma, scale=1e-6)
        return log_p
    
    def timo_log_prior(params):
        E, nu, sigma = params
        log_p = lognorm.logpdf(E, s=0.05, scale=np.exp(26.07))
        log_p += norm.logpdf(nu, loc=0.3, scale=0.05)
        log_p += halfnorm.logpdf(sigma, scale=1e-6)
        return log_p
    
    log_bf, se = compute_bridge_sampling_bf(
        eb_result.trace, timo_result.trace,
        eb_log_likelihood, timo_log_likelihood,
        eb_log_prior, timo_log_prior
    )
    
    return log_bf, se
```

3. Modify `compare_models`:
```python
def compare_models(
    self,
    eb_result: CalibrationResult,
    timo_result: CalibrationResult,
    data: SyntheticDataset,
    method: str = "waic"  # or "bridge"
) -> ModelComparison:
    """Compare models using specified method."""
    
    if method == "bridge":
        log_bf, se = self._compute_bayes_factor_bridge(eb_result, timo_result, data)
    else:
        # Existing WAIC-based comparison
        log_bf = eb_result.waic.elpd_waic - timo_result.waic.elpd_waic
        se = np.sqrt(eb_result.waic.se**2 + timo_result.waic.se**2)
    
    # ... rest of comparison logic ...
```

---

## Phase 5: Update Exports and Config ⏱️ 10 minutes

### Task 5.1: Update Module Exports

**File:** `apps/bayesian/__init__.py`

```python
from .calibration import (
    BayesianCalibrator,
    CalibrationResult,
    ConvergenceError,
    ConvergenceWarning,
)
from .model_selection import ModelSelector, ModelComparison
from .normalization import (
    DataNormalizer,
    NormalizationParams,
    compute_normalization_params,
)
from .bridge_sampling import BridgeSampler, compute_bridge_sampling_bf

__all__ = [
    "BayesianCalibrator",
    "CalibrationResult",
    "ConvergenceError",
    "ConvergenceWarning",
    "ModelSelector",
    "ModelComparison",
    "DataNormalizer",
    "NormalizationParams",
    "compute_normalization_params",
    "BridgeSampler",
    "compute_bridge_sampling_bf",
]
```

### Task 5.2: Final Config Updates

**File:** `configs/default_config.yaml`

Complete updated config:
```yaml
# Bayesian inference settings
bayesian:
  n_samples: 1500     # MCMC samples per chain
  n_tune: 800         # Tuning/warmup samples
  n_chains: 4         # Number of chains
  target_accept: 0.95 # Target acceptance rate
  random_seed: 42
  
  # Normalization settings
  normalization:
    enabled: true           # Enable data normalization
    method: "max_abs"       # Normalization method
  
  # Evidence estimation method
  evidence_method: "waic"   # Options: "waic", "bridge"

# Convergence diagnostics
convergence:
  r_hat_reject: 1.05        # Hard reject threshold
  r_hat_warn: 1.01          # Warning threshold
  ess_bulk_min: 400         # Minimum bulk ESS
  ess_tail_min: 200         # Minimum tail ESS
  retry_on_failure: true    # Auto-retry on convergence failure
  max_retries: 2            # Maximum retry attempts
  samples_multiplier: 1.5   # Multiply samples on retry

# Prior distributions
priors:
  elastic_modulus:
    distribution: lognormal
    mu: 26.07
    sigma: 0.05
  poisson_ratio:
    distribution: normal
    mu: 0.3
    sigma: 0.05  # Widened from 0.03
  observation_noise:
    distribution: halfnormal
    sigma: 1.0e-6
```

---

## Phase 6: Testing & Validation ⏱️ 30 minutes

### Task 6.1: Add Unit Tests

**File:** `tests/test_normalization.py` (NEW FILE)

```python
"""Tests for data normalization module."""

import numpy as np
import pytest
from apps.bayesian.normalization import (
    DataNormalizer,
    compute_normalization_params,
    normalize_displacements,
    denormalize_displacements,
)


class TestNormalization:
    """Test normalization functions."""
    
    def test_round_trip(self):
        """Test that normalize -> denormalize returns original."""
        displacements = np.array([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
        params = compute_normalization_params(displacements)
        
        normalized = normalize_displacements(displacements, params)
        recovered = denormalize_displacements(normalized, params)
        
        np.testing.assert_allclose(displacements, recovered, rtol=1e-10)
    
    def test_normalized_scale(self):
        """Test that normalized data is O(1)."""
        displacements = np.array([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
        params = compute_normalization_params(displacements)
        
        normalized = normalize_displacements(displacements, params)
        
        assert np.max(np.abs(normalized)) <= 1.0
        assert np.max(np.abs(normalized)) >= 0.1
    
    def test_data_normalizer_class(self):
        """Test DataNormalizer convenience class."""
        displacements = np.random.randn(100) * 1e-5
        
        normalizer = DataNormalizer(displacements)
        
        y_norm = normalizer.normalize_y(displacements)
        y_back = normalizer.denormalize_y(y_norm)
        
        np.testing.assert_allclose(displacements, y_back, rtol=1e-10)
```

### Task 6.2: Add Convergence Tests

**File:** `tests/test_convergence.py` (NEW FILE)

```python
"""Tests for convergence validation."""

import pytest
import numpy as np
import arviz as az
from apps.bayesian.calibration import (
    BayesianCalibrator,
    ConvergenceError,
    ConvergenceWarning,
)


class TestConvergenceValidation:
    """Test convergence checking logic."""
    
    def test_good_convergence_passes(self):
        """Test that well-converged chains pass validation."""
        # Create mock trace with good R-hat
        # ... test implementation ...
        pass
    
    def test_bad_convergence_raises(self):
        """Test that poorly-converged chains raise error."""
        # Create mock trace with bad R-hat
        # ... test implementation ...
        pass
    
    def test_warning_on_marginal_convergence(self):
        """Test that marginal convergence generates warning."""
        # ... test implementation ...
        pass
```

### Task 6.3: Integration Test

**File:** `tests/test_integration.py`

```python
"""Integration tests for full pipeline with improvements."""

def test_full_pipeline_with_normalization():
    """Test that pipeline runs with normalization enabled."""
    # Run pipeline on single L/h ratio
    # Verify normalization was applied
    # Verify results are in physical units
    pass

def test_bridge_sampling_vs_waic():
    """Compare bridge sampling and WAIC results."""
    # Run both methods
    # Verify they agree in direction (same model preferred)
    # Bridge sampling should have lower variance
    pass
```

---

## Execution Checklist

### Pre-Implementation
- [ ] Backup current working code
- [ ] Create feature branch: `git checkout -b feature/bayesian-improvements`
- [ ] Review this plan with stakeholders

### Phase 1: Config Changes
- [ ] Update `bayesian` section in config
- [ ] Update `priors` section in config
- [ ] Add `convergence` section to config
- [ ] Test: `python main.py --stage calibrate` still works

### Phase 2: Normalization
- [ ] Create `apps/bayesian/normalization.py`
- [ ] Add unit tests for normalization
- [ ] Integrate into `calibration.py`
- [ ] Test: Run calibration, verify normalized space sampling

### Phase 3: Convergence Validation
- [ ] Add `ConvergenceError` and `ConvergenceWarning` classes
- [ ] Implement `_validate_convergence` method
- [ ] Add retry logic
- [ ] Test: Verify R-hat checking works

### Phase 4: Bridge Sampling
- [ ] Create `apps/bayesian/bridge_sampling.py`
- [ ] Implement `BridgeSampler` class
- [ ] Integrate into `model_selection.py`
- [ ] Test: Compare bridge sampling vs WAIC results

### Phase 5: Finalization
- [ ] Update `__init__.py` exports
- [ ] Run full test suite
- [ ] Run full pipeline: `python main.py`
- [ ] Compare results before/after improvements
- [ ] Document any behavior changes

### Post-Implementation
- [ ] Code review
- [ ] Merge to main branch
- [ ] Update documentation

---

## Expected Outcomes

After implementing all changes:

| Metric | Before | After |
|--------|--------|-------|
| WAIC warnings | Frequent | Rare |
| R-hat values | Unchecked | < 1.01 enforced |
| Numerical stability | Occasional issues | Robust |
| Bayes factor accuracy | Approximate (WAIC) | Exact (Bridge) |
| L/h=50 handling | Inconsistent | Correct (EB preferred) |
| Occam penalty | Asymmetric | Balanced |

---

## Rollback Plan

If issues arise:

1. **Immediate rollback:** `git checkout main`
2. **Partial rollback:** Disable features via config:
   ```yaml
   bayesian:
     normalization:
       enabled: false
     evidence_method: "waic"  # Instead of "bridge"
   ```
3. **Keep improvements but reduce intensity:**
   ```yaml
   bayesian:
     n_samples: 800   # Reduce back
     n_chains: 2      # Reduce back
   ```
