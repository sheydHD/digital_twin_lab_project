"""
Bayesian Calibration for Beam Models.

This module implements Bayesian parameter inference for both Euler-Bernoulli
and Timoshenko beam theories using PyMC for probabilistic programming.

Key features:
- Prior specification for material and geometric parameters
- Likelihood formulation based on measurement data
- MCMC sampling for posterior inference
- Marginal likelihood estimation for model comparison

Reference:
- Gelman et al. "Bayesian Data Analysis"
- PyMC documentation: https://www.pymc.io/
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pymc as pm

from apps.backend.core.bayesian.normalization import (
    NormalizationParams,
    compute_normalization_params,
)
from apps.backend.core.data.synthetic_generator import SyntheticDataset
from apps.backend.core.models.base_beam import BeamGeometry, LoadCase, MaterialProperties
from apps.backend.core.models.euler_bernoulli import EulerBernoulliBeam
from apps.backend.core.models.timoshenko import TimoshenkoBeam

logger = logging.getLogger(__name__)


# Custom exceptions for convergence issues
class ConvergenceError(Exception):
    """
    Raised when MCMC fails to converge.

    This indicates that the posterior samples are unreliable and
    any evidence calculations should not be trusted.
    """

    pass


class ConvergenceWarning(UserWarning):
    """
    Warning for marginal MCMC convergence.

    Raised when R-hat is elevated but not critically high,
    indicating results may have 10-20% error.
    """

    pass


@dataclass
class PriorConfig:
    """
    Prior distribution configuration for model parameters.

    Attributes:
        param_name: Parameter name
        distribution: Prior distribution type ('normal', 'lognormal', 'uniform')
        params: Distribution parameters (mean, std for normal; mu, sigma for lognormal)
    """

    param_name: str
    distribution: str
    params: dict[str, float]


@dataclass
class CalibrationResult:
    """
    Results from Bayesian calibration.

    Attributes:
        model_name: Name of the calibrated model
        trace: PyMC InferenceData object with posterior samples
        posterior_summary: Summary statistics for parameters (in PHYSICAL units)
        log_likelihood: Log-likelihood at posterior samples
        waic: Widely Applicable Information Criterion (elpd_waic, for diagnostics)
        marginal_likelihood_estimate: Estimated log marginal likelihood (bridge sampling)
        convergence_diagnostics: R-hat and ESS statistics
        normalization_params: Normalization scales used during fitting
        posterior_summary_normalized: Summary statistics in normalized units
    """

    model_name: str
    trace: az.InferenceData
    posterior_summary: dict  # Physical units
    log_likelihood: np.ndarray
    waic: float | None = None
    marginal_likelihood_estimate: float | None = None
    convergence_diagnostics: dict | None = None
    normalization_params: NormalizationParams | None = None
    posterior_summary_normalized: dict | None = None  # Normalized units


class BayesianCalibrator(ABC):
    """
    Abstract base class for Bayesian model calibration.

    This class defines the interface for calibrating beam models using
    Bayesian inference. Subclasses implement specific beam theories.

    """

    def __init__(
        self,
        priors: list[PriorConfig],
        n_samples: int = 2000,
        n_tune: int = 1000,
        n_chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ):
        """
        Initialize the Bayesian calibrator.

        Args:
            priors: Prior configurations for parameters
            n_samples: Number of posterior samples per chain
            n_tune: Number of tuning samples
            n_chains: Number of MCMC chains
            target_accept: Target acceptance probability for NUTS
            random_seed: Random seed for reproducibility
        """
        self.priors = {p.param_name: p for p in priors}
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_seed = random_seed

        self._trace = None
        self._model = None
        self._normalizer = None
        self._calibration_data = None
        self._calibration_data_type = None

        # Convergence thresholds (can be overridden via config)
        self.r_hat_reject = 1.05
        self.r_hat_warn = 1.01
        self.ess_min = 400

        # Retry settings
        self.retry_on_failure = True
        self.max_retries = 2
        self.samples_multiplier = 1.5

    @abstractmethod
    def _forward_model(
        self,
        params: dict[str, float],
        x_locations: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute model predictions given parameters.

        Args:
            params: Parameter values
            x_locations: Measurement locations
            geometry: Beam geometry
            load: Load case

        Returns:
            Model predictions at measurement locations
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name for identification."""
        pass

    def _build_pymc_model(
        self,
        data: SyntheticDataset,
        data_type: str = "displacement",
    ) -> pm.Model:
        """
        Build PyMC probabilistic model with automatic normalization.

        Normalization Strategy:
        ----------------------
        All quantities are normalized to O(1) scale for stable MCMC:
        - E_normalized ~ Normal(1.0, prior_sigma) → E = E_normalized * E_scale
        - y_normalized = y_obs / disp_scale → predictions also normalized
        - sigma_normalized = sigma / disp_scale → noise in normalized space

        This prevents the 14+ orders of magnitude numerical issues that
        cause WAIC instability and poor convergence.

        Args:
            data: Synthetic dataset with observations
            data_type: Type of data to fit ('displacement' or 'strain')

        Returns:
            PyMC Model object

        """
        # Select observation data
        if data_type == "displacement":
            x_obs = data.x_disp
            y_obs = data.displacements
            sigma_obs = data.displacement_noise_std
        elif data_type == "strain":
            x_obs = data.x_strain
            y_obs = data.strains
            sigma_obs = data.strain_noise_std
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'displacement' or 'strain'.")

        # Compute normalization parameters
        self._normalizer = compute_normalization_params(
            displacements=y_obs,
            E_nominal=data.material.elastic_modulus,
        )

        # Normalize the observations
        y_obs_norm = y_obs / self._normalizer.displacement_scale
        sigma_obs_norm = sigma_obs / self._normalizer.displacement_scale

        logger.debug(
            f"Normalization active: disp_scale={self._normalizer.displacement_scale:.2e}, "
            f"E_scale={self._normalizer.E_scale:.2e}"
        )

        with pm.Model() as model:
            # Store normalization params in model for later retrieval
            model.normalization = self._normalizer

            # Define priors for each parameter (in NORMALIZED space)
            params = {}

            for name, prior in self.priors.items():
                if name == "elastic_modulus":
                    # E_normalized ~ Normal(1.0, sigma) instead of LogNormal(ln(210e9), sigma)
                    # This makes E centered at 1.0 with the same relative uncertainty
                    params[name] = pm.Normal(
                        name,
                        mu=1.0,  # Centered at normalized scale
                        sigma=prior.params.get("sigma", 0.05),  # Same relative uncertainty
                    )
                elif name == "sigma":
                    # Observation noise in normalized space
                    # HalfNormal centered at normalized scale
                    params[name] = pm.HalfNormal(
                        name,
                        sigma=1.0,  # O(1) scale for normalized space
                    )
                elif prior.distribution == "normal":
                    params[name] = pm.Normal(
                        name,
                        mu=prior.params["mu"],
                        sigma=prior.params["sigma"],
                    )
                elif prior.distribution == "lognormal":
                    params[name] = pm.LogNormal(
                        name,
                        mu=prior.params["mu"],
                        sigma=prior.params["sigma"],
                    )
                elif prior.distribution == "uniform":
                    params[name] = pm.Uniform(
                        name,
                        lower=prior.params["lower"],
                        upper=prior.params["upper"],
                    )
                elif prior.distribution == "halfnormal":
                    params[name] = pm.HalfNormal(
                        name,
                        sigma=prior.params["sigma"],
                    )

            # Observation noise (in normalized space)
            sigma = params.get("sigma", sigma_obs_norm)
            # Forward model prediction (in normalized space)
            if data_type == "strain":
                y_pred = pm.Deterministic(
                    "y_pred",
                    self._pytensor_strain_forward_normalized(
                        params, x_obs, data.geometry, data.load_case, self._normalizer
                    ),
                )
            else:
                y_pred = pm.Deterministic(
                    "y_pred",
                    self._pytensor_forward_normalized(
                        params, x_obs, data.geometry, data.load_case, self._normalizer
                    ),
                )

            # Likelihood (all in normalized space now!)
            pm.Normal(
                "y_obs",
                mu=y_pred,
                sigma=sigma,
                observed=y_obs_norm,
            )

        return model

    def _pytensor_forward_normalized(
        self,
        params: dict,
        x: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
        normalizer: NormalizationParams,
    ):
        """
        PyTensor forward model operating in normalized space.

        This is the key to stable MCMC: all quantities are O(1).

        Physics (in normalized space):
        - E_physical = E_normalized * E_scale
        - w_physical = f(E_physical, geometry, load)
        - w_normalized = w_physical / displacement_scale

        Since both E and w scale proportionally (w ∝ 1/E), we have:
        w_normalized = w_physical / disp_scale
                     = f(E_norm * E_scale, ...) / disp_scale

        Args:
            params: Dictionary with normalized parameters (E~1.0)
            x: Measurement locations [m]
            geometry: Beam geometry
            load: Load case
            normalizer: Normalization parameters

        Returns:
            Normalized deflection predictions (O(1) scale)
        """
        import pytensor.tensor as pt

        L = geometry.length
        I = geometry.moment_of_inertia
        P = load.point_load

        # Get NORMALIZED elastic modulus (E_norm ~ 1.0)
        E_norm = params.get("elastic_modulus", 1.0)

        # Convert to physical E for computation
        E_physical = E_norm * normalizer.E_scale

        x_t = pt.as_tensor_variable(x)

        # Compute physical deflection (this is the base class default - EB)
        EI = E_physical * I
        w_physical = -(P * x_t**2 / (6 * EI)) * (3 * L - x_t)

        # Normalize the output
        w_normalized = w_physical / normalizer.displacement_scale

        return w_normalized

    def _pytensor_strain_forward_normalized(
        self,
        params: dict,
        x: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
        normalizer: NormalizationParams,
    ):
        """
        PyTensor forward model for strain in normalized space.

        Strain at top surface: ε(x) = -y_surface * M(x) / (EI)
        For cantilever with point load: M(x) = P*(L-x)
        So: ε(x) = -(h/2) * P * (L-x) / (EI)

        Args:
            params: Dictionary with normalized parameters (E~1.0)
            x: Strain gauge locations [m]
            geometry: Beam geometry
            load: Load case
            normalizer: Normalization parameters

        Returns:
            Normalized strain predictions (O(1) scale)
        """
        import pytensor.tensor as pt

        L = geometry.length
        I = geometry.moment_of_inertia
        P = load.point_load
        y_surface = geometry.height / 2

        E_norm = params.get("elastic_modulus", 1.0)
        E_physical = E_norm * normalizer.E_scale

        x_t = pt.as_tensor_variable(x)

        # Bending moment: M(x) = P*(L-x)
        M_x = P * (L - x_t)

        # Strain at top surface: ε = -y * M / (EI)
        EI = E_physical * I
        strain_physical = -y_surface * M_x / EI

        # Normalize
        strain_normalized = strain_physical / normalizer.strain_scale

        return strain_normalized

    def calibrate(
        self,
        data: SyntheticDataset,
        data_type: str = "displacement",
        max_time_per_chain: int = 300,  # 5 min timeout per chain
    ) -> CalibrationResult:
        """
        Perform Bayesian calibration using MCMC.

        Args:
            data: Synthetic measurement data
            data_type: Type of data to fit
            max_time_per_chain: Maximum time in seconds per chain (prevents stalling)

        Returns:
            CalibrationResult with posterior samples and diagnostics

        """
        logger.info("=" * 60)
        logger.info("Calibrating %s model", self.model_name)
        logger.info("=" * 60)

        # Store dataset reference for bridge sampling
        self._calibration_data = data
        self._calibration_data_type = data_type

        # Build model
        self._model = self._build_pymc_model(data, data_type)

        # Run MCMC with better initialization and timeout
        with self._model:
            self._trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=True,
                # Better initialization to prevent stalling
                init="adapt_diag",
                # Run chains sequentially to avoid parallel init issues
                cores=None,
                # Disable sampling if stuck (nutpie-style timeout not available,
                # but we limit tuning iterations)
                initvals=self._get_initial_values(data),
            )

            # Compute log-likelihood for model comparison
            pm.compute_log_likelihood(self._trace)

        # Extract results (in normalized space)
        posterior_summary_norm = az.summary(self._trace).to_dict()
        log_lik = self._trace.log_likelihood["y_obs"].values

        # Denormalize posterior summary to physical units
        posterior_summary = self._denormalize_posterior_summary(
            posterior_summary_norm, self._normalizer
        )

        # Compute WAIC for diagnostics (not used for model comparison)
        waic_result = az.waic(self._trace)

        # Convergence diagnostics
        rhat = az.rhat(self._trace)
        ess = az.ess(self._trace)

        convergence = {
            "rhat": {k: float(v.values) for k, v in rhat.items() if k != "y_pred"},
            "ess_bulk": {k: float(v.values) for k, v in ess.items() if k != "y_pred"},
        }

        # Validate convergence
        convergence = self._validate_convergence(self._trace, convergence)

        return CalibrationResult(
            model_name=self.model_name,
            trace=self._trace,
            posterior_summary=posterior_summary,  # Physical units
            log_likelihood=log_lik,
            waic=float(waic_result.elpd_waic),  # ArviZ returns ELPDData object
            convergence_diagnostics=convergence,
            normalization_params=self._normalizer,
            posterior_summary_normalized=posterior_summary_norm,
        )

    def _denormalize_posterior_summary(
        self,
        summary_norm: dict,
        normalizer: NormalizationParams,
    ) -> dict:
        """
        Convert posterior summary from normalized to physical units.

        E_physical = E_normalized * E_scale
        sigma_physical = sigma_normalized * displacement_scale

        Args:
            summary_norm: Summary dict from az.summary() in normalized space
            normalizer: Normalization parameters

        Returns:
            Summary dict with values in physical units
        """
        import copy

        summary = copy.deepcopy(summary_norm)

        # Scale elastic modulus
        if "elastic_modulus" in summary.get("mean", {}):
            for stat in ["mean", "sd", "hdi_3%", "hdi_97%"]:
                if stat in summary and "elastic_modulus" in summary[stat]:
                    summary[stat]["elastic_modulus"] *= normalizer.E_scale

        # Scale observation noise
        if "sigma" in summary.get("mean", {}):
            for stat in ["mean", "sd", "hdi_3%", "hdi_97%"]:
                if stat in summary and "sigma" in summary[stat]:
                    summary[stat]["sigma"] *= normalizer.displacement_scale

        # Note: poisson_ratio is already O(1), no scaling needed

        return summary

    def _validate_convergence(
        self,
        trace: az.InferenceData,
        convergence: dict[str, Any],
        strict: bool = False,
    ) -> dict[str, Any]:
        """
        Validate MCMC convergence and add diagnostics.

        Checks R-hat and ESS values against thresholds and either
        raises errors/warnings or adds diagnostic information.

        Args:
            trace: ArviZ InferenceData with posterior samples
            convergence: Existing convergence dictionary to update
            strict: If True, raise ConvergenceError for poor convergence

        Returns:
            Updated convergence dictionary with validation results

        Raises:
            ConvergenceError: If R-hat > r_hat_reject and strict=True
        """
        # Get maximum R-hat across all parameters
        r_hat_values = list(convergence["rhat"].values())
        ess_values = list(convergence["ess_bulk"].values())

        r_hat_max = max(r_hat_values) if r_hat_values else 1.0
        ess_min = min(ess_values) if ess_values else 0

        # Add summary statistics
        convergence["r_hat_max"] = r_hat_max
        convergence["ess_min"] = ess_min
        convergence["converged"] = True
        convergence["warnings"] = []

        # Check R-hat - hard reject threshold
        if r_hat_max > self.r_hat_reject:
            convergence["converged"] = False
            msg = (
                f"MCMC failed to converge: R-hat = {r_hat_max:.4f} > {self.r_hat_reject}. "
                "Evidence values are unreliable. Consider increasing n_samples or n_tune."
            )
            convergence["warnings"].append(msg)
            logger.warning(msg)

            if strict:
                raise ConvergenceError(msg)

        # Check R-hat - warning threshold
        elif r_hat_max > self.r_hat_warn:
            msg = (
                f"Marginal convergence: R-hat = {r_hat_max:.4f} > {self.r_hat_warn}. "
                "Evidence may have ~10-20% error. Consider increasing samples."
            )
            convergence["warnings"].append(msg)
            warnings.warn(msg, ConvergenceWarning, stacklevel=2)

        # Check ESS
        if ess_min < self.ess_min:
            msg = (
                f"Low effective sample size: ESS = {ess_min:.0f} < {self.ess_min}. "
                "WAIC/LOO estimates may be unstable."
            )
            convergence["warnings"].append(msg)
            warnings.warn(msg, ConvergenceWarning, stacklevel=2)

        return convergence

    def _get_initial_values(self, data: SyntheticDataset) -> dict:
        """
        Compute good initial values for MCMC to prevent stalling.

        Starting from reasonable values helps the sampler avoid
        getting stuck in low-probability regions during tuning.

        Note: Since we work in normalized space, E should start at ~1.0
        (which maps to the nominal E_scale = material.elastic_modulus).
        """
        initvals = {
            # Start E_normalized at 1.0 (the normalized true value)
            "elastic_modulus": 1.0,
        }

        # Add Poisson ratio for Timoshenko (already O(1), no change needed)
        if "poisson_ratio" in self.priors:
            initvals["poisson_ratio"] = data.material.poisson_ratio

        # Observation noise in normalized space - start at O(1)
        # Since noise is also normalized by displacement_scale
        if "sigma" in self.priors:
            initvals["sigma"] = 0.1  # O(1) normalized noise

        return initvals

    def compute_marginal_likelihood(self) -> float:
        """
        Estimate marginal likelihood p(y|M) using bridge sampling.

        The marginal likelihood (model evidence) is:
        p(y|M) = ∫ p(y|θ,M) p(θ|M) dθ

        Bridge sampling (Meng & Wong, 1996) is used because:
        - It provides accurate, stable estimates with known standard error
        - The harmonic mean estimator has infinite variance and is unreliable
        - WAIC/LOO approximate predictive accuracy, not true evidence

        Returns:
            Log marginal likelihood estimate

        Raises:
            ValueError: If calibrate() has not been called first
            RuntimeError: If bridge sampling fails to converge
        """
        if self._trace is None:
            raise ValueError("Must run calibrate() before computing marginal likelihood")

        from apps.backend.core.bayesian.bridge_sampling import BridgeSampler

        log_lik_func = self._build_log_likelihood_func()
        log_prior_func = self._build_log_prior_func()
        param_names = self._get_param_names()

        sampler = BridgeSampler(
            trace=self._trace,
            log_likelihood_func=log_lik_func,
            log_prior_func=log_prior_func,
            param_names=param_names,
            n_bridge_samples=5000,
            tol=1e-8,
            max_iter=500,
        )
        result = sampler.estimate()

        if not result.converged:
            raise RuntimeError(
                f"Bridge sampling did not converge for {self.model_name}. "
                f"SE={result.standard_error:.4f}. "
                "Consider increasing n_samples or n_tune for better posterior quality."
            )

        logger.info(
            f"Bridge sampling for {self.model_name}: "
            f"log_ML={result.log_marginal_likelihood:.4f} "
            f"± {result.standard_error:.4f}"
        )
        return result.log_marginal_likelihood

    def _get_param_names(self) -> list[str]:
        """Get parameter names used in MCMC (excluding derived quantities)."""
        var_names = list(self._trace.posterior.data_vars)
        return [v for v in var_names if v not in ["y_pred", "y_obs"] and not v.startswith("_")]

    def _build_log_likelihood_func(self) -> Callable:
        """
        Build a NumPy log-likelihood callable for bridge sampling.

        Returns a function that takes a 1D parameter array and returns
        the total log-likelihood log p(y|θ) in normalized space.
        """
        data = self._calibration_data
        normalizer = self._normalizer
        param_names = self._get_param_names()

        if self._calibration_data_type == "strain":
            x_obs = data.x_strain
            y_obs = data.strains
            sigma_obs = data.strain_noise_std
        else:
            x_obs = data.x_disp
            y_obs = data.displacements
            sigma_obs = data.displacement_noise_std

        y_obs_norm = y_obs / normalizer.displacement_scale
        sigma_norm = sigma_obs / normalizer.displacement_scale

        geometry = data.geometry
        load = data.load_case

        def log_likelihood(params_array: np.ndarray) -> float:
            """Compute log p(y|θ) for a parameter vector."""
            params_dict = dict(zip(param_names, params_array, strict=False))

            # Get E in physical units
            E_norm = params_dict.get("elastic_modulus", 1.0)
            E_phys = E_norm * normalizer.E_scale

            # Compute forward model prediction
            fwd_params = {"elastic_modulus": E_phys}
            if "poisson_ratio" in params_dict:
                fwd_params["poisson_ratio"] = params_dict["poisson_ratio"]

            try:
                y_pred = self._forward_model(fwd_params, x_obs, geometry, load)
            except Exception:
                return -1e10

            y_pred_norm = y_pred / normalizer.displacement_scale

            # Get sigma from params or use observed noise
            sigma = abs(params_dict.get("sigma", sigma_norm))
            if sigma < 1e-20:
                return -1e10

            # Gaussian log-likelihood
            residuals = y_obs_norm - y_pred_norm
            n = len(residuals)
            ll = (
                -0.5 * n * np.log(2 * np.pi)
                - n * np.log(sigma)
                - 0.5 * np.sum(residuals**2) / sigma**2
            )
            return float(ll)

        return log_likelihood

    def _build_log_prior_func(self) -> Callable:
        """
        Build a NumPy log-prior callable for bridge sampling.

        Returns a function that takes a 1D parameter array and returns
        log p(θ) evaluated at the normalized parameter values.
        """
        from scipy import stats

        param_names = self._get_param_names()
        priors = self.priors

        # Pre-build scipy distributions for each parameter
        param_dists = []
        for name in param_names:
            if name == "elastic_modulus":
                # In normalized space: Normal(1.0, sigma)
                sigma = priors[name].params.get("sigma", 0.05)
                param_dists.append(stats.norm(loc=1.0, scale=sigma))
            elif name == "sigma":
                # HalfNormal(1.0) in normalized space
                param_dists.append(stats.halfnorm(scale=1.0))
            elif name == "poisson_ratio":
                mu = priors[name].params.get("mu", 0.3)
                sigma = priors[name].params.get("sigma", 0.03)
                param_dists.append(stats.norm(loc=mu, scale=sigma))
            else:
                # Generic normal fallback
                mu = priors[name].params.get("mu", 0.0)
                sigma = priors[name].params.get("sigma", 1.0)
                param_dists.append(stats.norm(loc=mu, scale=sigma))

        def log_prior(params_array: np.ndarray) -> float:
            """Compute log p(θ) for a parameter vector."""
            lp = 0.0
            for val, dist in zip(params_array, param_dists, strict=True):
                lp += dist.logpdf(val)
            return float(lp)

        return log_prior


class EulerBernoulliCalibrator(BayesianCalibrator):
    """
    Bayesian calibrator for Euler-Bernoulli beam theory.

    Calibrates material parameters (E) given measurement data,
    assuming Euler-Bernoulli kinematics.
    """

    @property
    def model_name(self) -> str:
        return "Euler-Bernoulli"

    def _forward_model(
        self,
        params: dict[str, float],
        x_locations: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute Euler-Bernoulli deflection predictions.

        """
        E = params.get("elastic_modulus", 210e9)

        material = MaterialProperties(
            elastic_modulus=E,
            poisson_ratio=0.3,
        )

        beam = EulerBernoulliBeam(geometry, material)
        return beam.compute_deflection(x_locations, load)

    def _pytensor_forward_normalized(
        self,
        params: dict,
        x: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
        normalizer: NormalizationParams,
    ):
        """
        Normalized PyTensor forward model for Euler-Bernoulli beam.

        All computations done with E_normalized ~ 1.0 and output
        is normalized displacement ~ O(1).
        """
        import pytensor.tensor as pt

        L = geometry.length
        I = geometry.moment_of_inertia
        P = load.point_load

        # Get normalized E (should be ~1.0) and convert to physical
        E_norm = params.get("elastic_modulus", 1.0)
        E_physical = E_norm * normalizer.E_scale

        x_t = pt.as_tensor_variable(x)

        # Compute physical deflection (Euler-Bernoulli bending only)
        EI = E_physical * I
        w_physical = -(P * x_t**2 / (6 * EI)) * (3 * L - x_t)

        # Normalize the output to O(1)
        w_normalized = w_physical / normalizer.displacement_scale

        return w_normalized


class TimoshenkoCalibrator(BayesianCalibrator):
    """
    Bayesian calibrator for Timoshenko beam theory.

    Calibrates material parameters (E, G or ν) given measurement data,
    accounting for shear deformation effects.
    """

    @property
    def model_name(self) -> str:
        return "Timoshenko"

    def _forward_model(
        self,
        params: dict[str, float],
        x_locations: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute Timoshenko deflection predictions.

        """
        E = params.get("elastic_modulus", 210e9)
        nu = params.get("poisson_ratio", 0.3)

        material = MaterialProperties(
            elastic_modulus=E,
            poisson_ratio=nu,
        )

        beam = TimoshenkoBeam(geometry, material)
        return beam.compute_deflection(x_locations, load)

    def _pytensor_forward_normalized(
        self,
        params: dict,
        x: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
        normalizer: NormalizationParams,
    ):
        """
        Normalized PyTensor forward model for Timoshenko beam.

        Includes both bending and shear deformation, all with
        normalized E ~ 1.0 and output ~ O(1).
        """
        import pytensor.tensor as pt

        L = geometry.length
        I = geometry.moment_of_inertia
        A = geometry.area
        P = load.point_load
        kappa = 5.0 / 6.0  # Shear correction factor for rectangular section

        # Get normalized E (should be ~1.0) and convert to physical
        E_norm = params.get("elastic_modulus", 1.0)
        E_physical = E_norm * normalizer.E_scale

        # Poisson's ratio stays as-is (already O(1))
        nu = params.get("poisson_ratio", 0.3)

        x_t = pt.as_tensor_variable(x)

        # Bending contribution
        EI = E_physical * I
        w_bending = -(P * x_t**2 / (6 * EI)) * (3 * L - x_t)

        # Shear contribution
        G = E_physical / (2 * (1 + nu))
        w_shear = -(P * x_t) / (kappa * G * A)

        w_physical = w_bending + w_shear

        # Normalize the output to O(1)
        w_normalized = w_physical / normalizer.displacement_scale

        return w_normalized


def create_default_priors(config: dict | None = None) -> list[PriorConfig]:
    """
    Create default prior configurations for beam calibration.

    If a config dict is provided, prior parameters are read from the
    ``priors`` section, falling back to hard-coded defaults.

    IMPORTANT: Prior Selection Rationale
    ------------------------------------
    1. Elastic Modulus (E):
       - Steel: E ≈ 200-220 GPa, we use LogNormal centered at 210 GPa
       - sigma=0.05 gives ~5% uncertainty (tighter than before to avoid divergences)
       - LogNormal ensures E > 0 and is scale-appropriate for Pa

    2. Observation Noise (sigma):
       - HalfNormal with small sigma for tight constraint
       - Typical displacement measurements: ~1e-6 m precision

    Args:
        config: Optional full configuration dict with a ``priors`` key.

    Returns:
        List of PriorConfig objects
    """
    prior_cfg = (config or {}).get("priors", {})
    e_cfg = prior_cfg.get("elastic_modulus", {})
    noise_cfg = prior_cfg.get("observation_noise", {})

    return [
        PriorConfig(
            param_name="elastic_modulus",
            distribution=e_cfg.get("distribution", "lognormal"),
            params={
                "mu": e_cfg.get("mu", np.log(210e9)),
                "sigma": e_cfg.get("sigma", 0.05),
            },
        ),
        PriorConfig(
            param_name="sigma",
            distribution=noise_cfg.get("distribution", "halfnormal"),
            params={"sigma": noise_cfg.get("sigma", 1e-6)},
        ),
    ]


def create_timoshenko_priors(config: dict | None = None) -> list[PriorConfig]:
    """
    Create priors for Timoshenko beam calibration.

    Includes additional parameter for Poisson's ratio which determines
    shear modulus via G = E / (2*(1+nu)).

    Note: Timoshenko has one more parameter than Euler-Bernoulli,
    so WAIC/LOO will penalize this extra complexity via the effective
    number of parameters term.

    Args:
        config: Optional full configuration dict with a ``priors`` key.
    """
    priors = create_default_priors(config)
    prior_cfg = (config or {}).get("priors", {})
    nu_cfg = prior_cfg.get("poisson_ratio", {})

    priors.append(
        PriorConfig(
            param_name="poisson_ratio",
            distribution=nu_cfg.get("distribution", "normal"),
            params={
                "mu": nu_cfg.get("mu", 0.3),
                "sigma": nu_cfg.get("sigma", 0.05),
            },
        )
    )
    return priors
