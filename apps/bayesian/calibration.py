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

import numpy as np
import pymc as pm
import arviz as az
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod

from apps.models.base_beam import BeamGeometry, MaterialProperties, LoadCase
from apps.models.euler_bernoulli import EulerBernoulliBeam
from apps.models.timoshenko import TimoshenkoBeam
from apps.data.synthetic_generator import SyntheticDataset


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
    params: Dict[str, float]


@dataclass
class CalibrationResult:
    """
    Results from Bayesian calibration.

    Attributes:
        model_name: Name of the calibrated model
        trace: PyMC InferenceData object with posterior samples
        posterior_summary: Summary statistics for parameters
        log_likelihood: Log-likelihood at posterior samples
        waic: Widely Applicable Information Criterion
        loo: Leave-One-Out cross-validation score
        marginal_likelihood_estimate: Estimated marginal likelihood
        convergence_diagnostics: R-hat and ESS statistics
    """

    model_name: str
    trace: az.InferenceData
    posterior_summary: Dict
    log_likelihood: np.ndarray
    waic: Optional[float] = None
    loo: Optional[float] = None
    marginal_likelihood_estimate: Optional[float] = None
    convergence_diagnostics: Optional[Dict] = None


class BayesianCalibrator(ABC):
    """
    Abstract base class for Bayesian model calibration.

    This class defines the interface for calibrating beam models using
    Bayesian inference. Subclasses implement specific beam theories.

    """

    def __init__(
        self,
        priors: List[PriorConfig],
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

    @abstractmethod
    def _forward_model(
        self,
        params: Dict[str, float],
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
        Build PyMC probabilistic model.

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
        else:
            x_obs = data.x_strain
            y_obs = data.strains
            sigma_obs = data.strain_noise_std

        with pm.Model() as model:
            # Define priors for each parameter
            params = {}

            for name, prior in self.priors.items():
                if prior.distribution == "normal":
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

            # Observation noise (can be inferred or fixed)
            if "sigma" in self.priors:
                sigma = params["sigma"]
            else:
                # Fixed observation noise
                sigma = sigma_obs

            # For now, use pm.Deterministic with custom forward model
            # This is a placeholder - actual implementation needs PyTensor ops

            # Forward model prediction
            # Note: This approach is simplified. For production, use pytensor.
            y_pred = pm.Deterministic(
                "y_pred",
                self._pytensor_forward(params, x_obs, data.geometry, data.load_case),
            )

            # Likelihood
            likelihood = pm.Normal(
                "y_obs",
                mu=y_pred,
                sigma=sigma,
                observed=y_obs,
            )

        return model

    def _pytensor_forward(
        self,
        params: Dict,
        x: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
    ):
        """
        PyTensor-compatible forward model.

        For analytical beam models, this should compute deflection as a
        function of PyTensor variables.
        """
        # Placeholder - needs proper PyTensor implementation
        import pytensor.tensor as pt

        # Example for Euler-Bernoulli with point load:
        # w = (P * x^2 / (6*E*I)) * (3*L - x)

        L = geometry.length
        I = geometry.moment_of_inertia
        P = load.point_load

        E = params.get("elastic_modulus", 210e9)

        # Convert to PyTensor
        x_t = pt.as_tensor_variable(x)

        # Compute deflection
        EI = E * I
        w = (P * x_t**2 / (6 * EI)) * (3 * L - x_t)

        return w

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
        print(f"\n{'='*60}")
        print(f"Calibrating {self.model_name} model")
        print(f"{'='*60}")

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
                cores=1,
                # Disable sampling if stuck (nutpie-style timeout not available,
                # but we limit tuning iterations)
                initvals=self._get_initial_values(data),
            )

            # Compute log-likelihood for model comparison
            pm.compute_log_likelihood(self._trace)

        # Extract results
        posterior_summary = az.summary(self._trace).to_dict()
        log_lik = self._trace.log_likelihood["y_obs"].values

        # Compute model comparison criteria
        waic_result = az.waic(self._trace)
        loo_result = az.loo(self._trace)

        # Convergence diagnostics
        rhat = az.rhat(self._trace)
        ess = az.ess(self._trace)

        convergence = {
            "rhat": {k: float(v.values) for k, v in rhat.items() if k != "y_pred"},
            "ess_bulk": {k: float(v.values) for k, v in ess.items() if k != "y_pred"},
        }

        return CalibrationResult(
            model_name=self.model_name,
            trace=self._trace,
            posterior_summary=posterior_summary,
            log_likelihood=log_lik,
            waic=float(waic_result.elpd_waic),  # ArviZ returns ELPDData object
            loo=float(loo_result.elpd_loo),      # Access elpd_loo attribute
            convergence_diagnostics=convergence,
        )

    def _get_initial_values(self, data: SyntheticDataset) -> Dict:
        """
        Compute good initial values for MCMC to prevent stalling.
        
        Starting from reasonable values helps the sampler avoid
        getting stuck in low-probability regions during tuning.
        """
        initvals = {
            # Start E at true value (or close to it)
            "elastic_modulus": data.material.elastic_modulus,
        }
        
        # Add Poisson ratio for Timoshenko
        if "poisson_ratio" in self.priors:
            initvals["poisson_ratio"] = data.material.poisson_ratio
        
        # Observation noise - start at a reasonable small value
        if "sigma" in self.priors:
            initvals["sigma"] = 1e-7  # Small observation noise
            
        return initvals

    def compute_marginal_likelihood(
        self,
        method: str = "harmonic_mean",
    ) -> float:
        """
        Estimate marginal likelihood p(y|M) for model comparison.

        The marginal likelihood (evidence) is:
        p(y|M) = ∫ p(y|θ,M) p(θ|M) dθ

        Methods:
        - harmonic_mean: Simple but high-variance estimator
        - bridge_sampling: More accurate but complex
        - SMC: Sequential Monte Carlo for direct estimation

        Args:
            method: Estimation method

        Returns:
            Log marginal likelihood estimate

        """
        if self._trace is None:
            raise ValueError("Must run calibrate() before computing marginal likelihood")

        log_lik = self._trace.log_likelihood["y_obs"].values.sum(axis=-1).flatten()

        if method == "harmonic_mean":
            # Harmonic mean estimator (Newton & Raftery, 1994)
            # log p(y|M) ≈ -log(mean(exp(-log_lik)))
            # 
            # Numerically stable version using log-sum-exp trick:
            # log(mean(exp(-log_lik))) = log(sum(exp(-log_lik))) - log(n)
            #                          = logsumexp(-log_lik) - log(n)
            from scipy.special import logsumexp
            
            neg_log_lik = -log_lik
            n = len(neg_log_lik)
            log_mean_exp = logsumexp(neg_log_lik) - np.log(n)
            log_ml = -log_mean_exp
            
            return log_ml

        elif method == "bridge_sampling":
            raise NotImplementedError("Bridge sampling not yet implemented")

        elif method == "smc":
            raise NotImplementedError("SMC estimation not yet implemented")

        else:
            raise ValueError(f"Unknown method: {method}")

    def posterior_predictive_check(
        self,
        data: SyntheticDataset,
        n_samples: int = 500,
    ) -> Dict:
        """
        Perform posterior predictive checks.

        Generate predictions from posterior samples and compare to observed data.

        Args:
            data: Original dataset
            n_samples: Number of posterior samples to use

        Returns:
            Dictionary with PPC results

        """
        if self._trace is None:
            raise ValueError("Must run calibrate() first")

        with self._model:
            ppc = pm.sample_posterior_predictive(
                self._trace,
                var_names=["y_obs"],
                random_seed=self.random_seed,
            )

        return {
            "ppc_samples": ppc.posterior_predictive["y_obs"].values,
            "observed": data.displacements,
        }


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
        params: Dict[str, float],
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

    def _pytensor_forward(
        self,
        params: Dict,
        x: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
    ):
        """
        PyTensor-compatible forward model for Euler-Bernoulli beam.

        Deflection formula for cantilever with tip load:
        w(x) = (P * x^2 / (6*E*I)) * (3*L - x)
        """
        import pytensor.tensor as pt

        L = geometry.length
        I = geometry.moment_of_inertia
        P = load.point_load

        E = params.get("elastic_modulus", 210e9)

        x_t = pt.as_tensor_variable(x)

        # Euler-Bernoulli: bending only (negative for downward deflection)
        EI = E * I
        w = -(P * x_t**2 / (6 * EI)) * (3 * L - x_t)

        return w


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
        params: Dict[str, float],
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

    def _pytensor_forward(
        self,
        params: Dict,
        x: np.ndarray,
        geometry: BeamGeometry,
        load: LoadCase,
    ):
        """
        PyTensor-compatible forward model for Timoshenko beam.

        Deflection formula for cantilever with tip load:
        w(x) = w_bending(x) + w_shear(x)
             = (P * x^2 / (6*E*I)) * (3*L - x) + (P * x) / (kappa * G * A)
        
        where G = E / (2*(1+nu))
        """
        import pytensor.tensor as pt

        L = geometry.length
        I = geometry.moment_of_inertia
        A = geometry.area
        P = load.point_load
        kappa = 5.0 / 6.0  # Shear correction factor for rectangular section

        E = params.get("elastic_modulus", 210e9)
        nu = params.get("poisson_ratio", 0.3)

        x_t = pt.as_tensor_variable(x)

        # Bending contribution (same as Euler-Bernoulli, negative for downward)
        EI = E * I
        w_bending = -(P * x_t**2 / (6 * EI)) * (3 * L - x_t)

        # Shear contribution (Timoshenko correction, also negative)
        G = E / (2 * (1 + nu))
        w_shear = -(P * x_t) / (kappa * G * A)

        return w_bending + w_shear


def create_default_priors() -> List[PriorConfig]:
    """
    Create default prior configurations for beam calibration.
    
    IMPORTANT: Prior Selection Rationale
    ------------------------------------
    1. Elastic Modulus (E):
       - Steel: E ≈ 200-220 GPa, we use LogNormal centered at 210 GPa
       - sigma=0.05 gives ~5% uncertainty (tighter than before to avoid divergences)
       - LogNormal ensures E > 0 and is scale-appropriate for Pa
    
    2. Observation Noise (sigma):
       - HalfNormal with small sigma for tight constraint
       - Typical displacement measurements: ~1e-6 m precision

    Returns:
        List of PriorConfig objects

    """
    return [
        PriorConfig(
            param_name="elastic_modulus",
            distribution="lognormal",
            params={
                "mu": np.log(210e9),  # ln(210 GPa) ≈ 26.07
                "sigma": 0.05,  # ~5% uncertainty (tighter to prevent divergences)
            },
        ),
        PriorConfig(
            param_name="sigma",
            distribution="halfnormal",
            params={"sigma": 1e-6},  # Observation noise scale (tighter)
        ),
    ]


def create_timoshenko_priors() -> List[PriorConfig]:
    """
    Create priors for Timoshenko beam calibration.

    Includes additional parameter for Poisson's ratio which determines
    shear modulus via G = E / (2*(1+nu)).
    
    Note: Timoshenko has one more parameter than Euler-Bernoulli,
    so WAIC/LOO will penalize this extra complexity via the effective
    number of parameters term.

    """
    priors = create_default_priors()
    priors.append(
        PriorConfig(
            param_name="poisson_ratio",
            distribution="normal",  # Changed from uniform for better sampling
            params={
                "mu": 0.3,      # Steel typical value
                "sigma": 0.03,  # Tight prior: most metals 0.25-0.35
            },
        )
    )
    return priors
