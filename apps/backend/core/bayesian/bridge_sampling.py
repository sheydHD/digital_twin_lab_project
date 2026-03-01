"""
Bridge Sampling for Marginal Likelihood Estimation.

This module implements bridge sampling, the gold standard method for
computing marginal likelihoods (model evidence) in Bayesian inference.

Why Bridge Sampling?
-------------------
- WAIC/LOO measure predictive accuracy, not true evidence
- Harmonic mean estimator has infinite variance (unreliable)
- Laplace approximation assumes Gaussian posterior (often violated)
- Bridge sampling provides accurate, stable estimates

Theory:
------
Bridge sampling uses the identity:
    p(y|M) = E_proposal[p(y|θ)p(θ|M) / q(θ)] / E_posterior[1 / q(θ)]

where q(θ) is a proposal distribution (usually fitted to posterior).

The optimal bridge function (Meng & Wong, 1996) minimizes variance.

Reference:
- Gronau et al. (2017) "bridgesampling: An R Package for Estimating
  Normalizing Constants" Journal of Statistical Software
- Meng & Wong (1996) "Simulating Ratios of Normalizing Constants"
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import arviz as az
import numpy as np
from scipy import special

logger = logging.getLogger(__name__)


@dataclass
class BridgeSamplingResult:
    """
    Result from bridge sampling estimation.

    Attributes:
        log_marginal_likelihood: Estimated log p(y|M)
        standard_error: Estimated standard error of log ML
        n_iterations: Number of bridge iterations to converge
        converged: Whether the iteration converged
        proposal_fit: Quality metrics for proposal distribution fit
    """

    log_marginal_likelihood: float
    standard_error: float
    n_iterations: int
    converged: bool
    proposal_fit: dict | None = None

    def __repr__(self) -> str:
        return (
            f"BridgeSamplingResult(\n"
            f"  log_ML={self.log_marginal_likelihood:.4f},\n"
            f"  SE={self.standard_error:.4f},\n"
            f"  converged={self.converged},\n"
            f"  iterations={self.n_iterations}\n"
            f")"
        )


class BridgeSampler:
    """
    Bridge sampling for marginal likelihood estimation.

    This class implements the iterative bridge sampling algorithm
    for computing p(y|M) from MCMC posterior samples.

    Example:
        >>> sampler = BridgeSampler(
        ...     trace=pymc_trace, log_likelihood_func=my_log_lik, log_prior_func=my_log_prior
        ... )
        >>> result = sampler.estimate()
        >>> print(f"Log marginal likelihood: {result.log_marginal_likelihood:.2f}")
    """

    def __init__(
        self,
        trace: az.InferenceData,
        log_likelihood_func: Callable[[np.ndarray], float],
        log_prior_func: Callable[[np.ndarray], float],
        param_names: list[str] | None = None,
        n_bridge_samples: int = 10000,
        tol: float = 1e-8,
        max_iter: int = 1000,
        seed: int | None = None,
    ):
        """
        Initialize bridge sampler.

        Args:
            trace: ArviZ InferenceData with posterior samples
            log_likelihood_func: Function(params_array) -> log p(y|θ)
            log_prior_func: Function(params_array) -> log p(θ)
            param_names: Names of parameters to include (None = all)
            n_bridge_samples: Samples from proposal distribution
            tol: Convergence tolerance
            max_iter: Maximum iterations
            seed: Random seed for reproducible proposal sampling
        """
        self.trace = trace
        self.log_likelihood = log_likelihood_func
        self.log_prior = log_prior_func
        self.param_names = param_names
        self.n_bridge = n_bridge_samples
        self.tol = tol
        self.max_iter = max_iter
        self._rng = np.random.default_rng(seed)

        # Will be computed during estimation
        self._posterior_samples = None
        self._proposal_mean = None
        self._proposal_cov = None

    def estimate(self) -> BridgeSamplingResult:
        """
        Estimate log marginal likelihood using bridge sampling.

        Returns:
            BridgeSamplingResult with estimate and diagnostics
        """
        logger.info("Starting bridge sampling estimation...")

        # Extract posterior samples
        self._posterior_samples = self._extract_posterior_samples()
        n_posterior = len(self._posterior_samples)
        n_params = self._posterior_samples.shape[1]

        logger.debug(f"Extracted {n_posterior} posterior samples, {n_params} parameters")

        # Fit proposal distribution (multivariate normal)
        self._proposal_mean, self._proposal_cov = self._fit_proposal()

        # Check proposal fit quality
        proposal_fit = self._assess_proposal_fit()

        # Generate samples from proposal
        try:
            proposal_samples = self._rng.multivariate_normal(
                self._proposal_mean,
                self._proposal_cov,
                size=self.n_bridge,
            )
        except np.linalg.LinAlgError:
            # Covariance not positive definite, add regularization
            logger.warning("Proposal covariance not positive definite, adding regularization")
            reg_cov = self._proposal_cov + 1e-6 * np.eye(n_params)
            proposal_samples = self._rng.multivariate_normal(
                self._proposal_mean, reg_cov, size=self.n_bridge
            )

        # Compute log densities for posterior samples
        log_post_likelihood = self._batch_log_likelihood(self._posterior_samples)
        log_post_prior = self._batch_log_prior(self._posterior_samples)
        log_post_proposal = self._log_mvn_pdf(
            self._posterior_samples, self._proposal_mean, self._proposal_cov
        )

        # Compute log densities for proposal samples
        log_prop_likelihood = self._batch_log_likelihood(proposal_samples)
        log_prop_prior = self._batch_log_prior(proposal_samples)
        log_prop_proposal = self._log_mvn_pdf(
            proposal_samples, self._proposal_mean, self._proposal_cov
        )

        # Iterative bridge sampling
        log_ml, n_iter, converged = self._iterate_bridge(
            log_post_likelihood,
            log_post_prior,
            log_post_proposal,
            log_prop_likelihood,
            log_prop_prior,
            log_prop_proposal,
            n_posterior,
            self.n_bridge,
        )

        # Estimate standard error
        se = self._estimate_standard_error(
            log_post_likelihood,
            log_post_prior,
            log_post_proposal,
            log_prop_likelihood,
            log_prop_prior,
            log_prop_proposal,
            log_ml,
            n_posterior,
            self.n_bridge,
        )

        logger.info(
            f"Bridge sampling complete: log_ML={log_ml:.4f}, SE={se:.4f}, converged={converged}"
        )

        return BridgeSamplingResult(
            log_marginal_likelihood=log_ml,
            standard_error=se,
            n_iterations=n_iter,
            converged=converged,
            proposal_fit=proposal_fit,
        )

    def _extract_posterior_samples(self) -> np.ndarray:
        """Extract and flatten posterior samples from ArviZ trace."""
        samples = []

        # Determine which variables to include
        if self.param_names is None:
            var_names = list(self.trace.posterior.data_vars)
            # Exclude deterministic/derived quantities
            var_names = [
                v for v in var_names if v not in ["y_pred", "y_obs"] and not v.startswith("_")
            ]
        else:
            var_names = self.param_names

        for var in var_names:
            if var in self.trace.posterior:
                var_samples = self.trace.posterior[var].values
                # Flatten chains and draws: (chains, draws) -> (n_samples,)
                flat = var_samples.reshape(-1)
                samples.append(flat)

        if not samples:
            raise ValueError("No valid posterior samples found")

        return np.column_stack(samples)

    def _fit_proposal(self) -> tuple[np.ndarray, np.ndarray]:
        """Fit multivariate normal proposal to posterior samples."""
        mean = np.mean(self._posterior_samples, axis=0)
        cov = np.cov(self._posterior_samples, rowvar=False)

        # Handle 1D case
        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Ensure positive definite by adding small regularization
        min_eig = np.min(np.linalg.eigvalsh(cov))
        if min_eig < 1e-10:
            cov = cov + (1e-6 - min_eig) * np.eye(len(mean))

        return mean, cov

    def _assess_proposal_fit(self) -> dict:
        """Assess how well the proposal fits the posterior."""
        # Compute Mahalanobis distances
        diff = self._posterior_samples - self._proposal_mean
        try:
            cov_inv = np.linalg.inv(self._proposal_cov)
            mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        except np.linalg.LinAlgError:
            mahal = np.zeros(len(self._posterior_samples))

        # For well-fitted MVN, Mahalanobis should be chi-distributed
        n_params = self._posterior_samples.shape[1]
        expected_mahal = np.sqrt(n_params)

        return {
            "mean_mahalanobis": float(np.mean(mahal)),
            "expected_mahalanobis": expected_mahal,
            "max_mahalanobis": float(np.max(mahal)),
            "proposal_mean": self._proposal_mean.tolist(),
            "proposal_cov_diag": np.diag(self._proposal_cov).tolist(),
        }

    def _log_mvn_pdf(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Compute log PDF of multivariate normal."""
        k = len(mean)

        # Add small regularization for numerical stability
        cov_reg = cov + 1e-10 * np.eye(k)

        try:
            cov_inv = np.linalg.inv(cov_reg)
            sign, log_det = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                log_det = k * np.log(1e-10)  # Fallback
        except np.linalg.LinAlgError:
            # Fallback for singular covariance
            return np.full(len(x), -1e10)

        diff = x - mean
        mahal = np.sum(diff @ cov_inv * diff, axis=1)
        log_pdf = -0.5 * (k * np.log(2 * np.pi) + log_det + mahal)

        return log_pdf

    def _batch_log_likelihood(self, samples: np.ndarray) -> np.ndarray:
        """Compute log-likelihood for batch of samples."""
        result = np.zeros(len(samples))
        for i, s in enumerate(samples):
            try:
                result[i] = self.log_likelihood(s)
            except Exception as e:
                logger.warning(f"Log-likelihood failed for sample {i}: {e}")
                result[i] = -1e10  # Very low likelihood
        return result

    def _batch_log_prior(self, samples: np.ndarray) -> np.ndarray:
        """Compute log-prior for batch of samples."""
        result = np.zeros(len(samples))
        for i, s in enumerate(samples):
            try:
                result[i] = self.log_prior(s)
            except Exception as e:
                logger.warning(f"Log-prior failed for sample {i}: {e}")
                result[i] = -1e10  # Very low prior
        return result

    def _iterate_bridge(
        self,
        log_post_lik: np.ndarray,
        log_post_prior: np.ndarray,
        log_post_prop: np.ndarray,
        log_prop_lik: np.ndarray,
        log_prop_prior: np.ndarray,
        log_prop_prop: np.ndarray,
        n1: int,
        n2: int,
    ) -> tuple[float, int, bool]:
        """
        Iterative bridge sampling algorithm.

        Uses the optimal bridge function from Meng & Wong (1996).

        Returns:
            Tuple of (log_marginal_likelihood, n_iterations, converged)
        """
        # Unnormalized log posterior = log_lik + log_prior
        log_post_unnorm = log_post_lik + log_post_prior
        log_prop_unnorm = log_prop_lik + log_prop_prior

        # Sample size ratio
        s1 = n1 / (n1 + n2)
        s2 = n2 / (n1 + n2)

        # Initial estimate (use median of log-likelihoods as starting point)
        log_ml = float(np.median(log_post_unnorm))

        converged = False
        n_iters = 0
        for _it in range(self.max_iter):
            n_iters += 1
            log_ml_old = log_ml

            # Numerator: E_proposal[l(θ) / (s1*l(θ) + s2*g(θ)*exp(log_ml))]
            # where l(θ) = p(y|θ)p(θ) and g(θ) = q(θ)
            log_num_terms = log_prop_unnorm - np.logaddexp(
                np.log(s1) + log_prop_unnorm,
                np.log(s2) + log_prop_prop + log_ml,
            )
            log_numerator = special.logsumexp(log_num_terms) - np.log(n2)

            # Denominator: E_posterior[g(θ) / (s1*l(θ) + s2*g(θ)*exp(log_ml))]
            log_den_terms = log_post_prop - np.logaddexp(
                np.log(s1) + log_post_unnorm,
                np.log(s2) + log_post_prop + log_ml,
            )
            log_denominator = special.logsumexp(log_den_terms) - np.log(n1)

            log_ml = log_numerator - log_denominator

            # Check convergence
            if abs(log_ml - log_ml_old) < self.tol:
                converged = True
                break

        if not converged:
            logger.warning(
                f"Bridge sampling did not converge after {self.max_iter} iterations. "
                f"Final change: {abs(log_ml - log_ml_old):.2e}"
            )

        return log_ml, n_iters, converged

    def _estimate_standard_error(
        self,
        log_post_lik: np.ndarray,
        log_post_prior: np.ndarray,
        log_post_prop: np.ndarray,
        log_prop_lik: np.ndarray,
        log_prop_prior: np.ndarray,
        log_prop_prop: np.ndarray,
        log_ml: float,
        n1: int,
        n2: int,
    ) -> float:
        """
        Estimate standard error of bridge sampling estimate.

        Uses the delta method approximation from Gronau et al. (2017).
        """
        log_post_unnorm = log_post_lik + log_post_prior
        log_prop_unnorm = log_prop_lik + log_prop_prior

        s1 = n1 / (n1 + n2)
        s2 = n2 / (n1 + n2)

        # Compute h1 and h2 terms
        h1 = np.exp(
            log_post_prop
            - np.logaddexp(
                np.log(s1) + log_post_unnorm,
                np.log(s2) + log_post_prop + log_ml,
            )
        )

        h2 = np.exp(
            log_prop_unnorm
            - np.logaddexp(
                np.log(s1) + log_prop_unnorm,
                np.log(s2) + log_prop_prop + log_ml,
            )
        )

        # Variance components
        var_h1 = np.var(h1) / n1
        var_h2 = np.var(h2) / n2

        # Total variance (delta method)
        mean_h1 = np.mean(h1)
        mean_h2 = np.mean(h2)

        if mean_h1 > 1e-10 and mean_h2 > 1e-10:
            # Delta method for ratio
            rel_var = var_h1 / mean_h1**2 + var_h2 / mean_h2**2
            se = np.sqrt(rel_var)
        else:
            # Fallback
            se = 1.0

        return float(se)


def compute_bayes_factor_bridge(
    trace1: az.InferenceData,
    trace2: az.InferenceData,
    log_lik_func1: Callable,
    log_lik_func2: Callable,
    log_prior_func1: Callable,
    log_prior_func2: Callable,
    param_names1: list[str] | None = None,
    param_names2: list[str] | None = None,
    n_bridge_samples: int = 10000,
    seed: int | None = None,
) -> tuple[float, float, BridgeSamplingResult, BridgeSamplingResult]:
    """
    Compute Bayes factor using bridge sampling for both models.

    This is the recommended way to compute true Bayes factors,
    as opposed to WAIC-based approximations.

    Args:
        trace1: Posterior trace for model 1 (e.g., Euler-Bernoulli)
        trace2: Posterior trace for model 2 (e.g., Timoshenko)
        log_lik_func1: Log-likelihood function for model 1
        log_lik_func2: Log-likelihood function for model 2
        log_prior_func1: Log-prior function for model 1
        log_prior_func2: Log-prior function for model 2
        param_names1: Parameter names for model 1
        param_names2: Parameter names for model 2
        n_bridge_samples: Number of bridge samples

    Returns:
        Tuple of (log_bayes_factor, combined_se, result1, result2)
        Positive log_BF favors model 1.
    """
    logger.info("Computing Bayes factor via bridge sampling...")

    sampler1 = BridgeSampler(
        trace1,
        log_lik_func1,
        log_prior_func1,
        param_names1,
        n_bridge_samples,
        seed=seed,
    )
    sampler2 = BridgeSampler(
        trace2,
        log_lik_func2,
        log_prior_func2,
        param_names2,
        n_bridge_samples,
        seed=seed,
    )

    result1 = sampler1.estimate()
    result2 = sampler2.estimate()

    log_bf = result1.log_marginal_likelihood - result2.log_marginal_likelihood
    combined_se = np.sqrt(result1.standard_error**2 + result2.standard_error**2)

    logger.info(
        f"Bayes factor: log_BF = {log_bf:.4f} ± {combined_se:.4f} (BF = {np.exp(log_bf):.2f})"
    )

    return log_bf, combined_se, result1, result2
