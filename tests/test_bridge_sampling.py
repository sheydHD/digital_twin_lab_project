"""
Unit tests for bridge sampling module.
"""

from __future__ import annotations

import numpy as np
import pytest

from apps.backend.core.bayesian.bridge_sampling import BridgeSampler, BridgeSamplingResult


class TestBridgeSamplingResult:
    def test_repr(self) -> None:
        result = BridgeSamplingResult(
            log_marginal_likelihood=-100.0,
            standard_error=0.5,
            n_iterations=42,
            converged=True,
        )
        text = repr(result)
        assert "log_ML=-100.0000" in text
        assert "converged=True" in text


class TestBridgeSampler:
    """Lightweight tests that don't require a real MCMC trace."""

    def test_seed_reproducibility(self) -> None:
        """Two samplers with the same seed should produce identical RNG states."""
        import arviz as az

        # Create a minimal mock trace
        rng = np.random.default_rng(0)
        posterior_data = {"param": rng.normal(size=(2, 100))}
        trace = az.from_dict(posterior=posterior_data)

        log_lik = lambda p: -0.5 * float(np.sum(p**2))  # noqa: E731
        log_prior = lambda p: -0.5 * float(np.sum(p**2))  # noqa: E731

        s1 = BridgeSampler(trace, log_lik, log_prior, seed=42, n_bridge_samples=50, max_iter=5)
        s2 = BridgeSampler(trace, log_lik, log_prior, seed=42, n_bridge_samples=50, max_iter=5)

        # Both RNGs should generate the same sequence
        a = s1._rng.random(10)
        b = s2._rng.random(10)
        np.testing.assert_array_equal(a, b)

    def test_extract_posterior_samples(self) -> None:
        import arviz as az

        rng = np.random.default_rng(7)
        data = {"x": rng.normal(5.0, 1.0, size=(2, 200))}
        trace = az.from_dict(posterior=data)

        sampler = BridgeSampler(
            trace,
            log_likelihood_func=lambda p: 0.0,
            log_prior_func=lambda p: 0.0,
            seed=0,
        )
        samples = sampler._extract_posterior_samples()
        assert samples.shape == (400, 1)
        assert np.mean(samples) == pytest.approx(5.0, abs=0.5)
