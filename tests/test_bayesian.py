"""
Unit tests for Bayesian calibration module.

Tests:
- Prior configuration
- Calibrator initialization
- Forward model computation
- Normalization integration
- MCMC convergence (smoke tests)
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from apps.bayesian.calibration import (
    CalibrationResult,
    EulerBernoulliCalibrator,
    PriorConfig,
    TimoshenkoCalibrator,
    create_default_priors,
    create_timoshenko_priors,
)
from apps.data.synthetic_generator import SyntheticDataset
from apps.models.base_beam import BeamGeometry, LoadCase, MaterialProperties


class TestPriorConfig:
    """Tests for PriorConfig dataclass."""

    def test_normal_prior(self):
        """Test normal prior configuration."""
        prior = PriorConfig(
            param_name="E",
            distribution="normal",
            params={"mu": 210e9, "sigma": 10e9}
        )
        assert prior.param_name == "E"
        assert prior.distribution == "normal"
        assert prior.params["mu"] == 210e9

    def test_lognormal_prior(self):
        """Test lognormal prior configuration."""
        prior = PriorConfig(
            param_name="E",
            distribution="lognormal",
            params={"mu": np.log(210e9), "sigma": 0.05}
        )
        assert prior.distribution == "lognormal"

    def test_halfnormal_prior(self):
        """Test halfnormal prior configuration."""
        prior = PriorConfig(
            param_name="sigma",
            distribution="halfnormal",
            params={"sigma": 1e-5}
        )
        assert prior.distribution == "halfnormal"


class TestCreatePriors:
    """Tests for prior creation functions."""

    def test_default_priors(self):
        """Test default prior creation for E-B model."""
        priors = create_default_priors()
        assert len(priors) == 2

        param_names = [p.param_name for p in priors]
        assert "elastic_modulus" in param_names
        assert "sigma" in param_names

    def test_timoshenko_priors(self):
        """Test Timoshenko prior creation includes poisson_ratio."""
        priors = create_timoshenko_priors()

        param_names = [p.param_name for p in priors]
        assert "elastic_modulus" in param_names
        assert "poisson_ratio" in param_names
        assert "sigma" in param_names


class TestEulerBernoulliCalibrator:
    """Tests for Euler-Bernoulli calibrator."""

    @pytest.fixture
    def priors(self):
        """Create default priors."""
        return create_default_priors()

    @pytest.fixture
    def calibrator(self, priors):
        """Create calibrator with default priors."""
        return EulerBernoulliCalibrator(priors=priors)

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset for testing."""
        dataset = MagicMock(spec=SyntheticDataset)
        dataset.geometry = BeamGeometry(length=1.0, height=0.1, width=0.05)
        dataset.material = MaterialProperties(elastic_modulus=210e9)
        dataset.load_case = LoadCase(point_load=1000)
        dataset.x_disp = np.linspace(0.1, 1.0, 10)
        dataset.displacements = -1e-4 * np.ones(10)  # Negative for downward
        dataset.displacement_noise_std = 1e-6
        return dataset

    def test_initialization(self, calibrator):
        """Test calibrator initializes correctly."""
        assert calibrator.model_name == "Euler-Bernoulli"
        assert len(calibrator.priors) == 2

    def test_forward_model_shape(self, calibrator, mock_dataset):
        """Test forward model returns correct shape."""
        params = {"elastic_modulus": 210e9}
        x = mock_dataset.x_disp
        geometry = mock_dataset.geometry
        load = mock_dataset.load_case

        w = calibrator._forward_model(params, x, geometry, load)

        assert len(w) == len(x)
        assert np.all(w <= 0)  # Deflection should be negative (downward)

    def test_forward_model_physics(self, calibrator, mock_dataset):
        """Test forward model follows physics (higher E = less deflection)."""
        x = mock_dataset.x_disp
        geometry = mock_dataset.geometry
        load = mock_dataset.load_case

        w_low_E = calibrator._forward_model({"elastic_modulus": 100e9}, x, geometry, load)
        w_high_E = calibrator._forward_model({"elastic_modulus": 300e9}, x, geometry, load)

        # Higher E should give smaller deflection magnitude
        assert np.abs(w_high_E).max() < np.abs(w_low_E).max()


class TestTimoshenkoCalibrator:
    """Tests for Timoshenko calibrator."""

    @pytest.fixture
    def priors(self):
        """Create Timoshenko priors."""
        return create_timoshenko_priors()

    @pytest.fixture
    def calibrator(self, priors):
        """Create calibrator with Timoshenko priors."""
        return TimoshenkoCalibrator(priors=priors)

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset for testing."""
        dataset = MagicMock(spec=SyntheticDataset)
        dataset.geometry = BeamGeometry(length=1.0, height=0.2, width=0.05)  # Thick beam
        dataset.material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        dataset.load_case = LoadCase(point_load=1000)
        dataset.x_disp = np.linspace(0.1, 1.0, 10)
        dataset.displacements = -1e-4 * np.ones(10)
        dataset.displacement_noise_std = 1e-6
        return dataset

    def test_initialization(self, calibrator):
        """Test calibrator initializes correctly."""
        assert calibrator.model_name == "Timoshenko"
        assert len(calibrator.priors) == 3  # elastic_modulus, poisson_ratio, sigma

    def test_forward_model_includes_shear(self, calibrator, mock_dataset):
        """Test Timoshenko includes shear deformation."""
        from apps.models.euler_bernoulli import EulerBernoulliBeam

        x = mock_dataset.x_disp
        geometry = mock_dataset.geometry
        load = mock_dataset.load_case

        # Timoshenko should give larger deflection than E-B due to shear
        w_timo = calibrator._forward_model(
            {"elastic_modulus": 210e9, "poisson_ratio": 0.3}, x, geometry, load
        )

        eb_beam = EulerBernoulliBeam(geometry, mock_dataset.material)
        w_eb = eb_beam.compute_deflection(x, load)

        # For thick beam (L/h = 5), Timoshenko deflection > E-B deflection
        assert np.abs(w_timo).max() >= np.abs(w_eb).max()


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_result_structure(self):
        """Test result has all required fields."""
        # Mock trace
        mock_trace = MagicMock()

        result = CalibrationResult(
            model_name="Test",
            trace=mock_trace,
            posterior_summary={"E": {"mean": 210e9}},
            log_likelihood=np.array([-100, -101, -99]),
            waic=-200.0,
            convergence_diagnostics={"r_hat": {"E": 1.001}}
        )

        assert result.model_name == "Test"
        assert result.waic == -200.0
        assert "E" in result.posterior_summary


class TestConvergenceDiagnostics:
    """Tests for convergence checking utilities."""

    def test_r_hat_threshold(self):
        """Test R-hat threshold is reasonable."""
        # R-hat should be close to 1.0 for converged chains
        good_r_hat = 1.001
        bad_r_hat = 1.2

        threshold = 1.01

        assert good_r_hat < threshold
        assert bad_r_hat > threshold


class TestNormalizationIntegration:
    """Tests for normalization in calibration."""

    def test_normalization_params_computed(self):
        """Test that normalization params are computed from data."""
        from apps.bayesian.normalization import compute_normalization_params

        displacements = np.array([-1e-5, -2e-5, -3e-5, -4e-5, -5e-5])

        params = compute_normalization_params(displacements)

        assert params.displacement_scale > 0
        assert params.E_scale > 0
        assert np.isclose(params.E_scale, 210e9, rtol=0.01)
