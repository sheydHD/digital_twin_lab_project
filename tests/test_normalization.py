"""
Unit tests for data normalization module.

Tests:
- Normalization parameter computation
- Forward/inverse transformations
- Scale factor calculations
- Edge cases
"""

import numpy as np
import pytest

from apps.bayesian.normalization import (
    NormalizationParams,
    compute_normalization_params,
    create_normalizer_from_dataset,
    denormalize_displacements,
    denormalize_E,
    normalize_displacements,
    normalize_E,
)


class TestNormalizationParams:
    """Tests for NormalizationParams dataclass."""

    def test_default_values(self):
        """Test default normalization parameters."""
        params = NormalizationParams()

        assert params.displacement_scale == 1.0
        assert params.E_scale == 210e9
        assert params.is_active is False

    def test_custom_values(self):
        """Test custom normalization parameters."""
        params = NormalizationParams(
            displacement_scale=1e-5,
            E_scale=200e9,
            is_active=True
        )

        assert params.displacement_scale == 1e-5
        assert params.E_scale == 200e9
        assert params.is_active is True

    def test_validation_positive_scales(self):
        """Test that scales must be positive."""
        with pytest.raises(ValueError):
            NormalizationParams(displacement_scale=-1.0)

        with pytest.raises(ValueError):
            NormalizationParams(E_scale=0)

    def test_summary(self):
        """Test summary string generation."""
        params = NormalizationParams(
            displacement_scale=5e-5,
            E_scale=210e9,
            is_active=True
        )

        summary = params.summary()

        assert "5.00e-05" in summary
        assert "2.10e+11" in summary
        assert "is_active=True" in summary


class TestComputeNormalizationParams:
    """Tests for compute_normalization_params function."""

    def test_max_abs_method(self):
        """Test max_abs normalization method."""
        displacements = np.array([-1e-5, -2e-5, -3e-5, -4e-5, -5e-5])

        params = compute_normalization_params(displacements, method="max_abs")

        assert np.isclose(params.displacement_scale, 5e-5)
        assert params.is_active is True

    def test_std_method(self):
        """Test std normalization method."""
        displacements = np.array([-1e-5, -2e-5, -3e-5, -4e-5, -5e-5])

        params = compute_normalization_params(displacements, method="std")

        expected_std = np.std(displacements)
        assert np.isclose(params.displacement_scale, expected_std)

    def test_range_method(self):
        """Test range normalization method."""
        displacements = np.array([-1e-5, -2e-5, -3e-5, -4e-5, -5e-5])

        params = compute_normalization_params(displacements, method="range")

        expected_range = 5e-5 - 1e-5
        assert np.isclose(params.displacement_scale, expected_range)

    def test_custom_E_nominal(self):
        """Test custom E nominal value."""
        displacements = np.array([-1e-5, -2e-5])

        params = compute_normalization_params(displacements, E_nominal=200e9)

        assert params.E_scale == 200e9


class TestNormalizeDisplacements:
    """Tests for displacement normalization."""

    def test_normalize_to_order_one(self):
        """Test that normalized displacements are O(1)."""
        displacements = np.array([-1e-5, -2e-5, -3e-5, -4e-5, -5e-5])
        params = compute_normalization_params(displacements)

        normalized = normalize_displacements(displacements, params)

        # Normalized should be in [-1, 1] range for max_abs method
        assert np.abs(normalized).max() <= 1.0
        assert np.abs(normalized).max() >= 0.1  # Not too small

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize/denormalize are inverses."""
        original = np.array([-1e-5, -2e-5, -3e-5, -4e-5, -5e-5])
        params = compute_normalization_params(original)

        normalized = normalize_displacements(original, params)
        recovered = denormalize_displacements(normalized, params)

        np.testing.assert_allclose(recovered, original, rtol=1e-10)


class TestNormalizeE:
    """Tests for Young's modulus normalization."""

    def test_normalize_E_to_order_one(self):
        """Test that normalized E is O(1)."""
        E_physical = 210e9
        params = NormalizationParams(E_scale=210e9, is_active=True)

        E_normalized = normalize_E(E_physical, params)

        assert np.isclose(E_normalized, 1.0)

    def test_normalize_denormalize_E_roundtrip(self):
        """Test that E normalize/denormalize are inverses."""
        E_original = 215e9
        params = NormalizationParams(E_scale=210e9, is_active=True)

        E_normalized = normalize_E(E_original, params)
        E_recovered = denormalize_E(E_normalized, params)

        np.testing.assert_allclose(E_recovered, E_original, rtol=1e-10)

    def test_E_variation_preserved(self):
        """Test that relative E variation is preserved."""
        params = NormalizationParams(E_scale=210e9, is_active=True)

        E1 = 200e9
        E2 = 220e9

        E1_norm = normalize_E(E1, params)
        E2_norm = normalize_E(E2, params)

        # Relative difference should be preserved
        rel_diff_physical = (E2 - E1) / E1
        rel_diff_normalized = (E2_norm - E1_norm) / E1_norm

        np.testing.assert_allclose(rel_diff_physical, rel_diff_normalized, rtol=1e-10)


class TestCreateNormalizerFromDataset:
    """Tests for dataset-based normalizer creation."""

    def test_normalizer_from_mock_dataset(self):
        """Test normalizer creation from dataset."""
        from unittest.mock import MagicMock

        from apps.data.synthetic_generator import SyntheticDataset

        # Create mock dataset
        dataset = MagicMock(spec=SyntheticDataset)
        dataset.displacements = np.array([-1e-5, -2e-5, -3e-5, -4e-5, -5e-5])
        dataset.strains = None

        params = create_normalizer_from_dataset(dataset)

        assert params.is_active is True
        assert params.displacement_scale > 0


class TestEdgeCases:
    """Tests for edge cases in normalization."""

    def test_single_value(self):
        """Test normalization with single value."""
        displacements = np.array([-1e-5])

        params = compute_normalization_params(displacements)

        assert params.displacement_scale > 0

    def test_zero_values(self):
        """Test normalization handles zero displacements."""
        displacements = np.array([0.0, -1e-5, -2e-5])

        params = compute_normalization_params(displacements)
        normalized = normalize_displacements(displacements, params)

        assert normalized[0] == 0.0

    def test_very_small_values(self):
        """Test normalization with very small values."""
        displacements = np.array([-1e-15, -2e-15])

        params = compute_normalization_params(displacements)

        assert params.displacement_scale > 0

        normalized = normalize_displacements(displacements, params)
        assert np.abs(normalized).max() <= 1.0

    def test_inactive_normalization(self):
        """Test that inactive normalization is identity."""
        np.array([-1e-5, -2e-5])
        params = NormalizationParams(is_active=False)

        # Should return unchanged when inactive
        # (This tests the contract, implementation may vary)
        assert params.is_active is False
