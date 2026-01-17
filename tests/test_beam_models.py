"""
Unit tests for beam model implementations.
"""

import numpy as np
import pytest

from apps.models.base_beam import BeamGeometry, MaterialProperties, LoadCase
from apps.models.euler_bernoulli import EulerBernoulliBeam
from apps.models.timoshenko import TimoshenkoBeam


class TestBeamGeometry:
    """Tests for BeamGeometry dataclass."""

    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.05)
        assert geometry.aspect_ratio == 10.0

    def test_computed_properties(self):
        """Test automatically computed properties."""
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.05)
        assert geometry.area == 0.005  # 0.1 * 0.05
        # I = bh³/12 = 0.05 * 0.1³/12 = 4.1667e-6
        assert np.isclose(geometry.moment_of_inertia, 4.1667e-6, rtol=1e-3)


class TestMaterialProperties:
    """Tests for MaterialProperties dataclass."""

    def test_shear_modulus_computation(self):
        """Test automatic shear modulus calculation."""
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        # G = E / (2*(1+ν)) = 210e9 / 2.6 = 80.77e9
        expected_G = 210e9 / (2 * 1.3)
        assert np.isclose(material.shear_modulus, expected_G)


class TestEulerBernoulliBeam:
    """Tests for Euler-Bernoulli beam model."""

    @pytest.fixture
    def setup_beam(self):
        """Create standard beam for testing."""
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.05)
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        return EulerBernoulliBeam(geometry, material)

    def test_tip_deflection_point_load(self, setup_beam):
        """Test tip deflection under point load."""
        beam = setup_beam
        load = LoadCase(point_load=1000)

        # Analytical: w_tip = PL³/(3EI)
        P, L = 1000, 1.0
        EI = beam.flexural_rigidity
        expected = P * L**3 / (3 * EI)

        tip_deflection = beam.tip_deflection(load)
        assert np.isclose(tip_deflection, expected, rtol=1e-6)

    def test_zero_deflection_at_fixed_end(self, setup_beam):
        """Test that deflection is zero at fixed end."""
        beam = setup_beam
        load = LoadCase(point_load=1000)

        x = np.array([0.0])
        w = beam.compute_deflection(x, load)
        assert np.isclose(w[0], 0.0)

    def test_deflection_shape(self, setup_beam):
        """Test that deflection increases along beam."""
        beam = setup_beam
        load = LoadCase(point_load=1000)

        x = np.linspace(0, 1.0, 10)
        w = beam.compute_deflection(x, load)

        # Deflection should increase monotonically
        assert all(np.diff(w) > 0)

    def test_natural_frequencies_order(self, setup_beam):
        """Test that natural frequencies increase with mode number."""
        beam = setup_beam
        frequencies = beam.compute_natural_frequencies(n_modes=5)

        assert all(np.diff(frequencies) > 0)

    # TODO: Add more tests as implementation is completed
    # TODO: Task T1 - Add tests for distributed load
    # TODO: Task T2 - Add tests for strain computation
    # TODO: Task T3 - Add convergence tests


class TestTimoshenkoBeam:
    """Tests for Timoshenko beam model."""

    @pytest.fixture
    def setup_beam(self):
        """Create standard beam for testing."""
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.05)
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        return TimoshenkoBeam(geometry, material)

    def test_tip_deflection_greater_than_eb(self, setup_beam):
        """Test that Timoshenko deflection >= Euler-Bernoulli."""
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.05)
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)

        eb_beam = EulerBernoulliBeam(geometry, material)
        timo_beam = TimoshenkoBeam(geometry, material)

        load = LoadCase(point_load=1000)

        tip_eb = eb_beam.tip_deflection(load)
        tip_timo = timo_beam.tip_deflection(load)

        # Timoshenko includes shear deformation, so deflection should be larger
        assert tip_timo >= tip_eb

    def test_shear_deformation_ratio(self, setup_beam):
        """Test shear deformation contribution."""
        beam = setup_beam
        load = LoadCase(point_load=1000)

        ratio = beam.shear_deformation_ratio(load)

        # For L/h=10, shear contribution should be small but positive
        assert 0 < ratio < 1

    def test_convergence_to_eb_for_slender_beam(self):
        """Test that Timoshenko converges to EB for slender beams."""
        # Very slender beam (L/h = 100)
        geometry = BeamGeometry(length=1.0, height=0.01, width=0.05)
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)

        eb_beam = EulerBernoulliBeam(geometry, material)
        timo_beam = TimoshenkoBeam(geometry, material)

        load = LoadCase(point_load=1000)

        tip_eb = eb_beam.tip_deflection(load)
        tip_timo = timo_beam.tip_deflection(load)

        # Should be very close for slender beam
        assert np.isclose(tip_timo, tip_eb, rtol=0.01)

    def test_shear_parameter(self, setup_beam):
        """Test shear parameter calculation."""
        beam = setup_beam

        # Φ = 12*EI / (κ*G*A*L²) should be positive
        phi = beam.shear_parameter
        assert phi > 0

    # TODO: Task T4 - Add tests for frequency comparison
    # TODO: Task T5 - Add tests for thick beams (L/h < 10)


class TestBeamComparison:
    """Comparison tests between beam theories."""

    def test_deflection_difference_increases_with_thickness(self):
        """Test that difference between theories increases for thick beams."""
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        load = LoadCase(point_load=1000)

        aspect_ratios = [5, 10, 20, 50]
        differences = []

        for L_h in aspect_ratios:
            geometry = BeamGeometry(length=1.0, height=1.0/L_h, width=0.05)
            eb = EulerBernoulliBeam(geometry, material)
            timo = TimoshenkoBeam(geometry, material)

            diff = (timo.tip_deflection(load) - eb.tip_deflection(load)) / timo.tip_deflection(load)
            differences.append(diff)

        # Difference should decrease as beam gets more slender
        assert all(np.diff(differences) < 0)
