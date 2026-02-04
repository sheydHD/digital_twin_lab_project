"""
Unit tests for beam model implementations.
"""

import numpy as np
import pytest

from apps.models.base_beam import BeamGeometry, LoadCase, MaterialProperties
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
        assert np.isclose(geometry.area, 0.005)  # 0.1 * 0.05
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

        # Analytical: w_tip = -PL³/(3EI) (negative for downward)
        P, L = 1000, 1.0
        EI = beam.flexural_rigidity
        expected = -P * L**3 / (3 * EI)

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
        """Test that deflection increases (in magnitude) along beam."""
        beam = setup_beam
        load = LoadCase(point_load=1000)

        x = np.linspace(0, 1.0, 10)
        w = beam.compute_deflection(x, load)

        # Deflection magnitude should increase monotonically
        # (deflection is negative for downward, so np.diff should be negative)
        assert all(np.diff(w) < 0)

    def test_natural_frequencies_order(self, setup_beam):
        """Test that natural frequencies increase with mode number."""
        beam = setup_beam
        frequencies = beam.compute_natural_frequencies(n_modes=5)

        assert all(np.diff(frequencies) > 0)

    def test_distributed_load_deflection(self, setup_beam):
        """Test deflection under distributed load (Task T1)."""
        beam = setup_beam
        q = 1000  # N/m
        load = LoadCase(distributed_load=q)

        # Analytical: w_tip = -qL⁴/(8EI) (negative for downward)
        L = beam.geometry.length
        EI = beam.flexural_rigidity
        expected = -q * L**4 / (8 * EI)

        tip_deflection = beam.tip_deflection(load)
        assert np.isclose(tip_deflection, expected, rtol=1e-6)

    def test_strain_at_surface(self, setup_beam):
        """Test strain computation at beam surface (Task T2)."""
        beam = setup_beam
        load = LoadCase(point_load=1000)

        # Strain at fixed end (x=0) should be maximum for cantilever
        x = np.array([0.0, 0.5, 1.0])
        y = beam.geometry.height / 2  # Top surface

        strain = beam.compute_strain(x, y, load)

        # Strain should be negative (compression) at top surface at fixed end
        # for a cantilever with downward tip load
        assert strain[0] < 0  # Compression at fixed end

        # Strain should be zero at free end (no moment)
        assert np.isclose(strain[-1], 0.0, atol=1e-12)

    def test_moment_distribution(self, setup_beam):
        """Test bending moment distribution (Task T3)."""
        beam = setup_beam
        P = 1000
        load = LoadCase(point_load=P)

        x = np.array([0.0, 0.5, 1.0])
        L = beam.geometry.length
        M = beam.compute_moment(x, load)

        # M(x) = P*(L - x) for cantilever with tip load
        expected = P * (L - x)
        assert np.allclose(M, expected, rtol=1e-6)


class TestTimoshenkoBeam:
    """Tests for Timoshenko beam model."""

    @pytest.fixture
    def setup_beam(self):
        """Create standard beam for testing."""
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.05)
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        return TimoshenkoBeam(geometry, material)

    def test_tip_deflection_greater_than_eb(self, setup_beam):
        """Test that Timoshenko deflection magnitude >= Euler-Bernoulli."""
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.05)
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)

        eb_beam = EulerBernoulliBeam(geometry, material)
        timo_beam = TimoshenkoBeam(geometry, material)

        load = LoadCase(point_load=1000)

        tip_eb = eb_beam.tip_deflection(load)
        tip_timo = timo_beam.tip_deflection(load)

        # Timoshenko includes shear, so |deflection| should be larger (more negative)
        assert abs(tip_timo) >= abs(tip_eb)

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

    def test_frequency_lower_than_eb(self):
        """Test that Timoshenko frequencies are lower than EB (Task T4)."""
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.05)
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)

        eb_beam = EulerBernoulliBeam(geometry, material)
        timo_beam = TimoshenkoBeam(geometry, material)

        freq_eb = eb_beam.compute_natural_frequencies(5)
        freq_timo = timo_beam.compute_natural_frequencies(5)

        # Timoshenko frequencies should be lower (more flexible model)
        for f_eb, f_t in zip(freq_eb, freq_timo, strict=False):
            assert f_t <= f_eb

    def test_thick_beam_significant_shear(self):
        """Test that thick beams have significant shear deformation (Task T5)."""
        # Very thick beam (L/h = 3)
        geometry = BeamGeometry(length=0.3, height=0.1, width=0.05)
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)

        timo_beam = TimoshenkoBeam(geometry, material)
        load = LoadCase(point_load=1000)

        ratio = timo_beam.shear_deformation_ratio(load)

        # For very thick beam, shear should contribute significantly (> 3%)
        assert ratio > 0.03


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
