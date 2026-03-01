"""
Unit tests for the 1D Timoshenko Beam FEM module.
"""

from __future__ import annotations

import numpy as np
import pytest

from apps.backend.core.fem.beam_fem import TimoshenkoBeamFEM


class TestTimoshenkoBeamFEM:
    """Tests for the 1D Timoshenko beam finite element model."""

    @pytest.fixture
    def standard_beam(self) -> TimoshenkoBeamFEM:
        """Create a standard steel cantilever beam FEM model."""
        return TimoshenkoBeamFEM(
            length=1.0,
            height=0.1,
            width=0.05,
            elastic_modulus=210e9,
            poisson_ratio=0.3,
            n_elements=20,
        )

    # ------------------------------------------------------------------
    # Element stiffness matrix properties
    # ------------------------------------------------------------------

    def test_element_stiffness_symmetry(self, standard_beam: TimoshenkoBeamFEM) -> None:
        Ke = standard_beam._Ke
        assert np.allclose(Ke, Ke.T), "Element stiffness matrix must be symmetric"

    def test_element_stiffness_positive_semidefinite(
        self, standard_beam: TimoshenkoBeamFEM
    ) -> None:
        eigenvalues = np.linalg.eigvalsh(standard_beam._Ke)
        assert np.all(eigenvalues >= -1e-6), "Ke eigenvalues must be non-negative"

    # ------------------------------------------------------------------
    # Boundary conditions & simple sanity checks
    # ------------------------------------------------------------------

    def test_fixed_end_zero_displacement(self, standard_beam: TimoshenkoBeamFEM) -> None:
        result = standard_beam.solve(point_load=1000.0)
        assert result.deflections[0] == pytest.approx(0.0, abs=1e-15)
        assert result.rotations[0] == pytest.approx(0.0, abs=1e-15)

    def test_tip_deflection_sign(self, standard_beam: TimoshenkoBeamFEM) -> None:
        """Positive point_load (downward) should produce negative tip deflection."""
        result = standard_beam.solve(point_load=1000.0)
        assert result.tip_deflection < 0

    def test_no_load_zero_solution(self, standard_beam: TimoshenkoBeamFEM) -> None:
        result = standard_beam.solve(point_load=0.0, distributed_load=0.0)
        assert np.allclose(result.deflections, 0.0)
        assert np.allclose(result.rotations, 0.0)

    # ------------------------------------------------------------------
    # Accuracy against analytical Timoshenko beam theory
    # ------------------------------------------------------------------

    def test_tip_deflection_matches_analytical(self, standard_beam: TimoshenkoBeamFEM) -> None:
        """FEM tip deflection should match Timoshenko analytical solution within 1%."""
        from apps.backend.core.models.base_beam import BeamGeometry, LoadCase, MaterialProperties
        from apps.backend.core.models.timoshenko import TimoshenkoBeam

        P = 1000.0
        result = standard_beam.solve(point_load=P)

        geometry = BeamGeometry(
            length=standard_beam.L,
            height=standard_beam.h,
            width=standard_beam.b,
        )
        material = MaterialProperties(
            elastic_modulus=standard_beam.E,
            poisson_ratio=standard_beam.nu,
        )
        analytical = TimoshenkoBeam(geometry, material)
        w_analytical = analytical.tip_deflection(LoadCase(point_load=P))

        assert abs(result.tip_deflection) == pytest.approx(abs(w_analytical), rel=0.01), (
            "FEM and analytical Timoshenko should agree within 1%"
        )

    # ------------------------------------------------------------------
    # Mesh convergence
    # ------------------------------------------------------------------

    def test_mesh_convergence(self) -> None:
        """Finer meshes should converge to a stable tip deflection."""
        P = 1000.0
        tips = []
        for n_elem in [5, 10, 20, 40]:
            fem = TimoshenkoBeamFEM(
                length=1.0,
                height=0.1,
                width=0.05,
                elastic_modulus=210e9,
                poisson_ratio=0.3,
                n_elements=n_elem,
            )
            tips.append(fem.solve(point_load=P).tip_deflection)

        # Each step closer → smaller change
        diffs = [abs(tips[i + 1] - tips[i]) for i in range(len(tips) - 1)]
        for i in range(len(diffs) - 1):
            assert diffs[i + 1] <= diffs[i] + 1e-14

    # ------------------------------------------------------------------
    # Result container tests
    # ------------------------------------------------------------------

    def test_result_shape(self, standard_beam: TimoshenkoBeamFEM) -> None:
        result = standard_beam.solve(point_load=500.0)
        n_nodes = standard_beam.n_nodes
        assert result.deflections.shape == (n_nodes,)
        assert result.rotations.shape == (n_nodes,)
        assert result.x.shape == (n_nodes,)

    def test_get_deflection_at(self, standard_beam: TimoshenkoBeamFEM) -> None:
        result = standard_beam.solve(point_load=500.0)
        x_query = np.array([0.0, 0.5, 1.0])
        interp = result.get_deflection_at(x_query)
        assert interp.shape == (3,)
        assert interp[0] == pytest.approx(0.0, abs=1e-12)

    # ------------------------------------------------------------------
    # Distributed load
    # ------------------------------------------------------------------

    def test_distributed_load(self, standard_beam: TimoshenkoBeamFEM) -> None:
        """Distributed load should produce deflection everywhere except fixed end."""
        result = standard_beam.solve(distributed_load=5000.0)
        assert result.deflections[0] == pytest.approx(0.0, abs=1e-15)
        assert result.tip_deflection < 0
