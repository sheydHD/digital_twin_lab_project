"""
Unit tests for FEM module.
"""

import numpy as np
import pytest

from apps.fem.cantilever_fem import CantileverFEM


class TestCantileverFEM:
    """Tests for the FEM cantilever model."""

    @pytest.fixture
    def setup_fem(self):
        """Create standard FEM model for testing."""
        return CantileverFEM(
            length=1.0,
            height=0.1,
            thickness=0.05,
            elastic_modulus=210e9,
            poisson_ratio=0.3,
            n_elements_x=20,
            n_elements_y=4,
        )

    def test_mesh_generation(self, setup_fem):
        """Test that mesh is generated correctly."""
        fem = setup_fem
        mesh = fem.mesh

        # Check dimensions
        n_nodes_x = fem.n_elem_x + 1
        n_nodes_y = fem.n_elem_y + 1
        expected_nodes = n_nodes_x * n_nodes_y

        assert mesh.n_nodes == expected_nodes
        assert mesh.n_elements == fem.n_elem_x * fem.n_elem_y

    def test_node_coordinates_range(self, setup_fem):
        """Test that node coordinates are within beam dimensions."""
        fem = setup_fem
        nodes = fem.mesh.nodes

        assert np.all(nodes[:, 0] >= 0)
        assert np.all(nodes[:, 0] <= fem.length)
        assert np.all(nodes[:, 1] >= -fem.height / 2)
        assert np.all(nodes[:, 1] <= fem.height / 2)

    def test_plane_stress_matrix_symmetry(self, setup_fem):
        """Test that constitutive matrix is symmetric."""
        fem = setup_fem
        D = fem.D

        assert np.allclose(D, D.T)

    def test_element_stiffness_symmetry(self, setup_fem):
        """Test that element stiffness matrix is symmetric."""
        fem = setup_fem
        Ke = fem._element_stiffness(0)

        assert np.allclose(Ke, Ke.T)

    def test_element_stiffness_positive_definite(self, setup_fem):
        """Test that element stiffness matrix is positive semi-definite."""
        fem = setup_fem
        Ke = fem._element_stiffness(0)

        eigenvalues = np.linalg.eigvalsh(Ke)
        # Should be non-negative (allowing for numerical tolerance)
        # Single element has 3 rigid body modes with near-zero eigenvalues
        assert np.all(eigenvalues >= -1e-5)

    def test_fem_solution_tip_deflection(self, setup_fem):
        """Test FEM solution against analytical tip deflection (Task T6)."""
        fem = setup_fem
        P = 1000  # Point load at tip

        # Solve FEM problem
        result = fem.solve(point_load=P)

        # Get analytical Timoshenko solution for comparison
        from apps.models.base_beam import BeamGeometry, LoadCase, MaterialProperties
        from apps.models.timoshenko import TimoshenkoBeam

        geometry = BeamGeometry(length=fem.length, height=fem.height, width=fem.thickness)
        material = MaterialProperties(
            elastic_modulus=fem.E,
            poisson_ratio=fem.nu,
        )
        timo = TimoshenkoBeam(geometry, material)
        load = LoadCase(point_load=P)
        w_analytical = timo.tip_deflection(load)

        # FEM should be within 25% of analytical (2D plane stress differs from 1D beam theory)
        # Compare magnitudes since both should be negative (downward)
        assert np.isclose(abs(result.tip_deflection), abs(w_analytical), rtol=0.25)

    def test_mesh_convergence(self):
        """Test that solution converges with mesh refinement (Task T7)."""
        P = 1000
        tip_deflections = []

        for n_elem_x in [10, 20, 40]:
            fem = CantileverFEM(
                length=1.0,
                height=0.1,
                thickness=0.05,
                elastic_modulus=210e9,
                poisson_ratio=0.3,
                n_elements_x=n_elem_x,
                n_elements_y=4,
            )
            result = fem.solve(point_load=P)
            tip_deflections.append(result.tip_deflection)

        # Differences should decrease with refinement
        diff1 = abs(tip_deflections[1] - tip_deflections[0])
        diff2 = abs(tip_deflections[2] - tip_deflections[1])
        assert diff2 < diff1

    def test_strain_extraction(self, setup_fem):
        """Test strain extraction at sensor locations (Task T8)."""
        fem = setup_fem
        P = 1000
        result = fem.solve(point_load=P)

        # Check that strains were computed
        assert result.strains is not None
        assert len(result.strains) == fem.mesh.n_elements

        # Extract surface strains
        x_data, strain_data = fem.extract_surface_strain(result.strains, "top")

        # Should have some data points
        assert len(x_data) > 0

        # For cantilever with downward tip load:
        # - Top surface is in tension (positive strain)
        # - Bottom surface would be in compression (negative strain)
        assert np.mean(strain_data) > 0


class TestShapeFunctions:
    """Tests for FEM shape functions."""

    @pytest.fixture
    def setup_fem(self):
        return CantileverFEM(
            length=1.0,
            height=0.1,
            thickness=0.05,
            elastic_modulus=210e9,
            poisson_ratio=0.3,
        )

    def test_shape_functions_sum_to_one(self, setup_fem):
        """Test that shape functions sum to 1 (partition of unity)."""
        fem = setup_fem

        # Test at various points
        test_points = [
            (0, 0),
            (0.5, 0.5),
            (-0.5, -0.5),
            (1, 1),
            (-1, -1),
        ]

        for xi, eta in test_points:
            N, _ = fem._q4_shape_functions(xi, eta)
            assert np.isclose(np.sum(N), 1.0)

    def test_shape_functions_at_corners(self, setup_fem):
        """Test shape function values at element corners."""
        fem = setup_fem

        # At node 1 (xi=-1, eta=-1), N1=1, others=0
        N, _ = fem._q4_shape_functions(-1, -1)
        assert np.isclose(N[0], 1.0)
        assert np.allclose(N[1:], 0.0)

        # At node 2 (xi=1, eta=-1), N2=1, others=0
        N, _ = fem._q4_shape_functions(1, -1)
        assert np.isclose(N[1], 1.0)
