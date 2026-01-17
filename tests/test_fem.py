"""
Unit tests for FEM module.
"""

import numpy as np
import pytest

from apps.fem.cantilever_fem import CantileverFEM, FEMMesh


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
        assert np.all(eigenvalues >= -1e-10)

    # TODO: Task T6 - Add FEM solution validation tests
    # TODO: Task T7 - Add mesh convergence tests
    # TODO: Task T8 - Add strain extraction tests


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
