"""
High-Fidelity Finite Element Model for Cantilever Beam.

This module provides a reference FEM implementation using 2D plane stress elements
to generate "ground truth" synthetic measurement data for Bayesian model selection.

The FEM model serves as a high-fidelity reference that captures:
- Full 2D stress/strain distribution
- Shear deformation effects accurately
- End effects and load introduction
- True structural response for validation

This is used to generate synthetic sensor data (displacements, strains) that will
be compared against analytical beam theory predictions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve


@dataclass
class FEMMesh:
    """
    Finite element mesh data structure.

    Attributes:
        nodes: Node coordinates (n_nodes, 2)
        elements: Element connectivity (n_elements, nodes_per_element)
        n_nodes: Total number of nodes
        n_elements: Total number of elements
        element_type: Type of element ('Q4', 'Q8', 'T3', 'T6')
    """

    nodes: np.ndarray
    elements: np.ndarray
    n_nodes: int
    n_elements: int
    element_type: str = "Q4"  # 4-node quadrilateral


@dataclass
class FEMResult:
    """
    Result container for FEM analysis.

    Attributes:
        displacements: Nodal displacements (n_nodes, 2)
        strains: Element strains (n_elements, 3) - [eps_xx, eps_yy, gamma_xy]
        stresses: Element stresses (n_elements, 3) - [sigma_xx, sigma_yy, tau_xy]
        mesh: Reference to mesh
        fem: Reference to FEM model
    """

    displacements: np.ndarray
    strains: np.ndarray
    stresses: np.ndarray
    mesh: FEMMesh
    fem: "CantileverFEM"

    @property
    def tip_deflection(self) -> float:
        """Get maximum deflection at beam tip."""
        # Find nodes at tip (x = length)
        tip_mask = np.abs(self.mesh.nodes[:, 0] - self.fem.length) < 1e-10
        tip_deflections = self.displacements[tip_mask, 1]
        # Return centerline deflection (average or center node)
        return np.mean(tip_deflections)

    def get_centerline_deflection(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get deflection along beam centerline."""
        return self.fem.extract_centerline_deflection(self.displacements)

    def get_strain_at_points(
        self,
        x_points: np.ndarray,
        y_coord: float,
    ) -> np.ndarray:
        """
        Interpolate axial strain at specified points.

        Args:
            x_points: x-coordinates for strain extraction
            y_coord: y-coordinate (distance from neutral axis)

        Returns:
            Interpolated axial strains at requested points
        """
        # Determine if top or bottom surface
        surface = "top" if y_coord > 0 else "bottom"

        # Extract surface strains
        x_data, strain_data = self.fem.extract_surface_strain(self.strains, surface)

        if len(x_data) < 2:
            return np.zeros_like(x_points)

        # Interpolate to requested points
        interp_func = interp1d(
            x_data, strain_data,
            kind='linear',
            bounds_error=False,
            fill_value=(strain_data[0], strain_data[-1])
        )
        return interp_func(x_points)


@dataclass
class FEMBoundaryConditions:
    """
    Boundary conditions for FEM analysis.

    Attributes:
        fixed_dofs: List of fixed degrees of freedom
        prescribed_displacements: Dict of DOF -> displacement value
        point_loads: Dict of DOF -> load value
        distributed_loads: Applied pressure/traction arrays
    """

    fixed_dofs: List[int]
    prescribed_displacements: Optional[dict] = None
    point_loads: Optional[dict] = None
    distributed_loads: Optional[np.ndarray] = None


class CantileverFEM:
    """
    2D Finite Element Model for cantilever beam using plane stress elements.

    This class generates high-fidelity synthetic data by solving the full
    2D elasticity problem, capturing effects that simplified beam theories
    cannot represent exactly.
    """

    def __init__(
        self,
        length: float,
        height: float,
        thickness: float,
        elastic_modulus: float,
        poisson_ratio: float,
        n_elements_x: int = 40,
        n_elements_y: int = 8,
    ):
        """
        Initialize cantilever beam FEM model.

        Args:
            length: Beam length [m]
            height: Beam height (cross-section) [m]
            thickness: Beam thickness (out-of-plane) [m]
            elastic_modulus: Young's modulus [Pa]
            poisson_ratio: Poisson's ratio [-]
            n_elements_x: Number of elements along length
            n_elements_y: Number of elements through height
        """
        self.length = length
        self.height = height
        self.thickness = thickness
        self.E = elastic_modulus
        self.nu = poisson_ratio
        self.n_elem_x = n_elements_x
        self.n_elem_y = n_elements_y

        # Generate mesh
        self.mesh = self._generate_mesh()

        # Compute element stiffness matrices
        self.D = self._plane_stress_matrix()

        # Global stiffness matrix (assembled lazily)
        self._K_global = None

    def _generate_mesh(self) -> FEMMesh:
        """
        Generate structured Q4 mesh for rectangular beam.

        Returns:
            FEMMesh object with nodes and connectivity

        """
        # Node generation
        n_nodes_x = self.n_elem_x + 1
        n_nodes_y = self.n_elem_y + 1

        # Create node coordinates
        x_coords = np.linspace(0, self.length, n_nodes_x)
        y_coords = np.linspace(-self.height / 2, self.height / 2, n_nodes_y)

        # Generate node grid
        nodes = np.zeros((n_nodes_x * n_nodes_y, 2))
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                node_id = i * n_nodes_y + j
                nodes[node_id, 0] = x
                nodes[node_id, 1] = y

        # Element connectivity (Q4 elements)
        n_elements = self.n_elem_x * self.n_elem_y
        elements = np.zeros((n_elements, 4), dtype=int)

        elem_id = 0
        for i in range(self.n_elem_x):
            for j in range(self.n_elem_y):
                # Node numbering (counter-clockwise)
                n1 = i * n_nodes_y + j
                n2 = (i + 1) * n_nodes_y + j
                n3 = (i + 1) * n_nodes_y + (j + 1)
                n4 = i * n_nodes_y + (j + 1)
                elements[elem_id] = [n1, n2, n3, n4]
                elem_id += 1

        return FEMMesh(
            nodes=nodes,
            elements=elements,
            n_nodes=n_nodes_x * n_nodes_y,
            n_elements=n_elements,
            element_type="Q4",
        )

    def _plane_stress_matrix(self) -> np.ndarray:
        """
        Compute plane stress constitutive matrix.

        D = (E / (1 - ν²)) * [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]

        Returns:
            3x3 constitutive matrix

        """
        E = self.E
        nu = self.nu

        factor = E / (1 - nu**2)
        D = factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

        return D

    def _q4_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Q4 shape functions and their derivatives.

        Args:
            xi: Natural coordinate [-1, 1]
            eta: Natural coordinate [-1, 1]

        Returns:
            N: Shape functions (4,)
            dN: Shape function derivatives (4, 2)

        """
        # Shape functions
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])

        # Derivatives w.r.t. natural coordinates
        dN_dxi = 0.25 * np.array([
            [-(1 - eta), -(1 - xi)],
            [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)],
            [-(1 + eta), (1 - xi)]
        ])

        return N, dN_dxi

    def _element_stiffness(self, elem_id: int) -> np.ndarray:
        """
        Compute element stiffness matrix using numerical integration.

        Args:
            elem_id: Element index

        Returns:
            8x8 element stiffness matrix (2 DOF per node, 4 nodes)

        """
        # Get element nodes
        elem_nodes = self.mesh.elements[elem_id]
        coords = self.mesh.nodes[elem_nodes]

        # Gauss points (2x2 integration)
        gauss_pts = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        weights = [1, 1]

        # Initialize element stiffness
        Ke = np.zeros((8, 8))

        for i, xi in enumerate(gauss_pts):
            for j, eta in enumerate(gauss_pts):
                # Get shape functions and derivatives
                N, dN_dxi = self._q4_shape_functions(xi, eta)

                # Jacobian matrix
                J = dN_dxi.T @ coords  # 2x2

                # Determinant and inverse
                det_J = np.linalg.det(J)
                J_inv = np.linalg.inv(J)

                # Derivatives in physical coordinates
                dN_dx = dN_dxi @ J_inv  # 4x2

                # Strain-displacement matrix B (3x8)
                B = np.zeros((3, 8))
                for k in range(4):
                    B[0, 2 * k] = dN_dx[k, 0]      # ε_xx
                    B[1, 2 * k + 1] = dN_dx[k, 1]  # ε_yy
                    B[2, 2 * k] = dN_dx[k, 1]      # γ_xy
                    B[2, 2 * k + 1] = dN_dx[k, 0]

                # Add contribution
                Ke += weights[i] * weights[j] * B.T @ self.D @ B * det_J * self.thickness

        return Ke

    def _assemble_global_stiffness(self) -> sparse.csr_matrix:
        """
        Assemble global stiffness matrix from element contributions.

        Returns:
            Sparse global stiffness matrix

        """
        n_dof = 2 * self.mesh.n_nodes

        # Use COO format for assembly
        rows = []
        cols = []
        data = []

        for elem_id in range(self.mesh.n_elements):
            Ke = self._element_stiffness(elem_id)
            elem_nodes = self.mesh.elements[elem_id]

            # Global DOF mapping
            global_dofs = []
            for node in elem_nodes:
                global_dofs.extend([2 * node, 2 * node + 1])

            # Add to global matrix
            for i, gi in enumerate(global_dofs):
                for j, gj in enumerate(global_dofs):
                    rows.append(gi)
                    cols.append(gj)
                    data.append(Ke[i, j])

        K = sparse.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof))
        return K.tocsr()

    def setup_cantilever_bc(
        self,
        point_load: float = 0.0,
        distributed_load: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up boundary conditions for cantilever beam.

        Fixed end at x=0, load applied at x=L.

        Args:
            point_load: Point load at tip [N]
            distributed_load: Uniformly distributed load [N/m]

        Returns:
            fixed_dofs: Array of constrained DOFs
            force_vector: Global force vector
        """
        n_dof = 2 * self.mesh.n_nodes

        # Find nodes at fixed end (x = 0)
        fixed_nodes = np.where(np.abs(self.mesh.nodes[:, 0]) < 1e-10)[0]

        # Fixed DOFs (both x and y directions)
        fixed_dofs = []
        for node in fixed_nodes:
            fixed_dofs.extend([2 * node, 2 * node + 1])
        fixed_dofs = np.array(fixed_dofs)

        # Force vector
        F = np.zeros(n_dof)

        # Apply tip load (distributed across tip nodes)
        if point_load != 0:
            tip_nodes = np.where(
                np.abs(self.mesh.nodes[:, 0] - self.length) < 1e-10
            )[0]

            # Distribute load across tip nodes (linear distribution for uniform stress)
            load_per_node = point_load / len(tip_nodes)
            for node in tip_nodes:
                F[2 * node + 1] = -load_per_node  # Negative y-direction

        # Apply distributed load on top surface
        if distributed_load != 0:
            # Find top surface nodes
            top_nodes = np.where(
                np.abs(self.mesh.nodes[:, 1] - self.height / 2) < 1e-10
            )[0]
            # Sort by x
            top_nodes = top_nodes[np.argsort(self.mesh.nodes[top_nodes, 0])]

            # Apply consistent nodal loads (trapezoidal rule)
            if len(top_nodes) > 1:
                for i in range(len(top_nodes) - 1):
                    n1, n2 = top_nodes[i], top_nodes[i + 1]
                    dx = self.mesh.nodes[n2, 0] - self.mesh.nodes[n1, 0]
                    load_segment = distributed_load * dx / 2
                    F[2 * n1 + 1] -= load_segment
                    F[2 * n2 + 1] -= load_segment

        return fixed_dofs, F

    def solve(
        self,
        point_load: float = 0.0,
        distributed_load: float = 0.0,
    ) -> FEMResult:
        """
        Solve the FEM problem and return results.

        Args:
            point_load: Point load at tip [N]
            distributed_load: Uniformly distributed load [N/m]

        Returns:
            FEMResult object with displacements, strains, stresses
        """
        # Assemble stiffness matrix
        if self._K_global is None:
            self._K_global = self._assemble_global_stiffness()

        # Get boundary conditions
        fixed_dofs, F = self.setup_cantilever_bc(point_load, distributed_load)

        # Apply boundary conditions (penalty method or reduction)
        K_mod = self._K_global.copy()
        n_dof = 2 * self.mesh.n_nodes

        # Reduction method
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

        K_ff = K_mod[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]

        # Solve reduced system
        U_f = spsolve(K_ff, F_f)

        # Reconstruct full displacement vector
        U = np.zeros(n_dof)
        U[free_dofs] = U_f

        # Reshape to (n_nodes, 2)
        displacements = U.reshape(-1, 2)

        # Compute strains and stresses at element centers
        strains = self._compute_element_strains(displacements)
        stresses = strains @ self.D.T  # σ = D * ε

        return FEMResult(
            displacements=displacements,
            strains=strains,
            stresses=stresses,
            mesh=self.mesh,
            fem=self,
        )

    def _compute_element_strains(self, displacements: np.ndarray) -> np.ndarray:
        """
        Compute strains at element centers.

        Args:
            displacements: Nodal displacements (n_nodes, 2)

        Returns:
            Element strains at centers (n_elements, 3)

        """
        strains = np.zeros((self.mesh.n_elements, 3))

        for elem_id in range(self.mesh.n_elements):
            elem_nodes = self.mesh.elements[elem_id]
            coords = self.mesh.nodes[elem_nodes]

            # Element displacements
            u_elem = displacements[elem_nodes].flatten()

            # Evaluate at element center (xi=0, eta=0)
            N, dN_dxi = self._q4_shape_functions(0, 0)

            # Jacobian
            J = dN_dxi.T @ coords
            J_inv = np.linalg.inv(J)
            dN_dx = dN_dxi @ J_inv

            # Strain-displacement matrix
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2 * k] = dN_dx[k, 0]
                B[1, 2 * k + 1] = dN_dx[k, 1]
                B[2, 2 * k] = dN_dx[k, 1]
                B[2, 2 * k + 1] = dN_dx[k, 0]

            strains[elem_id] = B @ u_elem

        return strains

    def extract_centerline_deflection(
        self,
        displacements: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract deflection along beam centerline (y=0).

        Args:
            displacements: Nodal displacements

        Returns:
            x_coords: Positions along beam
            w: Vertical deflections

        """
        # Find nodes along centerline
        centerline_mask = np.abs(self.mesh.nodes[:, 1]) < 1e-10
        centerline_nodes = np.where(centerline_mask)[0]

        # Sort by x-coordinate
        x_coords = self.mesh.nodes[centerline_nodes, 0]
        sort_idx = np.argsort(x_coords)
        centerline_nodes = centerline_nodes[sort_idx]
        x_coords = x_coords[sort_idx]

        # Extract vertical deflection
        w = displacements[centerline_nodes, 1]

        return x_coords, w

    def extract_surface_strain(
        self,
        strains: np.ndarray,
        surface: str = "top",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract axial strain along beam surface.

        Args:
            strains: Element strains
            surface: 'top' or 'bottom'

        Returns:
            x_coords: Positions along beam
            epsilon_xx: Axial strains

        """
        # Determine y threshold for surface elements
        if surface == "top":
            y_thresh = self.height / 4
        else:
            y_thresh = -self.height / 4

        # Find elements near surface
        surface_elements = []
        for elem_id in range(self.mesh.n_elements):
            elem_nodes = self.mesh.elements[elem_id]
            elem_y = np.mean(self.mesh.nodes[elem_nodes, 1])
            if (surface == "top" and elem_y > y_thresh) or \
               (surface == "bottom" and elem_y < y_thresh):
                surface_elements.append(elem_id)

        # Extract strains
        surface_elements = np.array(surface_elements)
        x_coords = np.zeros(len(surface_elements))
        for i, elem_id in enumerate(surface_elements):
            elem_nodes = self.mesh.elements[elem_id]
            x_coords[i] = np.mean(self.mesh.nodes[elem_nodes, 0])

        epsilon_xx = strains[surface_elements, 0]

        # Sort by x
        sort_idx = np.argsort(x_coords)
        return x_coords[sort_idx], epsilon_xx[sort_idx]
