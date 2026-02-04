"""
1D Timoshenko Beam Finite Element Model.

This module provides a 1D beam FEM implementation using Timoshenko beam elements
that naturally include shear deformation. This serves as a more appropriate
"ground truth" for comparing Euler-Bernoulli vs Timoshenko beam theories.

Key advantages over 2D plane stress FEM:
- Exact match with Timoshenko beam theory for uniform beams
- No constraint effects at boundaries that cause stiffness mismatch
- Shear deformation properly captured via shear correction factor
- Much faster computation
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BeamFEMResult:
    """
    Result container for 1D beam FEM analysis.

    Attributes:
        x: Node positions along the beam [m]
        deflections: Transverse displacements at nodes [m]
        rotations: Cross-section rotations at nodes [rad]
        n_elements: Number of elements used
    """
    x: np.ndarray
    deflections: np.ndarray
    rotations: np.ndarray
    n_elements: int

    @property
    def tip_deflection(self) -> float:
        """Get deflection at beam tip."""
        return self.deflections[-1]

    def get_deflection_at(self, x_points: np.ndarray) -> np.ndarray:
        """Interpolate deflections to arbitrary x positions."""
        return np.interp(x_points, self.x, self.deflections)


class TimoshenkoBeamFEM:
    """
    1D Timoshenko Beam Finite Element Model.

    Uses 2-node Timoshenko beam elements with:
    - Linear interpolation for deflection w
    - Linear interpolation for rotation θ
    - Reduced integration to avoid shear locking

    Each node has 2 DOFs: (w, θ)
    """

    def __init__(
        self,
        length: float,
        height: float,
        width: float,
        elastic_modulus: float,
        poisson_ratio: float,
        shear_correction_factor: float = 5/6,
        n_elements: int = 20,
    ):
        """
        Initialize 1D Timoshenko beam FEM.

        Args:
            length: Beam length [m]
            height: Cross-section height [m]
            width: Cross-section width (thickness) [m]
            elastic_modulus: Young's modulus E [Pa]
            poisson_ratio: Poisson's ratio ν [-]
            shear_correction_factor: Timoshenko shear factor κ [-]
            n_elements: Number of beam elements
        """
        self.L = length
        self.h = height
        self.b = width
        self.E = elastic_modulus
        self.nu = poisson_ratio
        self.kappa = shear_correction_factor
        self.n_elem = n_elements

        # Derived properties
        self.A = width * height  # Cross-sectional area
        self.I = width * height**3 / 12  # Second moment of area
        self.G = elastic_modulus / (2 * (1 + poisson_ratio))  # Shear modulus

        # Mesh
        self.n_nodes = n_elements + 1
        self.x_nodes = np.linspace(0, length, self.n_nodes)
        self.Le = length / n_elements  # Element length

        # Precompute element stiffness
        self._Ke = self._element_stiffness()

    def _element_stiffness(self) -> np.ndarray:
        """
        Compute Timoshenko beam element stiffness matrix (4x4).

        The element has 4 DOFs: [w1, θ1, w2, θ2]

        Uses exact integration for the Timoshenko beam element.
        """
        E, I, G, A, kappa = self.E, self.I, self.G, self.A, self.kappa
        L = self.Le

        # Timoshenko shear parameter
        phi = 12 * E * I / (kappa * G * A * L**2)

        # Stiffness matrix for Timoshenko beam element
        # Reference: Cook et al., "Concepts and Applications of FEA"
        coeff = E * I / (L**3 * (1 + phi))

        Ke = coeff * np.array([
            [12,           6*L,          -12,          6*L],
            [6*L,          (4+phi)*L**2, -6*L,         (2-phi)*L**2],
            [-12,          -6*L,         12,           -6*L],
            [6*L,          (2-phi)*L**2, -6*L,         (4+phi)*L**2]
        ])

        return Ke

    def _assemble_global_stiffness(self) -> np.ndarray:
        """Assemble global stiffness matrix."""
        n_dof = 2 * self.n_nodes  # 2 DOFs per node
        K = np.zeros((n_dof, n_dof))

        for e in range(self.n_elem):
            # Global DOF indices for element e
            # Node e has DOFs [2*e, 2*e+1], Node e+1 has DOFs [2*(e+1), 2*(e+1)+1]
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]

            # Add element contribution
            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += self._Ke[i, j]

        return K

    def solve(
        self,
        point_load: float = 0.0,
        distributed_load: float = 0.0,
    ) -> BeamFEMResult:
        """
        Solve the beam FEM problem for cantilever configuration.

        Boundary conditions:
        - Fixed at x=0: w(0) = 0, θ(0) = 0
        - Free at x=L with applied loads

        Args:
            point_load: Concentrated load at tip [N], positive downward
            distributed_load: Uniformly distributed load [N/m], positive downward

        Returns:
            BeamFEMResult with deflections and rotations
        """
        n_dof = 2 * self.n_nodes

        # Assemble stiffness
        K = self._assemble_global_stiffness()

        # Force vector
        F = np.zeros(n_dof)

        # Apply point load at tip (last node, w DOF)
        if point_load != 0:
            # Positive load causes negative (downward) deflection
            F[2*(self.n_nodes-1)] = -point_load

        # Apply distributed load (consistent nodal loads)
        if distributed_load != 0:
            q = distributed_load
            Le = self.Le
            # Consistent nodal loads for uniform distributed load on beam element
            # For each element: F_w1 = qL/2, F_θ1 = qL²/12, F_w2 = qL/2, F_θ2 = -qL²/12
            for e in range(self.n_elem):
                dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
                F[dofs[0]] -= q * Le / 2
                F[dofs[1]] -= q * Le**2 / 12
                F[dofs[2]] -= q * Le / 2
                F[dofs[3]] += q * Le**2 / 12

        # Apply boundary conditions (fixed at x=0)
        # DOFs 0 and 1 are fixed (w=0, θ=0 at first node)
        free_dofs = list(range(2, n_dof))

        # Solve reduced system
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]

        U_f = np.linalg.solve(K_ff, F_f)

        # Reconstruct full solution
        U = np.zeros(n_dof)
        U[free_dofs] = U_f

        # Extract deflections and rotations
        deflections = U[0::2]  # Even indices: w
        rotations = U[1::2]    # Odd indices: θ

        return BeamFEMResult(
            x=self.x_nodes.copy(),
            deflections=deflections,
            rotations=rotations,
            n_elements=self.n_elem,
        )


class EulerBernoulliFEM:
    """
    1D Euler-Bernoulli Beam Finite Element Model.

    Uses cubic Hermite beam elements (no shear deformation).
    This serves as reference to verify EB analytical solution.
    """

    def __init__(
        self,
        length: float,
        height: float,
        width: float,
        elastic_modulus: float,
        n_elements: int = 20,
    ):
        """Initialize EB beam FEM."""
        self.L = length
        self.h = height
        self.b = width
        self.E = elastic_modulus
        self.n_elem = n_elements

        self.I = width * height**3 / 12
        self.n_nodes = n_elements + 1
        self.x_nodes = np.linspace(0, length, self.n_nodes)
        self.Le = length / n_elements

        self._Ke = self._element_stiffness()

    def _element_stiffness(self) -> np.ndarray:
        """Compute Euler-Bernoulli beam element stiffness (4x4)."""
        E, I, L = self.E, self.I, self.Le

        coeff = E * I / L**3
        Ke = coeff * np.array([
            [12,    6*L,   -12,   6*L],
            [6*L,   4*L**2, -6*L,  2*L**2],
            [-12,   -6*L,  12,    -6*L],
            [6*L,   2*L**2, -6*L,  4*L**2]
        ])

        return Ke

    def _assemble_global_stiffness(self) -> np.ndarray:
        """Assemble global stiffness matrix."""
        n_dof = 2 * self.n_nodes
        K = np.zeros((n_dof, n_dof))

        for e in range(self.n_elem):
            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            for i in range(4):
                for j in range(4):
                    K[dofs[i], dofs[j]] += self._Ke[i, j]

        return K

    def solve(self, point_load: float = 0.0) -> BeamFEMResult:
        """Solve cantilever with point load at tip."""
        n_dof = 2 * self.n_nodes
        K = self._assemble_global_stiffness()

        F = np.zeros(n_dof)
        if point_load != 0:
            F[2*(self.n_nodes-1)] = -point_load

        free_dofs = list(range(2, n_dof))

        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]

        U_f = np.linalg.solve(K_ff, F_f)

        U = np.zeros(n_dof)
        U[free_dofs] = U_f

        deflections = U[0::2]
        rotations = U[1::2]

        return BeamFEMResult(
            x=self.x_nodes.copy(),
            deflections=deflections,
            rotations=rotations,
            n_elements=self.n_elem,
        )
