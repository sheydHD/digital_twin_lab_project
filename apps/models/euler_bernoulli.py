"""
Euler-Bernoulli Beam Theory Implementation.

The Euler-Bernoulli beam theory assumes:
- Plane sections remain plane and perpendicular to the neutral axis (no shear deformation)
- Small deflections and rotations
- Linear elastic material behavior
- Slender beam geometry (L/h >> 1)

This theory is accurate for long, slender beams where shear deformation is negligible.
"""

import numpy as np
from scipy.optimize import brentq

from .base_beam import BaseBeamModel, BeamGeometry, LoadCase, MaterialProperties


class EulerBernoulliBeam(BaseBeamModel):
    """
    Euler-Bernoulli beam model for cantilever beam analysis.

    Governing equation: EI * d⁴w/dx⁴ = q(x)

    Where:
        E = Elastic modulus
        I = Second moment of area
        w = Transverse deflection
        q = Distributed load
    """

    def __init__(
        self,
        geometry: BeamGeometry,
        material: MaterialProperties,
    ):
        """
        Initialize Euler-Bernoulli beam model.

        Args:
            geometry: Beam geometry parameters
            material: Material properties
        """
        super().__init__(geometry, material, name="Euler-Bernoulli")

    @property
    def flexural_rigidity(self) -> float:
        """Return the flexural rigidity EI [N·m²]."""
        return self.material.elastic_modulus * self.geometry.moment_of_inertia

    def compute_deflection(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute beam deflection using Euler-Bernoulli theory.

        For a cantilever beam (fixed at x=0, free at x=L):
        - Point load P at tip: w(x) = (Px²/6EI)(3L - x)
        - Distributed load q: w(x) = (qx²/24EI)(6L² - 4Lx + x²)
        - Moment M at tip: w(x) = (Mx²/2EI)

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Deflection values at each position [m]

        TODO: Task 1.1 - Verify deflection formulas against analytical solutions
        TODO: Task 1.2 - Add validation for input ranges
        TODO: Task 1.3 - Implement superposition for combined loading
        """
        x = np.asarray(x)
        L = self.geometry.length
        EI = self.flexural_rigidity

        # Initialize deflection array
        w = np.zeros_like(x, dtype=float)

        # Point load at tip contribution
        if load.point_load != 0:
            P = load.point_load
            # TODO: Task 1.4 - Implement point load deflection formula
            # Hint: w_P = (P * x**2 / (6 * EI)) * (3 * L - x)
            w_point = (P * x**2 / (6 * EI)) * (3 * L - x)
            w += w_point

        # Distributed load contribution
        if load.distributed_load != 0:
            q = load.distributed_load
            # TODO: Task 1.5 - Implement distributed load deflection formula
            # Hint: w_q = (q * x**2 / (24 * EI)) * (6 * L**2 - 4 * L * x + x**2)
            w_dist = (q * x**2 / (24 * EI)) * (6 * L**2 - 4 * L * x + x**2)
            w += w_dist

        # Applied moment contribution
        if load.moment != 0:
            M0 = load.moment
            # TODO: Task 1.6 - Implement moment deflection formula
            # Hint: w_M = (M0 * x**2) / (2 * EI)
            w_moment = (M0 * x**2) / (2 * EI)
            w += w_moment

        return w

    def compute_rotation(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute beam rotation (slope) using Euler-Bernoulli theory.

        Rotation θ = dw/dx

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Rotation values at each position [rad]

        TODO: Task 2.1 - Derive rotation formulas from deflection
        TODO: Task 2.2 - Validate rotation at boundaries (θ(0) = 0 for cantilever)
        """
        x = np.asarray(x)
        L = self.geometry.length
        EI = self.flexural_rigidity

        # Initialize rotation array
        theta = np.zeros_like(x, dtype=float)

        # Point load contribution
        if load.point_load != 0:
            P = load.point_load
            # TODO: Task 2.3 - Implement point load rotation formula
            # Hint: θ_P = (P * x / (2 * EI)) * (2 * L - x)
            theta_point = (P * x / (2 * EI)) * (2 * L - x)
            theta += theta_point

        # Distributed load contribution
        if load.distributed_load != 0:
            q = load.distributed_load
            # TODO: Task 2.4 - Implement distributed load rotation formula
            # Hint: θ_q = (q * x / (6 * EI)) * (3 * L**2 - 3 * L * x + x**2)
            theta_dist = (q * x / (6 * EI)) * (3 * L**2 - 3 * L * x + x**2)
            theta += theta_dist

        # Applied moment contribution
        if load.moment != 0:
            M0 = load.moment
            # TODO: Task 2.5 - Implement moment rotation formula
            theta_moment = (M0 * x) / EI
            theta += theta_moment

        return theta

    def compute_strain(
        self,
        x: np.ndarray,
        y: float,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute axial strain using Euler-Bernoulli theory.

        Strain ε = -y * d²w/dx² = -y * M(x) / EI = M(x) * y / (E * I)

        Args:
            x: Positions along the beam [m]
            y: Distance from neutral axis [m] (positive downward)
            load: Load case definition

        Returns:
            Strain values at each position [-]

        TODO: Task 3.1 - Verify strain distribution across beam height
        TODO: Task 3.2 - Add strain measurement locations for sensor placement
        """
        x = np.asarray(x)
        E = self.material.elastic_modulus
        I = self.geometry.moment_of_inertia

        # Compute bending moment
        M = self.compute_moment(x, load)

        # Axial strain from bending: ε = -y * κ = -y * M / (EI)
        # Note: Positive y is below neutral axis, positive M causes compression above
        # TODO: Task 3.3 - Implement strain formula
        strain = -y * M / (E * I)

        return strain

    def compute_natural_frequencies(
        self,
        n_modes: int = 5,
    ) -> np.ndarray:
        """
        Compute natural frequencies of cantilever beam using Euler-Bernoulli theory.

        For a cantilever beam, the characteristic equation is:
        cos(βL) * cosh(βL) + 1 = 0

        Natural frequency: ω_n = β_n² * sqrt(EI / (ρA))

        Args:
            n_modes: Number of vibration modes to compute

        Returns:
            Natural frequencies [Hz]

        TODO: Task 4.1 - Solve transcendental equation for eigenvalues
        TODO: Task 4.2 - Validate against known cantilever frequencies
        TODO: Task 4.3 - Compare with Timoshenko frequencies at high modes
        """
        E = self.material.elastic_modulus
        I = self.geometry.moment_of_inertia
        rho = self.material.density
        A = self.geometry.area
        L = self.geometry.length

        # Known eigenvalues βn*L for cantilever beam (first 10 modes)
        # These satisfy: cos(βL)*cosh(βL) + 1 = 0
        beta_L_approx = [1.8751, 4.6941, 7.8548, 10.9955, 14.1372,
                        17.2788, 20.4204, 23.5619, 26.7035, 29.8451]

        # TODO: Task 4.4 - Implement numerical solution for eigenvalues
        # For now, use approximate values
        frequencies = np.zeros(n_modes)

        for i in range(min(n_modes, len(beta_L_approx))):
            beta_n = beta_L_approx[i] / L
            # ω_n = β_n² * sqrt(EI / (ρA))
            omega_n = beta_n**2 * np.sqrt(E * I / (rho * A))
            # Convert to Hz: f = ω / (2π)
            frequencies[i] = omega_n / (2 * np.pi)

        return frequencies

    def compute_curvature(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute beam curvature κ = M(x) / EI.

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Curvature at each position [1/m]
        """
        M = self.compute_moment(x, load)
        return M / self.flexural_rigidity

    def tip_deflection(self, load: LoadCase) -> float:
        """
        Compute deflection at beam tip (x = L).

        Args:
            load: Load case definition

        Returns:
            Tip deflection [m]
        """
        L = self.geometry.length
        EI = self.flexural_rigidity

        w_tip = 0.0

        if load.point_load != 0:
            w_tip += load.point_load * L**3 / (3 * EI)

        if load.distributed_load != 0:
            w_tip += load.distributed_load * L**4 / (8 * EI)

        if load.moment != 0:
            w_tip += load.moment * L**2 / (2 * EI)

        return w_tip

    def tip_rotation(self, load: LoadCase) -> float:
        """
        Compute rotation at beam tip (x = L).

        Args:
            load: Load case definition

        Returns:
            Tip rotation [rad]
        """
        L = self.geometry.length
        EI = self.flexural_rigidity

        theta_tip = 0.0

        if load.point_load != 0:
            theta_tip += load.point_load * L**2 / (2 * EI)

        if load.distributed_load != 0:
            theta_tip += load.distributed_load * L**3 / (6 * EI)

        if load.moment != 0:
            theta_tip += load.moment * L / EI

        return theta_tip
