"""
Timoshenko Beam Theory Implementation.

The Timoshenko beam theory accounts for:
- Shear deformation effects
- Rotary inertia (for dynamic analysis)
- Better accuracy for thick beams and high frequencies

Key differences from Euler-Bernoulli:
- Cross-sections remain plane but NOT perpendicular to the deformed neutral axis
- Additional shear correction factor κ accounts for non-uniform shear distribution
- Includes two primary unknowns: deflection w(x) and rotation ψ(x)

The theory is more accurate for:
- Short, thick beams (L/h < 10)
- Higher vibration modes
- Sandwich and composite beams
"""

import numpy as np

from .base_beam import BaseBeamModel, BeamGeometry, LoadCase, MaterialProperties


class TimoshenkoBeam(BaseBeamModel):
    """
    Timoshenko beam model for cantilever beam analysis.

    Governing equations:
        GA*κ*(dw/dx - ψ) = V (shear equilibrium)
        EI * dψ/dx = M (moment-curvature)

    Where:
        G = Shear modulus
        A = Cross-sectional area
        κ = Shear correction factor
        ψ = Rotation of cross-section (independent of dw/dx)
    """

    def __init__(
        self,
        geometry: BeamGeometry,
        material: MaterialProperties,
    ):
        """
        Initialize Timoshenko beam model.

        Args:
            geometry: Beam geometry parameters
            material: Material properties (including shear_correction_factor)
        """
        super().__init__(geometry, material, name="Timoshenko")

    @property
    def flexural_rigidity(self) -> float:
        """Return the flexural rigidity EI [N·m²]."""
        return self.material.elastic_modulus * self.geometry.moment_of_inertia

    @property
    def shear_rigidity(self) -> float:
        """Return the shear rigidity κGA [N]."""
        kappa = self.material.shear_correction_factor
        G = self.material.shear_modulus
        A = self.geometry.area
        return kappa * G * A

    @property
    def shear_parameter(self) -> float:
        """
        Return the dimensionless shear parameter Φ.

        Φ = 12*EI / (κ*G*A*L²) = 12*E*I / (κ*G*A*L²)

        This parameter characterizes the relative importance of shear deformation.
        When Φ → 0, Timoshenko solution approaches Euler-Bernoulli.
        """
        EI = self.flexural_rigidity
        kGA = self.shear_rigidity
        L = self.geometry.length
        return 12 * EI / (kGA * L**2)

    def compute_deflection(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute beam deflection using Timoshenko theory.

        The total deflection includes both bending and shear contributions:
        w_total = w_bending + w_shear

        For a cantilever beam with point load P at tip:
        w(x) = (Px²/6EI)(3L - x) + (Px)/(κGA)

        The additional term Px/(κGA) accounts for shear deformation.

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Deflection values at each position [m]

        TODO: Task 5.1 - Implement Timoshenko deflection with shear correction
        TODO: Task 5.2 - Verify additional shear deflection term
        TODO: Task 5.3 - Compare with Euler-Bernoulli for slender beams
        """
        x = np.asarray(x)
        L = self.geometry.length
        EI = self.flexural_rigidity
        kGA = self.shear_rigidity

        # Initialize deflection array
        w = np.zeros_like(x, dtype=float)

        # Point load at tip contribution
        if load.point_load != 0:
            P = load.point_load
            # Bending contribution (same as Euler-Bernoulli)
            w_bending = (P * x**2 / (6 * EI)) * (3 * L - x)

            # TODO: Task 5.4 - Implement shear contribution for point load
            # Shear contribution: additional deflection due to shear deformation
            # Hint: w_shear = P * x / (κGA)
            w_shear = P * x / kGA

            w += w_bending + w_shear

        # Distributed load contribution
        if load.distributed_load != 0:
            q = load.distributed_load
            # Bending contribution (same as Euler-Bernoulli)
            w_bending = (q * x**2 / (24 * EI)) * (6 * L**2 - 4 * L * x + x**2)

            # TODO: Task 5.5 - Implement shear contribution for distributed load
            # Hint: w_shear = q * x * (L - x/2) / (κGA)
            w_shear = q * x * (L - x / 2) / kGA

            w += w_bending + w_shear

        # Applied moment contribution (no shear force, so no additional shear deflection)
        if load.moment != 0:
            M0 = load.moment
            # Moment does not produce shear force, so only bending contribution
            w_moment = (M0 * x**2) / (2 * EI)
            w += w_moment

        return w

    def compute_rotation(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute cross-section rotation ψ using Timoshenko theory.

        In Timoshenko theory, the rotation ψ is NOT equal to dw/dx.
        The relationship is: dw/dx = ψ + γ, where γ = V/(κGA) is the shear strain.

        The rotation ψ is related to the moment: ψ = ∫M/(EI)dx

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Cross-section rotation values at each position [rad]

        TODO: Task 6.1 - Implement Timoshenko rotation (ψ ≠ dw/dx)
        TODO: Task 6.2 - Compute shear angle γ separately
        """
        x = np.asarray(x)
        L = self.geometry.length
        EI = self.flexural_rigidity

        # Initialize rotation array
        psi = np.zeros_like(x, dtype=float)

        # Point load contribution
        if load.point_load != 0:
            P = load.point_load
            # TODO: Task 6.3 - Implement rotation for point load
            # In Timoshenko theory, ψ comes from integrating M/EI
            # For cantilever with tip load: ψ(x) = (P*x/(2EI))*(2L - x)
            # This is the same as Euler-Bernoulli because M(x) is the same
            psi_point = (P * x / (2 * EI)) * (2 * L - x)
            psi += psi_point

        # Distributed load contribution
        if load.distributed_load != 0:
            q = load.distributed_load
            # TODO: Task 6.4 - Implement rotation for distributed load
            psi_dist = (q * x / (6 * EI)) * (3 * L**2 - 3 * L * x + x**2)
            psi += psi_dist

        # Applied moment contribution
        if load.moment != 0:
            M0 = load.moment
            psi_moment = (M0 * x) / EI
            psi += psi_moment

        return psi

    def compute_shear_angle(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute shear angle γ = V/(κGA).

        This represents the additional rotation due to shear deformation.
        In Timoshenko theory: dw/dx = ψ + γ

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Shear angle at each position [rad]

        TODO: Task 6.5 - Implement shear angle computation
        """
        V = self.compute_shear(x, load)
        kGA = self.shear_rigidity
        return V / kGA

    def compute_strain(
        self,
        x: np.ndarray,
        y: float,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute axial strain using Timoshenko theory.

        In Timoshenko theory, the strain is: ε = -y * dψ/dx = -y * M/(EI)

        Note: This is the same relationship as Euler-Bernoulli for static loading,
        because the moment distribution is identical.

        Args:
            x: Positions along the beam [m]
            y: Distance from neutral axis [m] (positive downward)
            load: Load case definition

        Returns:
            Strain values at each position [-]

        TODO: Task 7.1 - Verify strain formulation for Timoshenko
        TODO: Task 7.2 - Consider shear strain component for completeness
        """
        x = np.asarray(x)
        E = self.material.elastic_modulus
        I = self.geometry.moment_of_inertia

        # Bending moment
        M = self.compute_moment(x, load)

        # Axial strain from bending (same as Euler-Bernoulli for static case)
        strain = -y * M / (E * I)

        return strain

    def compute_shear_strain(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute shear strain γ_xy in Timoshenko beam.

        For Timoshenko beam, the average shear strain is: γ = V / (κGA)

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Shear strain at each position [-]

        TODO: Task 7.3 - Implement shear strain computation
        """
        V = self.compute_shear(x, load)
        kGA = self.shear_rigidity
        return V / kGA

    def compute_natural_frequencies(
        self,
        n_modes: int = 5,
    ) -> np.ndarray:
        """
        Compute natural frequencies using Timoshenko beam theory.

        Timoshenko frequencies are lower than Euler-Bernoulli frequencies,
        especially for higher modes and shorter beams, due to:
        1. Shear deformation flexibility
        2. Rotary inertia effects

        The frequency equation for Timoshenko beam is more complex and involves
        the shear parameter and slenderness ratio.

        Args:
            n_modes: Number of vibration modes to compute

        Returns:
            Natural frequencies [Hz]

        TODO: Task 8.1 - Implement Timoshenko frequency equation
        TODO: Task 8.2 - Account for rotary inertia effects
        TODO: Task 8.3 - Compare frequency reduction vs Euler-Bernoulli
        TODO: Task 8.4 - Validate against published results
        """
        E = self.material.elastic_modulus
        I = self.geometry.moment_of_inertia
        G = self.material.shear_modulus
        rho = self.material.density
        A = self.geometry.area
        L = self.geometry.length
        kappa = self.material.shear_correction_factor

        # Euler-Bernoulli eigenvalues for reference
        beta_L_eb = [1.8751, 4.6941, 7.8548, 10.9955, 14.1372,
                    17.2788, 20.4204, 23.5619, 26.7035, 29.8451]

        # Slenderness ratio
        r = np.sqrt(I / A)  # radius of gyration
        slenderness = L / r

        # Shear correction parameters
        # s² = EI / (κGAL²) - represents shear deformation importance
        s_squared = E * I / (kappa * G * A * L**2)

        # r² = I / (AL²) - represents rotary inertia importance
        r_squared = I / (A * L**2)

        frequencies = np.zeros(n_modes)

        for i in range(min(n_modes, len(beta_L_eb))):
            beta_L = beta_L_eb[i]

            # TODO: Task 8.5 - Implement Timoshenko frequency correction
            # Approximate correction factor for Timoshenko beam
            # The correction becomes significant for β*L values and low L/h ratios
            #
            # Correction factor: ω_T / ω_EB ≈ 1 / sqrt(1 + β²*L²*(s² + r²))
            # For detailed implementation, see referenced paper

            correction_factor = 1.0 / np.sqrt(
                1 + beta_L**2 * (s_squared + r_squared)
            )

            # Euler-Bernoulli frequency
            beta_n = beta_L / L
            omega_eb = beta_n**2 * np.sqrt(E * I / (rho * A))

            # Apply Timoshenko correction
            omega_t = omega_eb * correction_factor

            frequencies[i] = omega_t / (2 * np.pi)

        return frequencies

    def tip_deflection(self, load: LoadCase) -> float:
        """
        Compute deflection at beam tip (x = L) including shear effects.

        Args:
            load: Load case definition

        Returns:
            Tip deflection [m]
        """
        L = self.geometry.length
        EI = self.flexural_rigidity
        kGA = self.shear_rigidity

        w_tip = 0.0

        if load.point_load != 0:
            P = load.point_load
            # Bending + Shear contributions
            w_tip += P * L**3 / (3 * EI) + P * L / kGA

        if load.distributed_load != 0:
            q = load.distributed_load
            # Bending + Shear contributions
            w_tip += q * L**4 / (8 * EI) + q * L**2 / (2 * kGA)

        if load.moment != 0:
            w_tip += load.moment * L**2 / (2 * EI)

        return w_tip

    def shear_deformation_ratio(self, load: LoadCase) -> float:
        """
        Compute the ratio of shear deflection to total deflection at tip.

        This ratio indicates the significance of shear deformation.
        Higher ratios suggest Timoshenko theory is more appropriate.

        Args:
            load: Load case definition

        Returns:
            Ratio of shear deflection to total deflection [-]

        TODO: Task 9.1 - Use this metric for model selection criteria
        """
        L = self.geometry.length
        EI = self.flexural_rigidity
        kGA = self.shear_rigidity

        if load.point_load != 0:
            w_bend = load.point_load * L**3 / (3 * EI)
            w_shear = load.point_load * L / kGA
            return w_shear / (w_bend + w_shear)

        if load.distributed_load != 0:
            w_bend = load.distributed_load * L**4 / (8 * EI)
            w_shear = load.distributed_load * L**2 / (2 * kGA)
            return w_shear / (w_bend + w_shear)

        return 0.0  # Pure moment loading has no shear deformation

    def compare_with_euler_bernoulli(self, load: LoadCase) -> dict:
        """
        Compare Timoshenko predictions with Euler-Bernoulli.

        Returns:
            Dictionary with comparison metrics
        """
        from .euler_bernoulli import EulerBernoulliBeam

        eb_beam = EulerBernoulliBeam(self.geometry, self.material)

        x = np.linspace(0, self.geometry.length, 100)

        w_eb = eb_beam.compute_deflection(x, load)
        w_t = self.compute_deflection(x, load)

        tip_eb = eb_beam.tip_deflection(load)
        tip_t = self.tip_deflection(load)

        freq_eb = eb_beam.compute_natural_frequencies(5)
        freq_t = self.compute_natural_frequencies(5)

        return {
            "tip_deflection_eb": tip_eb,
            "tip_deflection_timoshenko": tip_t,
            "deflection_ratio": tip_t / tip_eb if tip_eb != 0 else 1.0,
            "shear_deformation_contribution": self.shear_deformation_ratio(load),
            "frequency_ratios": freq_t / freq_eb,
            "aspect_ratio": self.geometry.aspect_ratio,
            "shear_parameter": self.shear_parameter,
        }
