"""
Base beam model abstract class.

Defines the interface for all beam theory implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BeamGeometry:
    """
    Beam geometry parameters.

    Attributes:
        length: Beam length [m]
        height: Cross-section height [m]
        width: Cross-section width [m]
        area: Cross-sectional area [m^2] (computed if not provided)
        moment_of_inertia: Second moment of area [m^4] (computed if not provided)
    """

    length: float
    height: float
    width: float
    area: Optional[float] = None
    moment_of_inertia: Optional[float] = None

    def __post_init__(self):
        """Compute derived geometric properties."""
        if self.area is None:
            self.area = self.width * self.height
        if self.moment_of_inertia is None:
            # Rectangular cross-section
            self.moment_of_inertia = (self.width * self.height**3) / 12

    @property
    def aspect_ratio(self) -> float:
        """Return the length-to-height ratio (L/h)."""
        return self.length / self.height


@dataclass
class MaterialProperties:
    """
    Material properties for beam analysis.

    Attributes:
        elastic_modulus: Young's modulus E [Pa]
        shear_modulus: Shear modulus G [Pa]
        poisson_ratio: Poisson's ratio ν [-]
        density: Material density ρ [kg/m^3]
        shear_correction_factor: Timoshenko shear correction factor κ [-]
    """

    elastic_modulus: float
    shear_modulus: Optional[float] = None
    poisson_ratio: float = 0.3
    density: float = 7850.0  # Steel default
    shear_correction_factor: float = 5 / 6  # Rectangular section default

    def __post_init__(self):
        """Compute shear modulus if not provided."""
        if self.shear_modulus is None:
            self.shear_modulus = self.elastic_modulus / (2 * (1 + self.poisson_ratio))


@dataclass
class LoadCase:
    """
    Load case definition for beam analysis.

    Attributes:
        point_load: Concentrated load at tip [N]
        distributed_load: Uniformly distributed load [N/m]
        moment: Applied moment at tip [N·m]
        frequency: Loading frequency for dynamic analysis [Hz]
    """

    point_load: float = 0.0
    distributed_load: float = 0.0
    moment: float = 0.0
    frequency: Optional[float] = None

    @property
    def is_dynamic(self) -> bool:
        """Check if load case is dynamic."""
        return self.frequency is not None and self.frequency > 0


class BaseBeamModel(ABC):
    """
    Abstract base class for beam theory models.

    This class defines the interface that all beam theory implementations
    (Euler-Bernoulli, Timoshenko) must follow.
    """

    def __init__(
        self,
        geometry: BeamGeometry,
        material: MaterialProperties,
        name: str = "BaseBeam",
    ):
        """
        Initialize the beam model.

        Args:
            geometry: Beam geometry parameters
            material: Material properties
            name: Model identifier name
        """
        self.geometry = geometry
        self.material = material
        self.name = name

    @abstractmethod
    def compute_deflection(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute beam deflection at specified positions.

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Deflection values at each position [m]
        """
        pass

    @abstractmethod
    def compute_rotation(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute beam rotation (slope) at specified positions.

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Rotation values at each position [rad]
        """
        pass

    @abstractmethod
    def compute_strain(
        self,
        x: np.ndarray,
        y: float,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute axial strain at specified positions and height.

        Args:
            x: Positions along the beam [m]
            y: Distance from neutral axis [m]
            load: Load case definition

        Returns:
            Strain values at each position [-]
        """
        pass

    @abstractmethod
    def compute_natural_frequencies(
        self,
        n_modes: int = 5,
    ) -> np.ndarray:
        """
        Compute natural frequencies of the cantilever beam.

        Args:
            n_modes: Number of vibration modes to compute

        Returns:
            Natural frequencies [Hz]
        """
        pass

    def compute_moment(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute bending moment along the beam.

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Bending moment at each position [N·m]
        """
        L = self.geometry.length
        P = load.point_load
        q = load.distributed_load
        M0 = load.moment

        # Cantilever beam: fixed at x=0, free at x=L
        # Moment from point load at tip
        M_point = P * (L - x)

        # Moment from distributed load
        M_dist = q * (L - x) ** 2 / 2

        # Applied moment at tip
        M_applied = np.where(x <= L, M0, 0.0)

        return M_point + M_dist + M_applied

    def compute_shear(
        self,
        x: np.ndarray,
        load: LoadCase,
    ) -> np.ndarray:
        """
        Compute shear force along the beam.

        Args:
            x: Positions along the beam [m]
            load: Load case definition

        Returns:
            Shear force at each position [N]
        """
        L = self.geometry.length
        P = load.point_load
        q = load.distributed_load

        # Shear from point load at tip
        V_point = np.where(x <= L, P, 0.0)

        # Shear from distributed load
        V_dist = q * (L - x)

        return V_point + V_dist

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"L={self.geometry.length:.3f}m, "
            f"h={self.geometry.height:.3f}m, "
            f"L/h={self.geometry.aspect_ratio:.1f}, "
            f"E={self.material.elastic_modulus/1e9:.1f}GPa)"
        )
