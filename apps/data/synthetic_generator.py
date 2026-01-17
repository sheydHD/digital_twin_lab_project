"""
Synthetic Data Generator for Beam Model Selection Study.

This module generates synthetic measurement data from the high-fidelity FEM model,
simulating real sensor measurements with configurable noise levels.

The synthetic data serves as "ground truth" observations for:
- Calibrating Euler-Bernoulli and Timoshenko beam models
- Evaluating model selection criteria across different beam geometries
- Validating Bayesian inference procedures
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import json
import h5py

from apps.fem.cantilever_fem import CantileverFEM
from apps.models.base_beam import BeamGeometry, MaterialProperties, LoadCase


@dataclass
class SensorConfiguration:
    """
    Configuration for virtual sensors on the beam.

    Attributes:
        displacement_locations: x-coordinates for displacement sensors [m]
        strain_locations: x-coordinates for strain gauges [m]
        strain_y_positions: y-positions for strain gauges (from neutral axis) [m]
        sampling_frequency: Measurement sampling rate [Hz] (for dynamic)
    """

    displacement_locations: np.ndarray
    strain_locations: np.ndarray
    strain_y_positions: Optional[np.ndarray] = None
    sampling_frequency: float = 1000.0

    def __post_init__(self):
        """Set default strain gauge positions at beam surfaces."""
        if self.strain_y_positions is None:
            # Default: strain gauges at top and bottom surfaces
            self.strain_y_positions = np.array([1.0, -1.0])  # Normalized by h/2


@dataclass
class NoiseModel:
    """
    Noise model for synthetic measurements.

    Attributes:
        displacement_std: Standard deviation for displacement noise [m]
        strain_std: Standard deviation for strain noise [-]
        relative_noise: Whether noise is relative to signal magnitude
        seed: Random seed for reproducibility
    """

    displacement_std: float = 1e-6  # 1 μm
    strain_std: float = 1e-6  # 1 microstrain
    relative_noise: bool = False
    noise_fraction: float = 0.001 # 2% relative noise if relative_noise=True
    seed: Optional[int] = 42


@dataclass
class SyntheticDataset:
    """
    Container for synthetic measurement data.

    Attributes:
        geometry: Beam geometry used
        material: Material properties used
        load_case: Applied load case
        x_disp: Displacement sensor locations
        displacements: Measured displacements
        displacement_noise_std: Applied noise level for displacements
        x_strain: Strain sensor locations
        y_strain: Strain sensor y-positions
        strains: Measured strains
        strain_noise_std: Applied noise level for strains
        metadata: Additional information
    """

    geometry: BeamGeometry
    material: MaterialProperties
    load_case: LoadCase
    x_disp: np.ndarray
    displacements: np.ndarray
    displacement_noise_std: float
    x_strain: np.ndarray
    y_strain: np.ndarray
    strains: np.ndarray
    strain_noise_std: float
    metadata: Dict = field(default_factory=dict)


class SyntheticDataGenerator:
    """
    Generate synthetic measurement data from FEM reference model.

    This class provides methods to:
    1. Run high-fidelity FEM simulations for different beam configurations
    2. Extract measurements at specified sensor locations
    3. Add realistic measurement noise
    4. Save datasets for Bayesian calibration

    """

    def __init__(
        self,
        sensors: SensorConfiguration,
        noise: NoiseModel,
        fem_refinement: Tuple[int, int] = (40, 8),
    ):
        """
        Initialize the synthetic data generator.

        Args:
            sensors: Sensor configuration
            noise: Noise model parameters
            fem_refinement: (n_elements_x, n_elements_y) for FEM mesh
        """
        self.sensors = sensors
        self.noise = noise
        self.fem_nx, self.fem_ny = fem_refinement

        # Set random seed
        if noise.seed is not None:
            np.random.seed(noise.seed)

    def _get_n_beam_elements(self, geometry: BeamGeometry) -> int:
        """
        Get number of 1D beam elements based on aspect ratio.
        
        For 1D Timoshenko beam elements, we need enough elements
        to capture the deflection shape accurately.
        """
        aspect_ratio = geometry.aspect_ratio
        
        # Use ~4 elements per unit aspect ratio, minimum 20
        n_elem = max(20, int(4 * aspect_ratio))
        
        # Cap to prevent excessive computation (not really needed for 1D)
        n_elem = min(n_elem, 200)
        
        return n_elem

    def generate_static_dataset(
        self,
        geometry: BeamGeometry,
        material: MaterialProperties,
        load: LoadCase,
    ) -> SyntheticDataset:
        """
        Generate synthetic static measurement data using 1D Timoshenko beam FEM.
        
        IMPORTANT - Ground Truth Physics:
        ---------------------------------
        Uses 1D Timoshenko beam finite elements which exactly capture:
        - Bending deformation (same as Euler-Bernoulli)
        - Shear deformation (via shear correction factor κ)
        
        This ensures the ground truth is physically consistent with beam theory,
        avoiding the 2D plane stress vs 1D beam theory mismatch issues.
        
        Expected model selection behavior:
        - Thick beams (L/h < 10): Significant shear -> Timoshenko fits better
        - Slender beams (L/h > 20): Shear negligible -> EB preferred (simpler)

        Args:
            geometry: Beam geometry
            material: Material properties
            load: Load case (point load, distributed load, or moment)

        Returns:
            SyntheticDataset with measurements
        """
        from apps.fem.beam_fem import TimoshenkoBeamFEM
        
        # Get number of beam elements
        n_elem = self._get_n_beam_elements(geometry)
        
        # Create 1D Timoshenko beam FEM (includes shear deformation exactly)
        fem = TimoshenkoBeamFEM(
            length=geometry.length,
            height=geometry.height,
            width=geometry.width,
            elastic_modulus=material.elastic_modulus,
            poisson_ratio=material.poisson_ratio,
            shear_correction_factor=material.shear_correction_factor,
            n_elements=n_elem,
        )

        # Solve FEM problem
        result = fem.solve(
            point_load=load.point_load,
            distributed_load=load.distributed_load,
        )

        # Get deflections at sensor locations
        w_sensors = result.get_deflection_at(self.sensors.displacement_locations)

        # Add noise to displacements
        disp_noise_std = self._compute_noise_level(
            w_sensors, self.noise.displacement_std, is_displacement=True
        )
        w_noisy = w_sensors + np.random.normal(0, disp_noise_std, w_sensors.shape)

        # Compute strains from beam theory: ε = -y * d²w/dx² ≈ -y * M/(EI)
        # For cantilever with point load: M(x) = P*(L-x), so ε = -y*P*(L-x)/(EI)
        # At top surface y = h/2
        E = material.elastic_modulus
        I = geometry.moment_of_inertia
        P = load.point_load
        L = geometry.length
        y_surface = geometry.height / 2
        
        x_strain = self.sensors.strain_locations
        M_x = P * (L - x_strain)  # Moment at strain gauge locations
        eps_sensors = -y_surface * M_x / (E * I)  # Axial strain at top surface

        # Add noise to strains
        strain_noise_std = self._compute_noise_level(
            eps_sensors, self.noise.strain_std, is_displacement=False
        )
        eps_noisy = eps_sensors + np.random.normal(0, strain_noise_std, eps_sensors.shape)

        return SyntheticDataset(
            geometry=geometry,
            material=material,
            load_case=load,
            x_disp=self.sensors.displacement_locations.copy(),
            displacements=w_noisy,
            displacement_noise_std=disp_noise_std,
            x_strain=self.sensors.strain_locations.copy(),
            y_strain=np.array([geometry.height / 2]),  # Top surface
            strains=eps_noisy,
            strain_noise_std=strain_noise_std,
            metadata={
                "fem_elements": n_elem,
                "generation_method": "FEM_1D_Timoshenko",
                "aspect_ratio": geometry.aspect_ratio,
                "shear_correction_factor": material.shear_correction_factor,
            },
        )

    def _compute_noise_level(
        self,
        signal: np.ndarray,
        base_std: float,
        is_displacement: bool,
    ) -> float:
        """
        Compute noise standard deviation.

        Args:
            signal: Clean signal values
            base_std: Base noise standard deviation
            is_displacement: Whether this is displacement data

        Returns:
            Noise standard deviation to apply
        """
        if self.noise.relative_noise:
            signal_magnitude = np.max(np.abs(signal))
            return self.noise.noise_fraction * signal_magnitude
        return base_std

    def generate_parametric_study(
        self,
        aspect_ratios: List[float],
        base_length: float,
        base_material: MaterialProperties,
        base_load: LoadCase,
        width: float = 0.1,
    ) -> List[SyntheticDataset]:
        """
        Generate datasets for multiple beam aspect ratios.

        This is the core data generation for the model selection study,
        creating synthetic measurements for beams ranging from slender
        (Euler-Bernoulli valid) to thick (Timoshenko required).

        Args:
            aspect_ratios: List of L/h ratios to study
            base_length: Reference beam length [m]
            base_material: Material properties
            base_load: Load case
            width: Beam width [m]

        Returns:
            List of SyntheticDataset objects

        """
        datasets = []

        for L_h in aspect_ratios:
            # Compute height for given aspect ratio
            height = base_length / L_h

            # Create geometry
            geometry = BeamGeometry(
                length=base_length,
                height=height,
                width=width,
            )

            # Update sensor locations relative to beam length
            self._scale_sensors(base_length)

            # Generate dataset
            dataset = self.generate_static_dataset(
                geometry=geometry,
                material=base_material,
                load=base_load,
            )

            datasets.append(dataset)

            print(f"Generated dataset for L/h = {L_h:.1f}")

        return datasets

    def _scale_sensors(self, length: float) -> None:
        """
        Scale sensor locations to beam length.

        """
        # Default: sensors at 20%, 40%, 60%, 80%, 100% of length
        n_disp = len(self.sensors.displacement_locations)
        self.sensors.displacement_locations = np.linspace(
            0.2 * length, length, n_disp
        )

        n_strain = len(self.sensors.strain_locations)
        self.sensors.strain_locations = np.linspace(
            0.1 * length, 0.9 * length, n_strain
        )

    def generate_frequency_sweep(
        self,
        geometry: BeamGeometry,
        material: MaterialProperties,
        frequencies: np.ndarray,
        excitation_amplitude: float = 1.0,
    ) -> List[SyntheticDataset]:
        """
        Generate synthetic data for dynamic frequency response.

        For model selection at different frequencies, this method generates
        synthetic FRF (Frequency Response Function) data.

        Args:
            geometry: Beam geometry
            material: Material properties
            frequencies: Excitation frequencies [Hz]
            excitation_amplitude: Force amplitude [N]

        Returns:
            List of datasets at each frequency

        """
        # Placeholder for dynamic analysis
        # This requires either dynamic FEM (computationally expensive) or
        # analytical frequency response functions for both beam theories

        raise NotImplementedError(
            "Dynamic data generation not yet implemented. "
            "TODO: Task 13 - Implement frequency-domain analysis"
        )


def save_dataset(dataset: SyntheticDataset, filepath: Path) -> None:
    """
    Save synthetic dataset to HDF5 file.

    Args:
        dataset: Dataset to save
        filepath: Output file path

    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        # Geometry group
        geom = f.create_group("geometry")
        geom.attrs["length"] = dataset.geometry.length
        geom.attrs["height"] = dataset.geometry.height
        geom.attrs["width"] = dataset.geometry.width
        geom.attrs["aspect_ratio"] = dataset.geometry.aspect_ratio

        # Material group
        mat = f.create_group("material")
        mat.attrs["elastic_modulus"] = dataset.material.elastic_modulus
        mat.attrs["poisson_ratio"] = dataset.material.poisson_ratio
        mat.attrs["density"] = dataset.material.density

        # Load group
        load = f.create_group("load")
        load.attrs["point_load"] = dataset.load_case.point_load
        load.attrs["distributed_load"] = dataset.load_case.distributed_load

        # Displacement data
        disp = f.create_group("displacements")
        disp.create_dataset("x", data=dataset.x_disp)
        disp.create_dataset("values", data=dataset.displacements)
        disp.attrs["noise_std"] = dataset.displacement_noise_std

        # Strain data
        strain = f.create_group("strains")
        strain.create_dataset("x", data=dataset.x_strain)
        strain.create_dataset("y", data=dataset.y_strain)
        strain.create_dataset("values", data=dataset.strains)
        strain.attrs["noise_std"] = dataset.strain_noise_std

        # Metadata
        for key, value in dataset.metadata.items():
            f.attrs[key] = str(value) if not isinstance(value, (int, float)) else value


def load_dataset(filepath: Path) -> SyntheticDataset:
    """
    Load synthetic dataset from HDF5 file.

    Args:
        filepath: Input file path

    Returns:
        SyntheticDataset object

    """
    with h5py.File(filepath, "r") as f:
        # Load geometry
        geom = f["geometry"]
        geometry = BeamGeometry(
            length=geom.attrs["length"],
            height=geom.attrs["height"],
            width=geom.attrs["width"],
        )

        # Load material
        mat = f["material"]
        material = MaterialProperties(
            elastic_modulus=mat.attrs["elastic_modulus"],
            poisson_ratio=mat.attrs["poisson_ratio"],
            density=mat.attrs["density"],
        )

        # Load load case
        load_grp = f["load"]
        load_case = LoadCase(
            point_load=load_grp.attrs["point_load"],
            distributed_load=load_grp.attrs["distributed_load"],
        )

        # Load measurements
        disp = f["displacements"]
        strain = f["strains"]

        return SyntheticDataset(
            geometry=geometry,
            material=material,
            load_case=load_case,
            x_disp=disp["x"][:],
            displacements=disp["values"][:],
            displacement_noise_std=disp.attrs["noise_std"],
            x_strain=strain["x"][:],
            y_strain=strain["y"][:],
            strains=strain["values"][:],
            strain_noise_std=strain.attrs["noise_std"],
            metadata=dict(f.attrs),
        )
