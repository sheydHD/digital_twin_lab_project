"""
Unit tests for the synthetic data generator module.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from apps.backend.core.data.synthetic_generator import (
    NoiseModel,
    SensorConfiguration,
    SyntheticDataGenerator,
    SyntheticDataset,
    save_dataset,
)
from apps.backend.core.models.base_beam import BeamGeometry, LoadCase, MaterialProperties


@pytest.fixture
def sensors() -> SensorConfiguration:
    return SensorConfiguration(
        displacement_locations=np.linspace(0.2, 1.0, 5),
        strain_locations=np.linspace(0.1, 0.9, 4),
    )


@pytest.fixture
def noise() -> NoiseModel:
    return NoiseModel(
        displacement_std=1e-6,
        strain_std=1e-6,
        relative_noise=True,
        noise_fraction=0.005,
        seed=42,
    )


@pytest.fixture
def generator(sensors: SensorConfiguration, noise: NoiseModel) -> SyntheticDataGenerator:
    return SyntheticDataGenerator(sensors=sensors, noise=noise)


class TestNoiseModel:
    def test_seed_reproducibility(self) -> None:
        """Same seed must produce identical samples."""
        n1 = NoiseModel(seed=123)
        n2 = NoiseModel(seed=123)
        assert n1.seed == n2.seed

    def test_default_values(self) -> None:
        n = NoiseModel()
        assert n.displacement_std >= 0
        assert n.strain_std >= 0


class TestSensorConfiguration:
    def test_locations_sorted(self, sensors: SensorConfiguration) -> None:
        assert np.all(np.diff(sensors.displacement_locations) > 0)

    def test_sensor_counts(self, sensors: SensorConfiguration) -> None:
        assert len(sensors.displacement_locations) == 5
        assert len(sensors.strain_locations) == 4


class TestSyntheticDataGenerator:
    def test_generate_single_dataset(self, generator: SyntheticDataGenerator) -> None:
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        load = LoadCase(point_load=1000.0)
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.1)

        dataset = generator.generate_static_dataset(
            geometry=geometry,
            material=material,
            load=load,
        )

        assert isinstance(dataset, SyntheticDataset)
        assert len(dataset.displacements) == 5
        assert dataset.geometry.aspect_ratio == pytest.approx(10.0)

    def test_parametric_study_shapes(self, generator: SyntheticDataGenerator) -> None:
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        load = LoadCase(point_load=1000.0)
        ratios = [5.0, 10.0, 20.0]

        datasets = generator.generate_parametric_study(
            aspect_ratios=ratios,
            base_length=1.0,
            base_material=material,
            base_load=load,
            width=0.1,
        )

        assert len(datasets) == 3
        for ds, ratio in zip(datasets, ratios, strict=True):
            assert ds.geometry.aspect_ratio == pytest.approx(ratio, rel=1e-3)

    def test_invalid_aspect_ratio(self, generator: SyntheticDataGenerator) -> None:
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        load = LoadCase(point_load=1000.0)

        with pytest.raises(ValueError, match="positive"):
            generator.generate_parametric_study(
                aspect_ratios=[-1.0],
                base_length=1.0,
                base_material=material,
                base_load=load,
                width=0.1,
            )


class TestSaveDataset:
    def test_hdf5_roundtrip(self, generator: SyntheticDataGenerator) -> None:
        material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
        load = LoadCase(point_load=1000.0)
        geometry = BeamGeometry(length=1.0, height=0.1, width=0.1)

        dataset = generator.generate_static_dataset(
            geometry=geometry,
            material=material,
            load=load,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            save_dataset(dataset, path)
            assert path.exists()

            with h5py.File(path, "r") as f:
                assert "displacements" in f
                assert "geometry" in f
                assert f["geometry"].attrs["aspect_ratio"] == pytest.approx(10.0)
