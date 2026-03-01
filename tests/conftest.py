"""Shared pytest fixtures for the test suite.

Import any of these in a test file by simply declaring the fixture name as a
parameter — pytest discovers them automatically from this conftest.py.

Fixture overview
----------------
standard_geometry   — canonical steel cantilever beam geometry (L/h = 10)
standard_material   — standard structural steel properties
standard_load       — 1 kN point load at the free end
minimal_config      — smallest viable pipeline config (2 aspect ratios, few
                      samples) used for fast orchestrator/integration tests
"""

from __future__ import annotations

import pytest

from apps.backend.core.models.base_beam import BeamGeometry, LoadCase, MaterialProperties
from apps.backend.core.utils.config import load_config

# ── Domain primitives ────────────────────────────────────────────────────────


@pytest.fixture
def standard_geometry() -> BeamGeometry:
    """Canonical slender beam: L=1 m, h=0.1 m, b=0.05 m  →  L/h = 10."""
    return BeamGeometry(length=1.0, height=0.1, width=0.05)


@pytest.fixture
def standard_material() -> MaterialProperties:
    """Standard structural steel: E=210 GPa, ν=0.3."""
    return MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)


@pytest.fixture
def standard_load() -> LoadCase:
    """1 kN point load at the free end; no distributed load."""
    return LoadCase(point_load=1000.0)


# ── Pipeline / orchestrator ───────────────────────────────────────────────────


@pytest.fixture
def minimal_config() -> dict:
    """
    Minimal pipeline configuration for fast integration tests.

    Uses only 2 aspect ratios and greatly reduced MCMC parameters so the
    full orchestrator can be tested without waiting minutes.
    """
    cfg = load_config("configs/default_config.yaml")
    cfg["beam_parameters"]["aspect_ratios"] = [10.0, 20.0]
    cfg.setdefault("data_generation", {}).update(
        n_displacement_sensors=3,
        n_strain_gauges=2,
    )
    cfg.setdefault("bayesian", {}).update(
        n_samples=100,
        n_tune=50,
        n_chains=1,
    )
    return cfg
