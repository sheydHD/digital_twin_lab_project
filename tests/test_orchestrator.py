"""
Unit / integration tests for the pipeline orchestrator.

These tests use the default_config.yaml and run only the
data-generation stage (which doesn't require MCMC).
"""

from __future__ import annotations

import pytest

from apps.backend.core.utils.config import load_config


@pytest.fixture
def config() -> dict:
    cfg = load_config("configs/default_config.yaml")
    # Shrink to make the test fast
    cfg["beam_parameters"]["aspect_ratios"] = [10.0, 20.0]
    cfg.setdefault("data_generation", {})["n_displacement_sensors"] = 3
    cfg.setdefault("data_generation", {})["n_strain_gauges"] = 2
    return cfg


class TestPipelineOrchestrator:
    def test_init(self, config: dict) -> None:
        from apps.backend.core.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator(config)
        assert orch.config is config
        assert orch.output_dir.exists()
        assert orch.datasets == []

    def test_data_generation(self, config: dict) -> None:
        from apps.backend.core.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator(config)
        datasets = orch.run_data_generation()

        assert len(datasets) == 2
        for ds in datasets:
            assert ds.displacements is not None
            assert len(ds.displacements) == 3
