"""
Unit tests for configuration loading utilities.
"""

from __future__ import annotations

import tempfile

import pytest
import yaml

from apps.backend.core.utils.config import load_config


class TestLoadConfig:
    def test_load_valid_config(self) -> None:
        config = {
            "beam_parameters": {"length": 1.0, "width": 0.1},
            "material": {"elastic_modulus": 210e9},
            "load": {"point_load": 1000.0},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            loaded = load_config(f.name)

        assert loaded["beam_parameters"]["length"] == 1.0
        assert loaded["material"]["elastic_modulus"] == 210e9

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_missing_sections_get_defaults(self) -> None:
        """Config with missing required sections should get empty dicts."""
        config = {"output_dir": "outputs"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            loaded = load_config(f.name)

        assert "beam_parameters" in loaded
        assert "material" in loaded
        assert "load" in loaded

    def test_scientific_notation_conversion(self) -> None:
        """Numeric strings like '210e9' should be converted to floats."""
        config = {
            "beam_parameters": {},
            "material": {"elastic_modulus": "2.1e+11"},
            "load": {},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            loaded = load_config(f.name)

        assert isinstance(loaded["material"]["elastic_modulus"], float)
        assert loaded["material"]["elastic_modulus"] == pytest.approx(2.1e11)

    def test_default_config_loads(self) -> None:
        """The shipped default_config.yaml should load without errors."""
        config = load_config("configs/default_config.yaml")
        assert "beam_parameters" in config
        assert "bayesian" in config
        assert config["bayesian"]["n_chains"] == 4
