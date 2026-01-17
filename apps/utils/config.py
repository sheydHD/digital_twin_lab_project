"""
Configuration Loading Utilities.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def _convert_numeric_strings(obj: Any) -> Any:
    """
    Recursively convert numeric strings to floats/ints.
    
    Handles scientific notation like '210e9' or '1.0e-6' that YAML
    may parse as strings.
    """
    if isinstance(obj, dict):
        return {k: _convert_numeric_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numeric_strings(item) for item in obj]
    elif isinstance(obj, str):
        # Try to convert to number if it looks numeric
        try:
            if '.' in obj or 'e' in obj.lower():
                return float(obj)
            return int(obj)
        except ValueError:
            return obj
    return obj


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert any numeric strings (handles scientific notation edge cases)
    config = _convert_numeric_strings(config)

    # Validate required sections
    required_sections = ["beam_parameters", "material", "load"]
    for section in required_sections:
        if section not in config:
            config[section] = {}

    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Output path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration dictionary.

    Returns:
        Default configuration
    """
    return {
        "beam_parameters": {
            "length": 1.0,
            "width": 0.1,
            "aspect_ratios": [5, 8, 10, 12, 15, 20, 30, 50],
        },
        "material": {
            "elastic_modulus": 210e9,
            "poisson_ratio": 0.3,
            "density": 7850,
        },
        "load": {
            "point_load": 1000,
            "distributed_load": 0,
        },
        "data_generation": {
            "n_displacement_sensors": 5,
            "n_strain_gauges": 4,
            "displacement_noise": 1e-6,
            "strain_noise": 1e-6,
            "relative_noise": True,
            "noise_fraction": 0.005,
            "seed": 42,
        },
        "bayesian": {
            "n_samples": 2000,
            "n_tune": 1000,
            "n_chains": 4,
            "target_accept": 0.9,
        },
        "output_dir": "outputs",
    }
