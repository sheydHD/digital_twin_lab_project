"""
Configuration Loading Utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
            if "." in obj or "e" in obj.lower():
                return float(obj)
            return int(obj)
        except ValueError:
            return obj
    return obj


def load_config(config_path: str) -> dict[str, Any]:
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

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert any numeric strings (handles scientific notation edge cases)
    config = _convert_numeric_strings(config)

    # Validate required sections
    required_sections = ["beam_parameters", "material", "load"]
    for section in required_sections:
        if section not in config:
            config[section] = {}

    return config
