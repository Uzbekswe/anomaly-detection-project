"""Utility functions for loading and resolving serving configuration."""

import os
import re
from pathlib import Path

import yaml

# Regex to match ${VAR:-default} patterns in YAML values
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def resolve_env_vars(obj: dict | list | str | int | float | bool | None) -> dict | list | str | int | float | bool | None:
    """Recursively resolve ${VAR:-default} patterns in a loaded YAML config.

    Works on strings, dicts, and lists. Non-string values pass through unchanged.
    """
    if isinstance(obj, str):
        def _replace(match: re.Match) -> str:
            env_key = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(env_key, default_val)
        return _ENV_VAR_PATTERN.sub(_replace, obj)
    elif isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]
    return obj


def load_serving_config(config_path: Path) -> dict:
    """Load serving configuration with environment variable resolution."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return resolve_env_vars(config)
