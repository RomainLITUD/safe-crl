"""YAML configuration helpers for SafeNav JAX envs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_env_config(
    env_id: str,
    config_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Loads constructor parameters for an env id from YAML."""
    path = Path(config_path) if config_path is not None else Path(config_dir or DEFAULT_CONFIG_DIR) / f"{env_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Env config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Env config must be a mapping: {path}")

    config_env_id = data.get("env_id")
    if config_env_id is not None and config_env_id != env_id:
        raise ValueError(f"Config {path} has env_id={config_env_id!r}, expected {env_id!r}.")

    params = data.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"Config params must be a mapping: {path}")
    return dict(params)
