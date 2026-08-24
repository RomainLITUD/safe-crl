"""Experiment artifact helpers for training runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _serializable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _serializable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _serializable(value.tolist())
        except (TypeError, ValueError):
            pass
    return repr(value)


def resolved_config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        raw = asdict(config)
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = vars(config)
    return _serializable(raw)


def write_resolved_config(run_dir: str | Path, config: Any) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dict = resolved_config_dict(config)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_dict, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    if yaml is not None:
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(config_dict, file, sort_keys=False)
    return config_dict


def append_metrics_json_line(path: str | Path, metrics_line: str) -> None:
    prefix = "METRICS_JSON\t"
    if not metrics_line.startswith(prefix):
        raise ValueError("Expected a METRICS_JSON payload line.")
    payload = json.loads(metrics_line[len(prefix) :])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
        file.write("\n")


def unique_run_dir(base_path: str | Path) -> Path:
    base_path = Path(base_path)
    candidate = base_path
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base_path}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
