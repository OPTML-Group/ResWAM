"""Configuration loading and environment variable expansion for ResWAM."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR:-default}`` expressions in strings, dicts, and lists."""
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_val)

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (mappings merge, other values replace)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML or JSON configuration file and expand environment variables.

    A top-level ``base: <relative path>`` key loads that file first and deep-merges this
    file's remaining keys on top of it.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in (".yaml", ".yml"):
        parsed = yaml.safe_load(raw_text) or {}
    else:
        parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    base = parsed.pop("base", None)
    if base is not None:
        parsed = deep_merge(load_config(config_path.parent / str(base)), parsed)
    return expand_env_vars(parsed)


def resolve_runtime_paths() -> dict[str, Path]:
    """Return standard configurable root directories from environment variables."""
    return {
        "data_root": Path(os.environ.get("RESWAM_DATA_ROOT", "./data")).resolve(),
        "cache_root": Path(os.environ.get("RESWAM_CACHE_ROOT", "./cache")).resolve(),
        "output_dir": Path(os.environ.get("RESWAM_OUTPUT_DIR", "./outputs")).resolve(),
        "checkpoint_root": Path(os.environ.get("RESWAM_CHECKPOINT_ROOT", "./checkpoints")).resolve(),
    }
