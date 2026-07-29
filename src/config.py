"""Project-wide configuration loader.

Loads the project YAML config and exposes it as a nested, attribute-accessible
dictionary so the rest of the codebase can do e.g. `config.model.bert_model_name`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
LEGACY_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class AttrDict(dict):
    """A dict that also allows attribute-style access, recursively."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.lower() in {"null", "none"}:
            return None
        if value.replace(".", "", 1).replace("-", "", 1).isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _normalize_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_data(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_data(item) for item in value]
    return _normalize_scalar(value)


def _merge_config_data(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config_data(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> AttrDict:
    """Load the YAML config file into an `AttrDict`.

    The loader first checks the configured path (or the `SENTIMENT_CONFIG_PATH`
    environment variable), then falls back to the project defaults so both
    config layouts in this repository work.
    """
    config_sources: list[Path] = []

    if config_path is not None:
        config_sources = [Path(config_path)]
    else:
        env_path = os.environ.get("SENTIMENT_CONFIG_PATH")
        if env_path:
            config_sources = [Path(env_path)]
        else:
            config_sources = [DEFAULT_CONFIG_PATH, LEGACY_CONFIG_PATH]

    merged_data: dict[str, Any] = {}
    loaded_paths: list[Path] = []

    for path in config_sources:
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            merged_data = _merge_config_data(merged_data, data)
            loaded_paths.append(path)

    if not merged_data:
        raise FileNotFoundError(
            f"Config file not found. Checked: {', '.join(str(p) for p in config_sources)}"
        )

    return AttrDict(_normalize_data(merged_data))


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


# Global config instance loaded at import time
config = load_config()