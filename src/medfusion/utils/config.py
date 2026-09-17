"""YAML config loading with inheritance and environment-variable expansion.

A config may list parent files under ``inherit`` (paths relative to the file). Parents are merged
left to right, then the file itself is merged on top. Strings may contain ``${VAR:default}``.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``. Lists are replaced, not merged."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    merged: dict = {}
    for parent in raw.pop("inherit", []) or []:
        merged = deep_merge(merged, load_config(path.parent / parent))
    merged = deep_merge(merged, raw)
    if overrides:
        merged = deep_merge(merged, overrides)
    return _expand(merged)


def parse_overrides(pairs: list[str]) -> dict:
    """Turn ``["train.epochs=2", "model.name=b0"]`` into a nested dict (values parsed as YAML)."""
    out: dict = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out
