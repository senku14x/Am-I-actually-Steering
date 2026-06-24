"""
Config loader: YAML files + ${env:VAR:default} interpolation + dict overrides.

Replaces v1's hardcoded Colab paths / module-level `CFG` globals. Compose a base config with a
model config:

    cfg = load_config("configs/base.yaml", "configs/model_qwen14b.yaml", seed=0)
    cfg.headline_layer            # 24
    cfg.data_dir                  # resolved from $DATA_ROOT or ./data
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{env:([^:}]+)(?::([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    """Recursively replace ${env:NAME:default} in strings using os.environ."""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        return _ENV.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


@dataclass
class Config:
    # identity / model
    model_name: str = ""
    model_short: str = ""
    n_layers: int = 0
    hidden_dim: int = 0
    headline_layer: int = 0
    max_new_tokens: int = 4096
    # data / io
    data_dir: str = "./data"
    results_dir: str = "./results"
    seed: int = 42
    # taxonomy / geometry
    labels: list[str] = field(default_factory=list)
    think_only: bool = True
    min_count: int = 20
    headline_pair: list[str] = field(default_factory=lambda: ["opponent_modeling", "deduction"])
    # stats
    n_permutation: int = 1000
    n_bootstrap: int = 1000
    n_seeds: int = 3
    # anything not modeled above is preserved here
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)} - {"extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs, extra=extra)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def results_path(self) -> Path:
        return Path(self.results_dir)


def load_config(*paths: str, **overrides: Any) -> Config:
    """Merge YAML configs left-to-right, interpolate env vars, apply keyword overrides."""
    merged: dict = {}
    for p in paths:
        with open(p) as f:
            merged = _deep_merge(merged, yaml.safe_load(f) or {})
    merged = _deep_merge(merged, overrides)
    merged = _interpolate(merged)
    return Config.from_dict(merged)
