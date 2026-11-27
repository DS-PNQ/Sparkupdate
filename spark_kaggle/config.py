"""YAML config loading with dotted-key command-line overrides.

Usage in every CLI:
    python -m spark_kaggle.<stage> --config configs/default.yaml --set sft.lr=5e-5 --set run_dir=/kaggle/working/run
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:  # pragma: no cover - attribute errors are user errors
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict) and not isinstance(obj, Cfg):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))


def _parse_value(raw: str) -> Any:
    try:
        val = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    if isinstance(val, str):  # YAML 1.1 reads "3e-5" as a string
        try:
            return float(val)
        except ValueError:
            pass
    return val


def apply_override(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = Cfg()
        cur = cur[p]
    cur[parts[-1]] = Cfg.wrap(value)


def load_config(path: str | os.PathLike | None, overrides: list[str] | None = None) -> Cfg:
    base_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(base_path) as f:
        cfg = yaml.safe_load(f) or {}
    if path and Path(path).resolve() != base_path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
    cfg = Cfg.wrap(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        apply_override(cfg, k.strip(), _parse_value(v))
    env_run_dir = os.environ.get("SPARK_RUN_DIR")
    if env_run_dir and not any(o.startswith("run_dir=") for o in overrides or []):
        cfg["run_dir"] = env_run_dir
    return Cfg.wrap(_interpolate(cfg, cfg))


_VAR = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")


def _interpolate(node: Any, root: Cfg, depth: int = 0) -> Any:
    """Resolve ${dotted.key} references (strings only, a few passes for chains)."""
    if depth > 8:
        return node
    if isinstance(node, dict):
        return {k: _interpolate(v, root, depth) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, root, depth) for v in node]
    if isinstance(node, str) and "${" in node:
        def sub(m: re.Match) -> str:
            val = root.get_path(m.group(1))
            if val is None:
                raise KeyError(f"config interpolation: unknown key {m.group(1)!r}")
            return str(val)

        return _interpolate(_VAR.sub(sub, node), root, depth + 1)
    return node


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="YAML config (merged over configs/default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="dotted.key=value override")
    return p


def cfg_from_args(args: argparse.Namespace) -> Cfg:
    return load_config(args.config, args.overrides)
