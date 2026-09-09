"""Load ResolveFlow YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {cfg_path}")
    data["_root"] = str(ROOT)
    return data


def resolve_path(cfg: dict[str, Any], relative: str) -> Path:
    root = Path(cfg.get("_root", ROOT))
    return (root / relative).resolve()
