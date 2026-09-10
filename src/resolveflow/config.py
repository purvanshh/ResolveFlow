"""Load ResolveFlow YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
_ENV_LOADED = False


def load_dotenv(override: bool = False) -> None:
    """Load project-root `.env` into the process environment (once)."""
    global _ENV_LOADED
    if _ENV_LOADED and not override:
        return
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv as _load

        _load(env_path, override=override)
    except ImportError:
        # Minimal fallback if python-dotenv is not installed
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'").strip('"')
                if key and (override or key not in __import__("os").environ):
                    __import__("os").environ[key] = value
    _ENV_LOADED = True


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv()
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {cfg_path}")
    data["_root"] = str(ROOT)
    return data


def resolve_path(cfg: dict[str, Any], relative: str) -> Path:
    root = Path(cfg.get("_root", ROOT))
    return (root / relative).resolve()
