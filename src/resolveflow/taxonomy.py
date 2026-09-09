"""Intent taxonomy loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_TAXONOMY_PATH = Path("configs/intents.yaml")


def load_taxonomy(path: str | Path | None = None) -> dict[str, Any]:
    taxonomy_path = Path(path) if path else DEFAULT_TAXONOMY_PATH
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy not found: {taxonomy_path}")
    data = yaml.safe_load(taxonomy_path.read_text())
    if not isinstance(data, dict) or "intents" not in data:
        raise ValueError("Taxonomy must be a mapping with an 'intents' list")
    return data


def intent_ids(taxonomy: dict[str, Any]) -> list[str]:
    return [str(i["id"]) for i in taxonomy.get("intents", [])]


def intent_names(taxonomy: dict[str, Any]) -> list[str]:
    return [str(i["name"]) for i in taxonomy.get("intents", [])]


def validate_taxonomy(taxonomy: dict[str, Any]) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors: list[str] = []
    intents = taxonomy.get("intents") or []
    if not intents:
        errors.append("No intents defined")
        return errors

    ids: list[str] = []
    names: list[str] = []
    for i, intent in enumerate(intents):
        prefix = f"intent[{i}]"
        for field in ("id", "name", "description"):
            if not intent.get(field):
                errors.append(f"{prefix}: missing {field}")
        iid = str(intent.get("id", ""))
        name = str(intent.get("name", ""))
        if iid:
            ids.append(iid)
        if name:
            names.append(name)
        if not intent.get("include"):
            errors.append(f"{prefix} ({name or iid}): missing include examples")
        if not intent.get("exclude"):
            errors.append(f"{prefix} ({name or iid}): missing exclude examples")
        if "escalation_default" not in intent:
            errors.append(f"{prefix} ({name or iid}): missing escalation_default")

    if len(ids) != len(set(ids)):
        errors.append("Duplicate intent ids")
    if len(names) != len(set(names)):
        errors.append("Duplicate intent names")

    n = len(intents)
    if n < 8 or n > 15:
        errors.append(f"Intent count {n} outside recommended 8–15 range")

    if "other_unclear" not in names:
        errors.append("Missing required intent name: other_unclear")

    return errors
