"""Golden evaluation set helpers: schema, validation, freeze, leakage."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from resolveflow.taxonomy import intent_names, load_taxonomy

GOLDEN_COLUMNS = [
    "example_id",
    "conversation_id",
    "message_id",
    "input_text",
    "context",
    "intent",
    "escalation_expected",
    "escalation_reason",
    "difficulty",
    "annotation_notes",
]

VALID_ESCALATION = {"auto_handle", "escalate", "uncertain"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}

DEFAULT_GOLDEN_CSV = Path("data/golden/golden_set.csv")
DEFAULT_MANIFEST = Path("data/golden/golden_set_manifest.json")


def empty_golden_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=GOLDEN_COLUMNS)


def load_golden_set(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else DEFAULT_GOLDEN_CSV
    if not csv_path.exists():
        return empty_golden_frame()
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    for col in GOLDEN_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[GOLDEN_COLUMNS]


def save_golden_set(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    csv_path = Path(path) if path else DEFAULT_GOLDEN_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in GOLDEN_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[GOLDEN_COLUMNS]
    out.to_csv(csv_path, index=False)
    return csv_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_golden_labels(
    df: pd.DataFrame,
    taxonomy: dict[str, Any] | None = None,
    *,
    require_complete: bool = True,
) -> list[str]:
    """Return validation errors."""
    errors: list[str] = []
    if taxonomy is None:
        taxonomy = load_taxonomy()
    allowed = set(intent_names(taxonomy))

    if df.empty and require_complete:
        errors.append("Golden set is empty")
        return errors

    if df["example_id"].duplicated().any():
        errors.append("Duplicate example_id values")

    for idx, row in df.iterrows():
        eid = row.get("example_id", f"row_{idx}")
        intent = str(row.get("intent", "")).strip()
        esc = str(row.get("escalation_expected", "")).strip()
        diff = str(row.get("difficulty", "")).strip()
        reason = str(row.get("escalation_reason", "")).strip()

        if require_complete or intent or esc or diff:
            if not intent:
                errors.append(f"{eid}: missing intent")
            elif intent not in allowed:
                errors.append(f"{eid}: invalid intent '{intent}'")
            if not esc:
                errors.append(f"{eid}: missing escalation_expected")
            elif esc not in VALID_ESCALATION:
                errors.append(f"{eid}: invalid escalation '{esc}'")
            if not diff:
                errors.append(f"{eid}: missing difficulty")
            elif diff not in VALID_DIFFICULTY:
                errors.append(f"{eid}: invalid difficulty '{diff}'")
            if esc == "escalate" and not reason:
                errors.append(f"{eid}: escalate requires escalation_reason")
            if not str(row.get("input_text", "")).strip():
                errors.append(f"{eid}: missing input_text")

    return errors


def freeze_golden_set(
    *,
    golden_path: Path | None = None,
    manifest_path: Path | None = None,
    sampling_method: str = "",
    random_seed: int = 42,
    dataset_version: str = "twcs.csv",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    golden_path = golden_path or DEFAULT_GOLDEN_CSV
    manifest_path = manifest_path or DEFAULT_MANIFEST
    df = load_golden_set(golden_path)
    errors = validate_golden_labels(df, require_complete=True)
    if errors:
        raise ValueError("Cannot freeze invalid golden set:\n" + "\n".join(errors[:20]))

    checksum = sha256_file(golden_path)
    manifest = {
        "status": "FROZEN",
        "sha256": checksum,
        "num_examples": int(len(df)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "sampling_method": sampling_method,
        "dataset_version": dataset_version,
        "checksum": checksum,
        "examples": int(len(df)),
    }
    if extra:
        manifest.update(extra)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def is_frozen(manifest_path: Path | None = None) -> bool:
    path = manifest_path or DEFAULT_MANIFEST
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return data.get("status") == "FROZEN"


def verify_frozen_checksum(
    golden_path: Path | None = None,
    manifest_path: Path | None = None,
) -> bool:
    golden_path = golden_path or DEFAULT_GOLDEN_CSV
    manifest_path = manifest_path or DEFAULT_MANIFEST
    if not manifest_path.exists() or not golden_path.exists():
        return False
    data = json.loads(manifest_path.read_text())
    expected = data.get("sha256") or data.get("checksum")
    return expected == sha256_file(golden_path)


def check_leakage(
    golden: pd.DataFrame,
    development: pd.DataFrame,
    *,
    text_col_golden: str = "input_text",
    text_col_dev: str = "raw_text",
    norm_col_dev: str = "normalized_text",
) -> dict[str, Any]:
    """
    Compare golden examples against a development message table.

    ``development`` should have conversation_id and text columns.
    """
    from resolveflow.data.normalize import normalize_customer_text

    g_convs = set(golden["conversation_id"].astype(str))
    d_convs = set(development["conversation_id"].astype(str))
    conv_overlap = sorted(g_convs & d_convs)

    g_text = set(golden[text_col_golden].astype(str).str.strip())
    d_text = set(development[text_col_dev].astype(str).str.strip())
    exact = sorted(g_text & d_text)

    g_norm = {normalize_customer_text(t) for t in golden[text_col_golden].astype(str)}
    if norm_col_dev in development.columns:
        d_norm = set(development[norm_col_dev].astype(str))
    else:
        d_norm = {normalize_customer_text(t) for t in development[text_col_dev].astype(str)}
    g_norm.discard("")
    d_norm.discard("")
    norm_overlap = sorted(g_norm & d_norm)

    return {
        "conversation_overlap": len(conv_overlap),
        "exact_text_overlap": len(exact),
        "normalized_text_overlap": len(norm_overlap),
        "conversation_ids": conv_overlap[:20],
        "status": "PASS"
        if not (conv_overlap or exact or norm_overlap)
        else "FAIL",
    }
