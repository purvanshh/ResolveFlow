"""Tests for golden-set schema and freeze integrity."""

from pathlib import Path

import pandas as pd
import pytest

from resolveflow.golden import (
    load_golden_set,
    save_golden_set,
    sha256_file,
    validate_golden_labels,
    verify_frozen_checksum,
)
from resolveflow.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_labels_valid():
    tax = load_taxonomy(ROOT / "configs" / "intents.yaml")
    df = load_golden_set(ROOT / "data" / "golden" / "golden_set.csv")
    assert 150 <= len(df) <= 250
    errors = validate_golden_labels(df, tax, require_complete=True)
    assert errors == [], errors[:10]


def test_duplicate_example_ids_rejected():
    tax = load_taxonomy(ROOT / "configs" / "intents.yaml")
    df = load_golden_set(ROOT / "data" / "golden" / "golden_set.csv").head(2).copy()
    df.loc[df.index[1], "example_id"] = df.iloc[0]["example_id"]
    errors = validate_golden_labels(df, tax, require_complete=True)
    assert any("Duplicate example_id" in e for e in errors)


def test_invalid_intent_rejected():
    tax = load_taxonomy(ROOT / "configs" / "intents.yaml")
    df = load_golden_set(ROOT / "data" / "golden" / "golden_set.csv").head(1).copy()
    df.loc[df.index[0], "intent"] = "not_a_real_intent"
    errors = validate_golden_labels(df, tax, require_complete=True)
    assert any("invalid intent" in e for e in errors)


def test_escalate_requires_reason():
    tax = load_taxonomy(ROOT / "configs" / "intents.yaml")
    df = load_golden_set(ROOT / "data" / "golden" / "golden_set.csv").head(1).copy()
    df.loc[df.index[0], "escalation_expected"] = "escalate"
    df.loc[df.index[0], "escalation_reason"] = ""
    errors = validate_golden_labels(df, tax, require_complete=True)
    assert any("escalation_reason" in e for e in errors)


def test_frozen_checksum_matches_when_present():
    manifest = ROOT / "data" / "golden" / "golden_set_manifest.json"
    golden = ROOT / "data" / "golden" / "golden_set.csv"
    if not manifest.exists():
        pytest.skip("manifest not frozen yet")
    assert verify_frozen_checksum(golden, manifest)
    # Tamper detection: checksum of current file equals manifest
    import json

    data = json.loads(manifest.read_text())
    assert data["sha256"] == sha256_file(golden)
