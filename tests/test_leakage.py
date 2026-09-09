"""Leakage tests between golden and development message tables."""

from pathlib import Path

import pandas as pd
import pytest

from resolveflow.golden import check_leakage, load_golden_set

ROOT = Path(__file__).resolve().parents[1]


def test_no_leakage_against_development_set():
    dev_path = ROOT / "data" / "interim" / "development_messages.parquet"
    if not dev_path.exists():
        pytest.skip("development set not built; run check_leakage.py --build-dev")
    golden = load_golden_set(ROOT / "data" / "golden" / "golden_set.csv")
    dev = pd.read_parquet(dev_path)
    result = check_leakage(golden, dev)
    assert result["conversation_overlap"] == 0
    assert result["exact_text_overlap"] == 0
    assert result["normalized_text_overlap"] == 0
    assert result["status"] == "PASS"


def test_leakage_detector_finds_overlap():
    golden = pd.DataFrame(
        {
            "example_id": ["g1"],
            "conversation_id": ["99"],
            "message_id": ["1"],
            "input_text": ["hello world charge"],
            "context": [""],
            "intent": ["other_unclear"],
            "escalation_expected": ["auto_handle"],
            "escalation_reason": [""],
            "difficulty": ["easy"],
            "annotation_notes": [""],
        }
    )
    dev = pd.DataFrame(
        {
            "conversation_id": ["99"],
            "raw_text": ["hello world charge"],
            "normalized_text": ["hello world charge"],
        }
    )
    result = check_leakage(golden, dev)
    assert result["status"] == "FAIL"
    assert result["conversation_overlap"] == 1
