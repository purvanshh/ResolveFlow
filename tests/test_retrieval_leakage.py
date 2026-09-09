"""Golden examples must never enter the retrieval corpus/index."""

from pathlib import Path

import pandas as pd
import pytest

from resolveflow.golden import load_golden_set

ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_corpus_excludes_golden_conversations():
    cases_path = ROOT / "data" / "processed" / "historical_cases.parquet"
    if not cases_path.exists():
        pytest.skip("historical cases not built yet")
    golden = load_golden_set(ROOT / "data" / "golden" / "golden_set.csv")
    cases = pd.read_parquet(cases_path)
    g_convs = set(golden["conversation_id"].astype(str))
    c_convs = set(cases["conversation_id"].astype(str))
    overlap = g_convs & c_convs
    assert overlap == set(), f"Golden conversations in retrieval corpus: {list(overlap)[:5]}"


def test_retrieval_corpus_excludes_exact_customer_text():
    cases_path = ROOT / "data" / "processed" / "historical_cases.parquet"
    if not cases_path.exists():
        pytest.skip("historical cases not built yet")
    golden = load_golden_set(ROOT / "data" / "golden" / "golden_set.csv")
    cases = pd.read_parquet(cases_path)
    g_text = set(golden["input_text"].astype(str).str.strip())
    c_text = set(cases["customer_message"].astype(str).str.strip())
    overlap = g_text & c_text
    assert overlap == set(), f"Exact text overlap count={len(overlap)}"


def test_retrieval_metadata_case_ids_aligned_when_present():
    import json

    meta = ROOT / "data" / "processed" / "retrieval_metadata.json"
    ids_path = ROOT / "data" / "processed" / "retrieval_case_ids.json"
    cases_path = ROOT / "data" / "processed" / "historical_cases.parquet"
    if not (meta.exists() and ids_path.exists() and cases_path.exists()):
        pytest.skip("index not built")
    case_ids = set(json.loads(ids_path.read_text()))
    cases = pd.read_parquet(cases_path)
    assert case_ids <= set(cases["case_id"].astype(str))
    golden = load_golden_set(ROOT / "data" / "golden" / "golden_set.csv")
    g_convs = set(golden["conversation_id"].astype(str))
    indexed = cases[cases["case_id"].astype(str).isin(case_ids)]
    assert set(indexed["conversation_id"].astype(str)) & g_convs == set()
