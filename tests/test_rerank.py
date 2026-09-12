"""Unit tests for deterministic retrieval reranking (no gold leakage)."""

from resolveflow.retrieval import RetrievedCase
from resolveflow.retrieval.rerank import (
    cue_overlap_score,
    rerank_cases,
    resolution_cues,
)


def _case(cid, sim, intent, brand, summary=""):
    return RetrievedCase(
        case_id=cid,
        similarity=sim,
        customer_message="issue",
        brand_response=brand,
        intent=intent,
        resolution_summary=summary or f"Brand replied: {brand}",
    )


def test_resolution_cues_extract_actions():
    cues = resolution_cues("Please refund my order and replace the damaged item")
    assert "refund" in cues
    assert "replace" in cues


def test_cue_overlap_jaccard():
    assert cue_overlap_score({"refund"}, {"refund", "contact"}) == 0.5
    assert cue_overlap_score(set(), {"refund"}) == 0.0


def test_intent_rerank_prefers_matching_intent():
    cases = [
        _case("a", 0.80, "delivery_delay", "Where is the package?"),
        _case("b", 0.78, "refund_request", "We can look into a refund."),
    ]
    out = rerank_cases(
        cases,
        query="I want a refund for my late package",
        predicted_intent="refund_request",
        mode="intent",
    )
    assert out[0].case_id == "b"


def test_resolution_rerank_prefers_cue_overlap():
    cases = [
        _case("a", 0.81, "delivery_delay", "Sorry for the wait. Check tracking."),
        _case("b", 0.79, "delivery_delay", "Please contact us about a refund option."),
    ]
    out = rerank_cases(
        cases,
        query="Package late — can I get a refund?",
        predicted_intent="delivery_delay",
        mode="resolution",
    )
    assert out[0].case_id == "b"


def test_none_mode_preserves_order():
    cases = [
        _case("a", 0.9, "x", "a"),
        _case("b", 0.8, "y", "b"),
    ]
    out = rerank_cases(cases, query="q", predicted_intent="y", mode="none")
    assert [c.case_id for c in out] == ["a", "b"]


def test_rerank_does_not_use_gold_fields():
    """Sanity: scoring API has no gold_intent / gold_resolution parameters."""
    import inspect

    from resolveflow.retrieval import rerank as mod

    sig = str(inspect.signature(mod.rerank_cases))
    assert "gold" not in sig.lower()
