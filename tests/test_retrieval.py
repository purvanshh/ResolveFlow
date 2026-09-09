"""Retrieval unit tests with a tiny in-memory index."""

import numpy as np
import pandas as pd

from resolveflow.retrieval import RetrievedCase
from resolveflow.retrieval.evidence import format_evidence
from resolveflow.retrieval.index import RetrievalIndex


def _tiny_index() -> RetrievalIndex:
    cases = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "conversation_id": "10",
                "customer_message": "My payment failed at checkout",
                "customer_context": "",
                "brand_response": "Please try again or contact support.",
                "intent": "payment_billing",
                "resolution_summary": "Brand asked customer to retry or contact support.",
            },
            {
                "case_id": "c2",
                "conversation_id": "11",
                "customer_message": "Where is my refund?",
                "customer_context": "",
                "brand_response": "Refunds can take several days.",
                "intent": "refund_status",
                "resolution_summary": "Brand mentioned refund timing.",
            },
        ]
    )
    # Hand-crafted 2D embeddings: payment-like vs refund-like
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return RetrievalIndex(
        emb,
        ["c1", "c2"],
        cases,
        metadata={"embedding_model": "toy", "num_cases": 2},
    )


def test_known_similar_query_retrieves_expected_case():
    index = _tiny_index()
    # Query near payment vector
    hits = index.search(np.array([0.99, 0.01], dtype=np.float32), top_k=1)
    assert hits[0][0] == "c1"
    assert hits[0][1] > 0.9


def test_threshold_can_return_no_evidence():
    index = _tiny_index()
    hits = index.search(
        np.array([0.6, 0.6], dtype=np.float32),
        top_k=5,
        similarity_threshold=0.99,
    )
    assert hits == []


def test_determinism_same_ordering():
    index = _tiny_index()
    q = np.array([0.8, 0.2], dtype=np.float32)
    a = index.search(q, top_k=2, similarity_threshold=0.0)
    b = index.search(q, top_k=2, similarity_threshold=0.0)
    assert a == b


def test_evidence_formatter_empty():
    assert "No sufficiently similar" in format_evidence([])
    text = format_evidence(
        [
            RetrievedCase(
                case_id="c1",
                similarity=0.9,
                customer_message="payment failed",
                brand_response="try again",
                intent="payment_billing",
                resolution_summary="Brand asked to retry.",
            )
        ]
    )
    assert "CASE 1" in text
    assert "payment_billing" in text
