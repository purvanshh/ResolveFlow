"""Responder unit tests."""

from resolveflow.agent.responder import Responder
from resolveflow.llm import MockProvider
from resolveflow.schemas import RetrievedEvidence


def test_mock_responder_returns_text():
    r = Responder(MockProvider(), brand="AmazonHelp", mode="llm")
    out = r.draft(
        message="I was charged twice",
        context="",
        intent="payment_billing",
        evidence=[
            RetrievedEvidence(
                case_id="c1",
                similarity=0.9,
                customer_issue="charged twice",
                historical_response="Please contact us via chat.",
                intent="payment_billing",
            )
        ],
    )
    assert isinstance(out, str)
    assert len(out) > 10
