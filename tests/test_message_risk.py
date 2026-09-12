"""Message-level risk cue tests (conservative patterns)."""

from resolveflow.agent.policy import EscalationPolicy
from resolveflow.agent.safety import detect_message_risk
from resolveflow.schemas import IntentPrediction, SafetyResult


def test_refund_money_cue():
    assert "refund_money_request" in detect_message_risk(
        "At least give me my money back"
    )
    assert "refund_money_request" in detect_message_risk("Refund options?")


def test_defective_wrong_item_cue():
    assert "defective_or_wrong_item" in detect_message_risk(
        "how to return the defective item"
    )
    assert "defective_or_wrong_item" in detect_message_risk(
        "shipping the wrong item then delayed"
    )


def test_never_received_cue():
    assert "never_received_item" in detect_message_risk(
        "my delivery didn't arrive yesterday"
    )
    assert "never_received_item" in detect_message_risk(
        "still i didnt get the order"
    )


def test_fraud_cue():
    assert "fraud_or_account_takeover" in detect_message_risk(
        "unauthorized charge on my card"
    )


def test_safe_delay_not_flagged():
    """Bare delay language must not trip risk cues."""
    assert detect_message_risk("My package is delayed past the promised date") == []
    assert detect_message_risk("Where is my package? Tracking hasn't moved.") == []
    assert detect_message_risk("Please cancel my order if possible") == []


def test_message_risk_forces_escalate_despite_non_high_risk_intent():
    p = EscalationPolicy()
    d = p.decide(
        message="Package late — at least give me my money back",
        intent=IntentPrediction(intent="delivery_delay", confidence=0.9),
        top_similarity=0.8,
        evidence_count=2,
        evidence_sufficient=True,
        safety=SafetyResult(safe=True),
    )
    assert d.escalate
    assert "message_risk_cues" in d.triggers
