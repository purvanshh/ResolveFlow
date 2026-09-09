"""Escalation policy tests."""

from resolveflow.agent.policy import EscalationPolicy
from resolveflow.schemas import IntentPrediction, SafetyResult


def _intent(name, conf=0.9):
    return IntentPrediction(intent=name, confidence=conf)


def test_high_risk_intent_escalates():
    p = EscalationPolicy()
    d = p.decide(
        message="charged twice",
        intent=_intent("payment_billing", 0.95),
        top_similarity=0.9,
        evidence_count=3,
        evidence_sufficient=True,
        safety=SafetyResult(safe=True),
    )
    assert d.escalate
    assert "high_risk_intent" in d.triggers


def test_low_retrieval_escalates():
    p = EscalationPolicy()
    d = p.decide(
        message="hello there about my order timing",
        intent=_intent("delivery_delay", 0.8),
        top_similarity=0.2,
        evidence_count=0,
        evidence_sufficient=False,
        safety=SafetyResult(safe=True),
    )
    assert d.escalate


def test_clean_low_risk_can_auto_handle():
    p = EscalationPolicy(high_risk_intents=["payment_billing"])
    d = p.decide(
        message="my package is delayed past the promised date",
        intent=_intent("delivery_delay", 0.8),
        top_similarity=0.8,
        evidence_count=2,
        evidence_sufficient=True,
        safety=SafetyResult(safe=True),
    )
    assert not d.escalate
