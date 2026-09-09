"""Tests for baseline predictors."""

from resolveflow.baselines import (
    AlwaysEscalateBaseline,
    MajorityBaseline,
    TfidfBaseline,
    majority_label,
)


def test_majority_always_same():
    b = MajorityBaseline("refund_request", "Thanks")
    for text in ["hello", "charged twice", "package late"]:
        intent, conf = b.predict_intent(text)
        assert intent == "refund_request"
        assert conf == 1.0
        assert b.decide_escalation(text) == "escalate"


def test_always_escalate():
    b = AlwaysEscalateBaseline("other_unclear", "Thanks")
    assert b.decide_escalation("anything") == "escalate"


def test_tfidf_trains_on_fixture():
    texts = [
        "my package is delayed and late",
        "delivery is late again today",
        "shipment delayed past promised date",
        "I want a refund please",
        "please refund my order now",
        "give me my money back refund",
        "app keeps crashing on checkout",
        "website error currently unavailable",
        "the app crashed again",
    ]
    labels = [
        "delivery_delay",
        "delivery_delay",
        "delivery_delay",
        "refund_request",
        "refund_request",
        "refund_request",
        "technical_issue",
        "technical_issue",
        "technical_issue",
    ]
    model = TfidfBaseline.train(
        texts,
        labels,
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
        trivial_reply="Thanks",
        confidence_threshold=0.3,
    )
    intent, conf = model.predict_intent("my delivery is delayed")
    assert intent == "delivery_delay"
    assert conf > 0.0
    assert majority_label(labels) in set(labels)
