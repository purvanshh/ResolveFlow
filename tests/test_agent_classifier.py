"""Classifier tests."""

from resolveflow.agent.classifier import IntentClassifier
from resolveflow.llm import MockProvider
from resolveflow.taxonomy import intent_names, load_taxonomy


def test_mock_classifier_known_pattern():
    tax = load_taxonomy()
    clf = IntentClassifier(
        MockProvider(intents=intent_names(tax)),
        brand="AmazonHelp",
        taxonomy=tax,
        mode="llm",
    )
    pred = clf.classify("I was charged twice for my order")
    assert pred.intent == "payment_billing"
    assert 0 <= pred.confidence <= 1
