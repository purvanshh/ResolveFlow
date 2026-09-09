"""Tests for taxonomy config."""

from pathlib import Path

from resolveflow.taxonomy import intent_names, load_taxonomy, validate_taxonomy

ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_loads_and_is_valid():
    tax = load_taxonomy(ROOT / "configs" / "intents.yaml")
    errors = validate_taxonomy(tax)
    assert errors == [], errors
    names = intent_names(tax)
    assert "other_unclear" in names
    assert 8 <= len(names) <= 15
    assert len(names) == len(set(names))


def test_every_intent_has_boundaries():
    tax = load_taxonomy(ROOT / "configs" / "intents.yaml")
    for intent in tax["intents"]:
        assert intent["include"]
        assert intent["exclude"]
        assert intent["description"]
        assert "escalation_default" in intent
