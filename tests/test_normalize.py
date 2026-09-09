"""Normalization unit tests."""

from resolveflow.data.normalize import normalize_customer_text


def test_normalize_strips_brand_and_urls():
    raw = "HEY @AmazonHelp!!!! My card was charged twice 😭😭 https://example.com/x"
    out = normalize_customer_text(raw, brand="AmazonHelp")
    assert "AmazonHelp" not in out
    assert "[URL]" in out
    assert "charged twice" in out
    assert "😭" in out
    assert "!!!!" not in out


def test_normalize_empty():
    assert normalize_customer_text(None) == ""
    assert normalize_customer_text("   ") == ""
