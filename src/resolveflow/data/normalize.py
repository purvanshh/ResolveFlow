"""Lightweight customer-message normalization for intent discovery."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_PUNCT_RE = re.compile(r"([!?\.])\1{2,}")


def normalize_customer_text(text: str, *, brand: str | None = None) -> str:
    """
    Light normalization that preserves meaning.

    - URLs → [URL]
    - @brand / @handles removed (brand mention is implied by channel)
    - collapse whitespace
    - soften extreme repeated punctuation (!!!! → !)
    - keep emojis and casing mostly intact
    """
    if text is None:
        return ""
    out = str(text)
    out = _URL_RE.sub("[URL]", out)
    if brand:
        # Remove exact brand handle case-insensitively, keep other @handles stripped too
        brand_re = re.compile(rf"@{re.escape(brand)}\b", re.IGNORECASE)
        out = brand_re.sub("", out)
    out = _MENTION_RE.sub("", out)
    out = _REPEAT_PUNCT_RE.sub(r"\1", out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out
