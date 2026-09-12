"""Deterministic resolution-/intent-aware reranking over dense retrieval hits.

Uses only inference-time signals: query text, predicted intent (if provided),
and historical case fields already on RetrievedCase. Never uses gold labels.
"""

from __future__ import annotations

import re
from typing import Iterable

from resolveflow.retrieval import RetrievedCase

# Action / resolution cues mined from customer or brand text (no invented outcomes).
_CUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("refund", re.compile(r"\brefund", re.I)),
    ("replace", re.compile(r"\breplac", re.I)),
    ("return", re.compile(r"\breturn", re.I)),
    ("cancel", re.compile(r"\bcancel", re.I)),
    ("track", re.compile(r"\b(track|ship|deliver|courier|carrier)\b", re.I)),
    ("account", re.compile(r"\b(account|password|login|sign[\s-]?in)\b", re.I)),
    ("payment", re.compile(r"\b(payment|charge|billing|card|invoice)\b", re.I)),
    ("contact", re.compile(r"\b(contact|reach out|call|chat|dm|direct message)\b", re.I)),
]

DEFAULT_INTENT_BONUS = 0.12
DEFAULT_RESOLUTION_BONUS = 0.10


def resolution_cues(text: str) -> frozenset[str]:
    """Extract coarse resolution/action cues present in free text."""
    t = text or ""
    return frozenset(name for name, pat in _CUE_PATTERNS if pat.search(t))


def cue_overlap_score(query_cues: Iterable[str], hist_cues: Iterable[str]) -> float:
    """Jaccard overlap in [0, 1]; 0 if either side empty."""
    q, h = set(query_cues), set(hist_cues)
    if not q or not h:
        return 0.0
    return len(q & h) / len(q | h)


def rerank_score(
    case: RetrievedCase,
    *,
    query: str,
    predicted_intent: str | None = None,
    intent_bonus: float = DEFAULT_INTENT_BONUS,
    resolution_bonus: float = DEFAULT_RESOLUTION_BONUS,
    use_intent: bool = False,
    use_resolution: bool = False,
) -> float:
    """Transparent score = similarity + optional bonuses (inference-time only)."""
    score = float(case.similarity)
    if use_intent and predicted_intent and case.intent == predicted_intent:
        score += intent_bonus
    if use_resolution:
        q_cues = resolution_cues(query)
        hist_text = " ".join(
            [
                case.brand_response or "",
                case.resolution_summary or "",
            ]
        )
        score += resolution_bonus * cue_overlap_score(q_cues, resolution_cues(hist_text))
    return score


def rerank_cases(
    cases: list[RetrievedCase],
    *,
    query: str,
    predicted_intent: str | None = None,
    mode: str = "none",
    top_k: int | None = None,
    intent_bonus: float = DEFAULT_INTENT_BONUS,
    resolution_bonus: float = DEFAULT_RESOLUTION_BONUS,
) -> list[RetrievedCase]:
    """
    Rerank retrieved candidates.

    Modes:
      - none: preserve order
      - intent: similarity + intent agreement with predicted_intent
      - resolution: similarity + resolution-cue overlap with query
      - combined: both bonuses
    """
    mode = (mode or "none").lower().strip()
    if mode in {"", "none", "off", "false"}:
        out = list(cases)
        return out[:top_k] if top_k is not None else out

    use_intent = mode in {"intent", "combined", "intent_resolution"}
    use_resolution = mode in {"resolution", "combined", "intent_resolution"}

    scored = [
        (
            rerank_score(
                c,
                query=query,
                predicted_intent=predicted_intent,
                intent_bonus=intent_bonus,
                resolution_bonus=resolution_bonus,
                use_intent=use_intent,
                use_resolution=use_resolution,
            ),
            -float(c.similarity),  # tie-break: higher raw sim first
            i,
            c,
        )
        for i, c in enumerate(cases)
    ]
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    out = [c for *_, c in scored]
    return out[:top_k] if top_k is not None else out
