"""Deterministic safety validation for drafted replies."""

from __future__ import annotations

import re

from resolveflow.schemas import SafetyResult

# Flags
UNSUPPORTED_MONETARY_CLAIM = "UNSUPPORTED_MONETARY_CLAIM"
UNSUPPORTED_TIMELINE = "UNSUPPORTED_TIMELINE"
UNSUPPORTED_POLICY = "UNSUPPORTED_POLICY"
UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
ACCOUNT_SPECIFIC_CLAIM = "ACCOUNT_SPECIFIC_CLAIM"
MISSING_EVIDENCE = "MISSING_EVIDENCE"
PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
EXCESSIVE_LENGTH = "EXCESSIVE_LENGTH"

_MONEY = re.compile(
    r"(\$\s?\d|\£\s?\d|\€\s?\d|\b\d+\s?(usd|gbp|eur|dollars|pounds)\b|"
    r"\brefund(ed)?\s+(of\s+)?(\$|£|€)?\d)",
    re.I,
)
_TIMELINE = re.compile(
    r"\b(within\s+\d+\s*(business\s+)?(day|hour|minute)s?|"
    r"in\s+\d+\s*(business\s+)?(day|hour)s?|"
    r"\b\d+\s*-\s*\d+\s*(business\s+)?days|"
    r"tomorrow|24\s*hours|next\s+week)\b",
    re.I,
)
_ACTION_DONE = re.compile(
    r"\b(we have|we've|i have|i've)\s+(already\s+)?"
    r"(refunded|cancelled|canceled|processed|issued|escalated|updated|closed|unlocked)\b",
    re.I,
)
_POLICY = re.compile(r"\b(our policy|company policy|we always|guaranteed refund)\b", re.I)
_ACCOUNT_SEE = re.compile(
    r"\b(i can see your account|looking at your account|your (last )?transaction id|"
    r"i('ve| have) checked your (account|order))\b",
    re.I,
)
_INJECTION = re.compile(
    r"(ignore (all )?(previous|prior|above) instructions|system prompt|"
    r"reveal (the )?prompt|mark this as safe|override (the )?policy)",
    re.I,
)


def validate_reply(
    reply: str,
    *,
    customer_message: str = "",
    evidence_text: str = "",
    max_chars: int = 320,
    require_evidence: bool = False,
    has_evidence: bool = True,
) -> SafetyResult:
    flags: list[str] = []
    notes: list[str] = []
    text = reply or ""

    if len(text) > max_chars:
        flags.append(EXCESSIVE_LENGTH)
        notes.append(f"reply length {len(text)} > {max_chars}")

    evidence_blob = (evidence_text or "").lower()
    # Monetary / timeline / action / policy only flagged if not supported in evidence
    if _MONEY.search(text) and not _MONEY.search(evidence_blob):
        flags.append(UNSUPPORTED_MONETARY_CLAIM)
    if _TIMELINE.search(text) and not _TIMELINE.search(evidence_blob):
        flags.append(UNSUPPORTED_TIMELINE)
    if _ACTION_DONE.search(text):
        # completed-action claims are almost never supportable from public tweets
        flags.append(UNSUPPORTED_ACTION)
    if _POLICY.search(text) and not _POLICY.search(evidence_blob):
        flags.append(UNSUPPORTED_POLICY)
    if _ACCOUNT_SEE.search(text):
        flags.append(ACCOUNT_SPECIFIC_CLAIM)

    if require_evidence and not has_evidence:
        flags.append(MISSING_EVIDENCE)

    # Injection attempt in customer message shouldn't appear as compliance in reply
    if _INJECTION.search(customer_message or ""):
        if re.search(r"\b(system prompt|internal instructions|marked as safe)\b", text, re.I):
            flags.append(PROMPT_INJECTION_DETECTED)
        # Always note detection even if reply looks fine
        notes.append("customer_message_contains_injection_patterns")

    if _INJECTION.search(text):
        flags.append(PROMPT_INJECTION_DETECTED)

    return SafetyResult(safe=len(flags) == 0, flags=flags, notes=notes)


def detect_human_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(speak to (a )?(human|person|agent|representative)|"
            r"real (human|person)|talk to (someone|an? agent)|"
            r"customer service (please)?|manager please)\b",
            text or "",
            re.I,
        )
    )


def detect_prompt_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))
