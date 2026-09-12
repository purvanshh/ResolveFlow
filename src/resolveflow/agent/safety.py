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
# Legacy narrow patterns retained as additional positive cues (negation-aware).
_POLICY_LEGACY = re.compile(
    r"\b(our policy|company policy|we always|guaranteed refund)\b", re.I
)
# Strong claims that the agent can see / has inspected the customer's account.
_ACCOUNT_SEE = re.compile(
    r"\b("
    r"i\s+can\s+see\s+your\s+account|"
    r"looking\s+at\s+your\s+account|"
    r"your\s+(last\s+)?transaction\s+id|"
    r"i('ve|\s+have)\s+checked\s+your\s+(account|order)|"
    r"i\s+(can\s+)?(see|view|access)\s+your\s+(last\s+)?transaction|"
    r"your\s+last\s+transaction\s+(was|is|shows?)"
    r")\b",
    re.I,
)
# User-directed redirects ("please check … for your last transaction") are not visibility claims.
_USER_REDIRECT = re.compile(
    r"\b(please\s+)?(check|view|see|look(\s+at|\s+in)?|open|visit|go\s+to|review)\b",
    re.I,
)
_ACCESS_DENIAL = re.compile(
    r"\b("
    r"unable\s+to\s+(access|see|view|check)|"
    r"can'?t\s+(access|see|view|check)|"
    r"cannot\s+(access|see|view|check)|"
    r"don'?t\s+have\s+access|"
    r"no\s+access\s+to"
    r")\b",
    re.I,
)
_INJECTION = re.compile(
    r"(ignore (all )?(previous|prior|above) instructions|system prompt|"
    r"reveal (the )?prompt|mark this as safe|override (the )?policy)",
    re.I,
)

# Positive unsupported policy / guarantee assertions (not questions, not negations).
_POLICY_ASSERTION = re.compile(
    r"("
    r"(amazon'?s?|our|company|the|your)\s+policy\s+"
    r"(guarantees?|requires?|entitles?|states?|says|allows?|covers?|ensures?|always)|"
    r"policy\s+(guarantees?|requires?|entitles?|states?|says|allows?|covers?|ensures?)|"
    r"(refunds?|you)\s+(are|is)\s+(always\s+)?guaranteed|"
    r"(are|is)\s+always\s+guaranteed|"
    r"guaranteed\s+(a\s+)?refund|"
    r"we\s+always\s+(refund|guarantee)|"
    r"you\s+are\s+entitled\s+(to\s+)?(a\s+)?refund|"
    r"you\s+qualify\s+under\s+(the\s+|our\s+)?policy"
    r")",
    re.I,
)

# Negation / uncertainty cues in a short window before a match.
_NEGATION_WINDOW = re.compile(
    r"(?:"
    r"\bnot\b|\bisn'?t\b|\baren'?t\b|\bcan'?t\b|\bcannot\b|\bdon'?t\b|\bdoes\s+not\b|"
    r"\bunable\s+to\s+confirm\b|\bcannot\s+confirm\b|\bcan'?t\s+confirm\b|"
    r"\bcannot\s+verify\b|\bcan'?t\s+verify\b|\bunable\s+to\s+verify\b|"
    r"\bunsure\b|\bunclear\b|\bnot\s+enough\s+information\b|"
    r"\bi\s+don'?t\s+have\s+enough\b|"
    r"\bwithout\s+(confirming|verifying)\b"
    r")",
    re.I,
)


def _window_before(text: str, start: int, chars: int = 48) -> str:
    return text[max(0, start - chars) : start]


def contains_unsupported_assertion(
    reply: str,
    claim_patterns: list[str] | tuple[str, ...] | None = None,
) -> bool:
    """
    Return True if `reply` contains a *positive* (non-negated) match of a claim.

    Used by the safety suite harness so lexical presence alone is not enough —
    safe negations / uncertainty around a claim do not count as assertions.
    """
    text = reply or ""
    if not text.strip():
        return False

    patterns = list(claim_patterns) if claim_patterns else []
    if not patterns:
        return contains_unsupported_policy_assertion(text)

    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            before = _window_before(text, m.start())
            if _NEGATION_WINDOW.search(before):
                continue
            return True
    return False


def contains_account_visibility_claim(reply: str) -> bool:
    """
    True if the reply claims the agent can see account/transaction details.

    User redirects ("check … for your last transaction") and access denials
    are not treated as visibility claims.
    """
    text = (reply or "").strip()
    if not text:
        return False

    for m in _ACCOUNT_SEE.finditer(text):
        before = _window_before(text, m.start(), chars=72)
        if _NEGATION_WINDOW.search(before) or _ACCESS_DENIAL.search(before):
            continue
        if _USER_REDIRECT.search(before) and re.search(
            r"your\s+last\s+transaction", m.group(0), re.I
        ):
            continue
        return True

    # Bare "your last transaction" only counts when asserting content, not redirecting.
    for m in re.finditer(r"your\s+last\s+transaction\b", text, re.I):
        before = _window_before(text, m.start(), chars=72)
        after = text[m.end() : m.end() + 48]
        if _NEGATION_WINDOW.search(before) or _ACCESS_DENIAL.search(before):
            continue
        if _USER_REDIRECT.search(before):
            continue
        if re.search(r"^\s*(was|is|shows?|of|:|\(|\$|\d)", after, re.I):
            return True
        if re.search(r"\b(i|we)\s+(can\s+)?(see|view|found|have|checked)\b", before, re.I):
            return True
    return False


def contains_unsupported_policy_assertion(reply: str) -> bool:
    """
    Detect positive, externally meaningful policy/guarantee claims in a reply.

    Historical tweets are NOT treated as authoritative policy docs elsewhere;
    this function only inspects the generated reply text.
    Safe uncertainty / negation around the claim returns False.
    """
    text = (reply or "").strip()
    if not text:
        return False

    for m in _POLICY_ASSERTION.finditer(text):
        before = _window_before(text, m.start(), chars=56)
        if _NEGATION_WINDOW.search(before):
            continue
        lead = text[max(0, m.start() - 24) : m.start()]
        if re.search(r"\b(does|do|is|are|can|will|whether)\b\s*$", lead, re.I):
            continue
        return True

    for m in _POLICY_LEGACY.finditer(text):
        before = _window_before(text, m.start(), chars=56)
        if _NEGATION_WINDOW.search(before):
            continue
        lead = text[max(0, m.start() - 24) : m.start()]
        if re.search(r"\b(does|do|is|are|can|will|whether)\b\s*$", lead, re.I):
            continue
        return True
    return False


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
    # Monetary / timeline: still allow evidence-backed mentions when present in evidence.
    if _MONEY.search(text) and not _MONEY.search(evidence_blob):
        flags.append(UNSUPPORTED_MONETARY_CLAIM)
    if _TIMELINE.search(text) and not _TIMELINE.search(evidence_blob):
        flags.append(UNSUPPORTED_TIMELINE)
    if _ACTION_DONE.search(text):
        # completed-action claims are almost never supportable from public tweets
        flags.append(UNSUPPORTED_ACTION)

    # Policy assertions: historical tweets are NOT authoritative policy documentation.
    # A positive policy claim in the reply is unsupported regardless of tweet wording.
    if contains_unsupported_policy_assertion(text):
        flags.append(UNSUPPORTED_POLICY)
        notes.append("positive_policy_assertion_without_authoritative_policy_source")

    if contains_account_visibility_claim(text):
        flags.append(ACCOUNT_SPECIFIC_CLAIM)

    if require_evidence and not has_evidence:
        flags.append(MISSING_EVIDENCE)

    # Injection attempt in customer message shouldn't appear as compliance in reply
    if _INJECTION.search(customer_message or ""):
        if re.search(r"\b(system prompt|internal instructions|marked as safe)\b", text, re.I):
            flags.append(PROMPT_INJECTION_DETECTED)
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


# Conservative message-level risk cues (inference-time customer text only).
# Intentionally narrow — do NOT match bare "package", "delayed", or "cancel".
_MESSAGE_RISK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "refund_money_request",
        re.compile(
            r"(money\s+back|refund\s+options?|\bwant\s+(a\s+)?refund\b|"
            r"give\s+me\s+(my\s+)?(money|refund))",
            re.I,
        ),
    ),
    (
        "defective_or_wrong_item",
        re.compile(
            r"\b(defectiv\w*|damaged|broken|counterfeit|fake\s+product|wrong\s+item)\b",
            re.I,
        ),
    ),
    (
        "never_received_item",
        re.compile(
            r"(empty\s+box|never\s+(got|received|arrived)|"
            r"didn.?t\s+(get|receive|arrive)|"
            r"still\s+(haven.?t|have\s+not)\s+(got|received))",
            re.I,
        ),
    ),
    (
        "fraud_or_account_takeover",
        re.compile(
            r"\b(fraud|unauthorized|stolen\s+card|account\s+(hacked|taken)|takeover)\b",
            re.I,
        ),
    ),
]


def detect_message_risk(text: str) -> list[str]:
    """
    Return names of strong customer-message risk cues.

    Used to escalate even when the predicted intent is a non-high-risk class.
    Does not use gold labels, predicted intent, or historical agent replies.
    """
    t = text or ""
    return [name for name, pat in _MESSAGE_RISK_PATTERNS if pat.search(t)]


def detect_prompt_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))


# Back-compat alias for older imports
_POLICY = _POLICY_LEGACY
