"""Rule-assisted annotator helpers (not a substitute for guideline review)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Label:
    intent: str
    escalation_expected: str
    escalation_reason: str
    difficulty: str
    annotation_notes: str = ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def annotate_example(input_text: str, context: str = "") -> Label:
    """
    Assign primary intent following configs/intents.yaml boundaries.

    This encodes the annotation guide as deterministic checks for solo
    annotation throughput; ambiguous cases should be reviewed and may be
    overridden to other_unclear.
    """
    text = _norm(input_text)
    ctx = _norm(context)
    blob = f"{ctx}\n{text}".strip()

    # Extremely thin messages without disambiguating context
    thin = (
        len(text) <= 12
        or text in {"ok", "okay", "yes", "no", "nope", "done", "thanks", "thank you", "hi", "hello", "pls", "please"}
        or text in {"[url]", "url"}
        or re.fullmatch(r"\[url\]", text) is not None
    )
    ctx_useful = len(ctx) > 40 and not re.fullmatch(r"(customer|brand):.{0,20}", ctx)
    if thin and not ctx_useful:
        return Label(
            "other_unclear",
            "escalate",
            "Insufficient evidence to choose a safe auto-handle intent.",
            "hard",
            "thin message",
        )

    # Gratitude / acknowledgement only
    if re.fullmatch(r"(thanks|thank you|thx|ty|merci|gracias)[!\. ]*", text):
        return Label("other_unclear", "auto_handle", "", "easy", "acknowledgement only")

    def hit(pat: str, s: str = blob) -> bool:
        return re.search(pat, s, flags=re.I) is not None

    # Account access
    if hit(r"can'?t log|cannot log|locked (out|my account|account)|account (locked|suspended|closed|compromised)|hacked|reset (my )?password|two.?factor|otp"):
        return Label(
            "account_access",
            "escalate",
            "Account security/access requires identity verification.",
            "medium" if len(text) > 40 else "hard",
        )

    # Technical
    if hit(r"app (keeps )?(crash|crashing|crashed)|website (is )?(down|broken|impossible)|site (error|down)|currently unavailable|glitch|checkout (button|won'?t|doesn'?t)|navigate|browser|bug\b"):
        # but if clearly about faulty item email via site, still technical_issue for site UX
        return Label("technical_issue", "auto_handle", "", "medium")

    # Package missing / misdelivered (before generic delay)
    if hit(
        r"marked (as )?delivered|says? delivered|showing delivered|delivered but|never (arrived|received|got)|didn'?t (arrive|receive|get)|not (received|delivered|here|arrive)|wrong (house|address|door)|stolen|porch pirate|left (it )?(at|with)|misdeliver|no package|nothing (arrived|here|at)"
    ):
        return Label(
            "package_missing_or_misdelivered",
            "escalate",
            "Missing/misdelivered package needs order-specific investigation.",
            "medium",
        )

    # Refund status before refund request
    if hit(r"where('?s| is) (my )?(refund|money)|refund (hasn'?t|has not|not) (arrived|been|processed|received|come)|waiting (for )?(my )?(refund|money)|refund (pending|status)|still (haven'?t|waiting).{0,40}refund"):
        return Label(
            "refund_status",
            "escalate",
            "Refund timing/amount is account- and transaction-specific.",
            "medium",
        )

    # Payment / billing
    if hit(r"charged twice|double charg|billed twice|payment failed|card (was )?declined|unauthori[sz]ed|unrecognised charge|unrecognized charge|amazon pay|gift ?card|promo code|billing|overcharg"):
        return Label(
            "payment_billing",
            "escalate",
            "Payment/billing disputes need account verification.",
            "medium",
        )

    # Cancellation
    if hit(r"cancel(l)?(ing|led)? (my |the |this )?order|please cancel|want( to)? cancel|cancel (it|order)|don'?t (want|send).{0,30}(order|deliver)"):
        return Label("cancellation_request", "auto_handle", "", "easy")

    # Return
    if hit(r"\breturn(ing|ed)?\b.{0,40}\b(item|product|order|label|pickup)\b|\breturn label\b|schedule (a )?pickup|send( it)? back"):
        # if clearly asking where refund after return -> refund_status already caught
        return Label("return_request", "auto_handle", "", "easy")

    # Refund request
    if hit(r"(want|need|please|get|give).{0,20}refund|money back|refund (me|this|my|the)"):
        return Label(
            "refund_request",
            "escalate",
            "Issuing a refund is a financial action needing order verification.",
            "easy",
        )

    # Order quality
    if hit(r"broken|damaged|defective|wrong (item|product)|counterfeit|fake product|missing (part|pieces|item|accessory)|empty box|faulty"):
        return Label(
            "order_quality_issue",
            "escalate",
            "Quality claims need order-specific evidence review.",
            "medium",
        )

    # Prime membership (not mere prime shipping delay)
    if hit(r"cancel (amazon )?prime|prime (membership|subscription)|charged for prime|prime benefit|renew(ed|al)? prime|prime trial"):
        return Label("prime_membership", "auto_handle", "", "medium")

    # Delivery delay
    if hit(r"delay|delayed|late|still hasn'?t (shipped|arrived)|expected (delivery|by)|next[- ]day|one[- ]day delivery|guaranteed delivery|not (here|arrived) yet|shipping (slow|late)|promised"):
        return Label("delivery_delay", "auto_handle", "", "easy")

    # Follow-ups that only complain about no response
    if hit(r"no (response|reply|help)|any update|still waiting|customer service") and len(text) < 80:
        return Label(
            "other_unclear",
            "escalate",
            "Follow-up without recoverable primary issue in message/context.",
            "hard",
            "support chase without clear issue",
        )

    # Default
    difficulty = "hard" if len(text) < 40 or len(text) > 280 else "medium"
    return Label(
        "other_unclear",
        "escalate",
        "No single defined intent clearly supported.",
        difficulty,
        "fallback",
    )
