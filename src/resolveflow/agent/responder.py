"""Grounded reply generation."""

from __future__ import annotations

import re

from resolveflow.llm.base import LLMProvider
from resolveflow.prompts.responder import build_responder_system, build_responder_user
from resolveflow.prompts.versions import RESPONDER_VERSION
from resolveflow.retrieval.evidence import format_evidence
from resolveflow.schemas import RetrievedEvidence


class Responder:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        brand: str,
        max_chars: int = 280,
        mode: str = "llm",  # llm | grounded_template
    ):
        self.provider = provider
        self.brand = brand
        self.max_chars = max_chars
        self.mode = mode

    def draft(
        self,
        *,
        message: str,
        context: str,
        intent: str,
        evidence: list[RetrievedEvidence],
    ) -> str:
        if self.mode == "grounded_template":
            return self._template_draft(message=message, intent=intent, evidence=evidence)

        evidence_block = format_evidence(
            [
                # adapt schema objects to RetrievedCase-like via duck typing in format_evidence
                _EvidenceAdapter(e)
                for e in evidence
            ]
            if evidence
            else []
        )
        system = build_responder_system(self.brand, max_chars=self.max_chars)
        user = build_responder_user(
            message=message,
            context=context,
            intent=intent,
            evidence_block=evidence_block,
        )
        data = self.provider.generate_structured(
            system=system,
            user=user,
            schema_hint={"reply": "string"},
            temperature=0.0,
        )
        reply = str(data.get("reply") or data.get("text") or "").strip()
        if not reply:
            reply = self._template_draft(message=message, intent=intent, evidence=evidence)
        reply = _ensure_length(reply, self.max_chars)
        return reply

    def _template_draft(
        self,
        *,
        message: str,
        intent: str,
        evidence: list[RetrievedEvidence],
    ) -> str:
        """
        Offline grounded drafter: uses intent + evidence patterns without inventing
        refunds/timelines/actions. Used when LLM API unavailable / ablation.
        """
        asks_dm = any(
            re.search(r"\b(dm|direct message|contact us|phone or chat|reach out)\b", e.historical_response, re.I)
            for e in evidence
        )
        if intent in {
            "payment_billing",
            "refund_request",
            "refund_status",
            "account_access",
            "package_missing_or_misdelivered",
        }:
            base = (
                "Sorry you're dealing with this. We can't verify account or payment details "
                "from this channel. Please contact us through the available support channel "
                "so the case can be reviewed securely."
            )
        elif intent == "delivery_delay":
            base = (
                "Sorry for the delay with your delivery. Please check the latest tracking "
                "update in your order details, and contact support if the status looks wrong "
                "so we can look into it."
            )
        elif intent in {"cancellation_request", "return_request"}:
            base = (
                "Thanks for reaching out. Please use your order's cancellation/return options "
                "in your account, or contact support if those options aren't available for this order."
            )
        elif intent == "technical_issue":
            base = (
                "Sorry you're hitting a technical issue. Please try again on another device/browser "
                "if you can, and contact support with a screenshot if it continues."
            )
        elif intent == "prime_membership":
            base = (
                "Thanks for the note about Prime. Please review your membership settings in your "
                "account, or contact support so they can check the subscription details with you."
            )
        elif intent == "order_quality_issue":
            base = (
                "Sorry your order had a quality issue. Please contact support with the order details "
                "and photos if available so the team can review replacement/return options."
            )
        else:
            base = (
                "Thanks for contacting us. We don't have enough clear details here to resolve this "
                "safely. Please reply with more information or contact support so a specialist can help."
            )
        if asks_dm and intent != "other_unclear":
            base = base.rstrip(".") + ". A private support channel is best for account-specific help."
        return _ensure_length(base, self.max_chars)


class _EvidenceAdapter:
    """Duck-type adapter for format_evidence."""

    def __init__(self, e: RetrievedEvidence):
        self.customer_message = e.customer_issue
        self.resolution_summary = e.resolution_summary or e.historical_response
        self.brand_response = e.historical_response
        self.intent = e.intent
        self.similarity = e.similarity


def _ensure_length(text: str, max_chars: int) -> str:
    t = " ".join(text.split()).strip()
    if len(t) <= max_chars:
        return t
    # Prefer sentence boundary
    cut = t[: max_chars - 1]
    if "." in cut:
        cut = cut[: cut.rfind(".") + 1]
    else:
        cut = cut.rsplit(" ", 1)[0] + "."
    return cut.strip()
