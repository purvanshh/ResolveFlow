"""Deterministic mock LLM for offline tests (not for final reported eval)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from resolveflow.llm.base import LLMProvider


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, intents: list[str] | None = None):
        self.intents = intents or ["other_unclear"]

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema_hint: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        blob = (system + "\n" + user).lower()
        # Classifier path
        if "assign exactly one intent" in system.lower() or "intent taxonomy" in blob:
            intent = self._guess_intent(user)
            return {
                "intent": intent,
                "confidence": 0.72 if intent != "other_unclear" else 0.41,
                "reasoning_signals": ["mock_keyword_match"],
            }
        # Responder path
        if "draft" in system.lower() or "response drafting" in system.lower():
            reply = (
                "Thanks for contacting us. Based on similar cases, please reach out "
                "through the appropriate support channel so we can review the details. "
                "We cannot confirm account-specific actions from this channel."
            )
            if "charged twice" in user.lower() or "duplicate" in user.lower():
                reply = (
                    "Sorry about the duplicate charge. We can't verify the transaction "
                    "from here — please contact support through the available channel "
                    "so the payment can be reviewed."
                )
            return {"reply": reply, "text": reply}
        # Generic
        digest = hashlib.sha1(user.encode()).hexdigest()[:8]
        return {"text": f"mock_response_{digest}", "reply": f"mock_response_{digest}"}

    def _guess_intent(self, user: str) -> str:
        t = user.lower()
        rules = [
            (r"charged twice|duplicate charg|payment failed|gift ?card|amazon pay", "payment_billing"),
            (r"where.*(refund|money)|refund (hasn|not|pending)", "refund_status"),
            (r"want (a )?refund|money back|please refund", "refund_request"),
            (r"marked (as )?delivered|says delivered|never (arrived|received)|wrong (house|address)", "package_missing_or_misdelivered"),
            (r"delay|delayed|late|next.?day", "delivery_delay"),
            (r"broken|damaged|wrong (item|product)|faulty", "order_quality_issue"),
            (r"cancel (my |the )?order|please cancel", "cancellation_request"),
            (r"return label|want to return|schedule .{0,10}pickup", "return_request"),
            (r"prime (membership|subscription)|cancel prime", "prime_membership"),
            (r"can.?t log|hacked|password|account (locked|frozen)", "account_access"),
            (r"app (crash|crashed)|website|currently unavailable", "technical_issue"),
            (r"speak to (a )?human|real person|customer service agent", "other_unclear"),
        ]
        for pat, intent in rules:
            if re.search(pat, t) and intent in self.intents:
                return intent
        return "other_unclear" if "other_unclear" in self.intents else self.intents[0]
